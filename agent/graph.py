"""
Agentic RAG graph (LangGraph).

Flow:
  START → query_analyzer → retriever → verifier → [pass: generator → END | fail (≤2 retries): retriever]
"""

import operator
from dataclasses import dataclass, field
from typing import TypedDict, Annotated

from anthropic import Anthropic
from langgraph.graph import StateGraph, START, END

from config import ANTHROPIC_API_KEY, LLM_MODEL, TOP_K_RETRIEVAL, TOP_K_RERANK, MAX_CHUNKS_TOTAL, MAX_SELF_CORRECT_RETRIES, DISABLE_HYDE
from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import rerank
from retrieval.query_analyzer import analyze_query
from retrieval.hyde import generate_hypothetical_doc
from generation.generator import generate, GeneratorResponse


# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    query: str
    filters: dict
    sub_queries: list[str]
    retrieved_chunks: list[dict]
    candidate_chunks: list[dict]   # pre-rerank pool — used for recall computation
    verification_passed: bool
    retry_count: int
    response: GeneratorResponse | None
    trace: Annotated[list[dict], operator.add]   # accumulates across all nodes


@dataclass
class AgentResult:
    response: GeneratorResponse
    trace: list[dict] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    candidate_chunks: list[dict] = field(default_factory=list)  # pre-rerank pool


# ── Singletons ────────────────────────────────────────────────────────────────

_searcher: HybridSearcher | None = None
_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)


def _get_searcher() -> HybridSearcher:
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher()
    return _searcher


def reset_searcher() -> None:
    """Close and release the Qdrant lock — call before running ingestion."""
    global _searcher
    if _searcher is not None:
        try:
            _searcher.client.close()
        except Exception:
            pass
        _searcher = None


# ── Nodes ─────────────────────────────────────────────────────────────────────

def query_analyzer_node(state: RAGState) -> dict:
    analysis = analyze_query(state["query"])
    filters = {
        "tickers": analysis["tickers"],
        "filing_types": analysis["filing_types"],
    }
    sub_queries = analysis["sub_queries"] if analysis["is_complex"] else [state["query"]]
    return {
        "filters": filters,
        "sub_queries": sub_queries,
        "trace": [{
            "step": "query_analysis",
            "tickers": analysis["tickers"],
            "filing_types": analysis["filing_types"],
            "is_complex": analysis["is_complex"],
            "sub_queries": sub_queries,
        }],
    }


def retriever_node(state: RAGState) -> dict:
    searcher = _get_searcher()
    all_chunks: list[dict] = []
    all_candidates: list[dict] = []
    seen_ids: set[str] = set()
    seen_candidate_ids: set[str] = set()
    per_query_counts: dict[str, int] = {}

    for sub_q in state["sub_queries"]:
        hyde_doc = None if DISABLE_HYDE else generate_hypothetical_doc(sub_q)
        candidates = searcher.search(sub_q, state["filters"], top_k=TOP_K_RETRIEVAL, hyde_doc=hyde_doc)
        # accumulate unique candidates for recall denominator
        for c in candidates:
            if c["id"] not in seen_candidate_ids:
                seen_candidate_ids.add(c["id"])
                all_candidates.append(c)
        reranked = rerank(sub_q, candidates, top_k=TOP_K_RERANK)
        count = 0
        for chunk in reranked:
            if chunk["id"] not in seen_ids:
                seen_ids.add(chunk["id"])
                all_chunks.append(chunk)
                count += 1
        per_query_counts[sub_q] = count

    existing = state.get("retrieved_chunks", [])
    for chunk in existing:
        if chunk["id"] not in seen_ids:
            seen_ids.add(chunk["id"])
            all_chunks.append(chunk)

    # Cap total chunks: scale up for multi-company queries so each company gets representation
    n_tickers = len(state.get("filters", {}).get("tickers", []))
    effective_cap = MAX_CHUNKS_TOTAL if n_tickers <= 2 else min(MAX_CHUNKS_TOTAL * n_tickers // 2, 30)
    all_chunks = all_chunks[:effective_cap]

    return {
        "retrieved_chunks": all_chunks,
        "candidate_chunks": all_candidates,
        "trace": [{
            "step": "retrieval",
            "attempt": state.get("retry_count", 0) + 1,
            "sub_queries": state["sub_queries"],
            "chunks_per_query": per_query_counts,
            "total_chunks": len(all_chunks),
        }],
    }


def verifier_node(state: RAGState) -> dict:
    if state.get("retry_count", 0) >= MAX_SELF_CORRECT_RETRIES:
        return {
            "verification_passed": True,
            "retry_count": state.get("retry_count", 0),
            "trace": [{"step": "verification", "passed": True, "reason": "max retries reached — proceeding"}],
        }

    context_preview = "\n\n".join(
        c["payload"]["text"][:300] for c in state["retrieved_chunks"][:5]
    )
    prompt = (
        f"Query: {state['query']}\n\n"
        f"Retrieved context (preview):\n{context_preview}\n\n"
        "Does the retrieved context contain enough information to answer the query? "
        "Reply with exactly one word: YES or NO."
    )
    response = _anthropic.messages.create(
        model=LLM_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = response.content[0].text.strip().upper()
    passed = "YES" in verdict
    retry_count = state.get("retry_count", 0) + (0 if passed else 1)

    if not passed:
        broadened = [f"{q} overview financial performance" for q in state["sub_queries"]]
        return {
            "verification_passed": False,
            "retry_count": retry_count,
            "sub_queries": broadened,
            "trace": [{"step": "verification", "passed": False, "reason": "context insufficient — broadening queries", "broadened_queries": broadened}],
        }

    return {
        "verification_passed": True,
        "retry_count": retry_count,
        "trace": [{"step": "verification", "passed": True, "reason": "context sufficient"}],
    }


def generator_node(state: RAGState) -> dict:
    resp = generate(state["query"], state["retrieved_chunks"])
    return {
        "response": resp,
        "trace": [{"step": "generation", "chunks_used": len(state["retrieved_chunks"]), "citations": len(resp.citations)}],
    }


# ── Conditional edges ─────────────────────────────────────────────────────────

def after_verifier(state: RAGState) -> str:
    return "generate" if state["verification_passed"] else "retrieve"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(RAGState)
    g.add_node("query_analyzer", query_analyzer_node)
    g.add_node("retriever", retriever_node)
    g.add_node("verifier", verifier_node)
    g.add_node("generator", generator_node)
    g.add_edge(START, "query_analyzer")
    g.add_edge("query_analyzer", "retriever")
    g.add_edge("retriever", "verifier")
    g.add_conditional_edges("verifier", after_verifier, {"generate": "generator", "retrieve": "retriever"})
    g.add_edge("generator", END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Public API ────────────────────────────────────────────────────────────────

def _initial_state(user_query: str) -> RAGState:
    return {
        "query": user_query,
        "filters": {},
        "sub_queries": [],
        "retrieved_chunks": [],
        "candidate_chunks": [],
        "verification_passed": False,
        "retry_count": 0,
        "response": None,
        "trace": [],
    }


def query(user_query: str) -> AgentResult:
    graph = get_graph()
    final_state = graph.invoke(_initial_state(user_query))
    return AgentResult(
        response=final_state["response"],
        trace=final_state["trace"],
        retrieved_chunks=final_state["retrieved_chunks"],
        candidate_chunks=final_state["candidate_chunks"],
    )


def stream_query(user_query: str):
    """Yield (node_name, state_update) as each node completes."""
    graph = get_graph()
    for chunk in graph.stream(_initial_state(user_query)):
        for node_name, node_output in chunk.items():
            yield node_name, node_output
