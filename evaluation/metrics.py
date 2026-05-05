"""
Evaluation metrics.

Retriever : Precision@k, Recall@k, F1@k  (LLM-as-judge for relevance)
Generator : Faithfulness                  (LLM-as-judge)
Answer    : Accuracy                      (LLM-as-judge vs ground truth)
"""

import re
from anthropic import Anthropic
from config import ANTHROPIC_API_KEY, LLM_MODEL

_client = Anthropic(api_key=ANTHROPIC_API_KEY)


# ── Shared judge ──────────────────────────────────────────────────────────────

def _score(prompt: str) -> float:
    """Ask Claude to score 0-10, return normalised 0-1."""
    resp = _client.messages.create(
        model=LLM_MODEL,
        max_tokens=16,
        messages=[{"role": "user", "content": prompt}],
    )
    nums = re.findall(r"\d+", resp.content[0].text)
    return min(int(nums[0]), 10) / 10.0 if nums else 0.5


# ── Retrieval metrics (LLM judges relevance of each chunk) ────────────────────

def judge_chunk_relevance(query: str, chunks: list[dict]) -> list[bool]:
    """
    Single batched LLM call: which chunks are genuinely useful for answering the query?
    Returns a bool list aligned with chunks.
    """
    if not chunks:
        return []

    numbered = "\n\n".join(
        f"[{i+1}] {c['payload']['text'][:350]}"
        for i, c in enumerate(chunks)
    )
    prompt = (
        f"Query: {query}\n\n"
        f"Chunks (1-{len(chunks)}):\n{numbered}\n\n"
        "List the numbers of chunks that are genuinely useful for answering the query. "
        "Comma-separated integers only, e.g. '1, 3, 5'. If none, reply '0'."
    )
    resp = _client.messages.create(
        model=LLM_MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    relevant = {int(n) for n in re.findall(r"\d+", text) if 1 <= int(n) <= len(chunks)}
    return [i + 1 in relevant for i in range(len(chunks))]


def retrieval_metrics(
    query: str,
    retrieved_chunks: list[dict],
    candidate_chunks: list[dict] | None = None,
) -> dict[str, float]:
    """
    Precision@k  = relevant in reranked top-k / k
    Recall@k     = relevant in reranked top-k / relevant in pre-rerank candidate pool
    F1@k         = harmonic mean

    candidate_chunks is the pre-rerank pool (TOP_K_RETRIEVAL).
    If omitted, recall falls back to the old approximation (recall = precision).
    """
    if not retrieved_chunks:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "k": 0, "relevant": 0}

    k = len(retrieved_chunks)
    retrieved_ids = {c["id"] for c in retrieved_chunks}

    if candidate_chunks and len(candidate_chunks) > k:
        # Judge the full candidate pool in one call
        relevance_candidates = judge_chunk_relevance(query, candidate_chunks)
        relevant_candidate_ids = {
            candidate_chunks[i]["id"]
            for i, rel in enumerate(relevance_candidates) if rel
        }
        n_relevant_retrieved = len(retrieved_ids & relevant_candidate_ids)
        n_relevant_total = len(relevant_candidate_ids)

        precision = n_relevant_retrieved / k
        recall = (n_relevant_retrieved / n_relevant_total) if n_relevant_total > 0 else 0.0
    else:
        # Fallback: judge only retrieved chunks
        relevance = judge_chunk_relevance(query, retrieved_chunks)
        n_relevant_retrieved = sum(relevance)
        n_relevant_total = n_relevant_retrieved
        precision = n_relevant_retrieved / k
        recall = precision  # can't distinguish without wider pool

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "k": k,
        "relevant": n_relevant_retrieved,
        "relevant_in_pool": n_relevant_total,
    }


# ── Generator metrics ─────────────────────────────────────────────────────────

def faithfulness(answer: str, context: str) -> float:
    """Does every claim in the answer trace back to the retrieved context? (0-1)"""
    if not context.strip() or answer.startswith("ERROR:"):
        return None
    return _score(
        "Evaluate whether the AI answer is faithful to the source context.\n\n"
        f"Context:\n{context[:4000]}\n\n"
        f"Answer:\n{answer}\n\n"
        "Score 0-10: 10 = every claim is supported by context, 0 = major hallucinations.\n"
        "Reply with a single integer."
    )


def answer_accuracy(answer: str, ground_truth: str) -> float:
    """
    Semantic accuracy vs ground truth (0-1).
    Accuracy = how completely and correctly the answer matches the expected answer.
    Requires manually written ground_truth in test_set.json.
    """
    if not ground_truth or "TBD" in ground_truth or answer.startswith("ERROR:"):
        return None
    return _score(
        "Compare the AI answer to the ground truth.\n\n"
        f"Ground truth:\n{ground_truth}\n\n"
        f"AI answer:\n{answer}\n\n"
        "Score 0-10: 10 = correct and complete, 0 = wrong or missing key facts.\n"
        "Reply with a single integer."
    )
