from dataclasses import dataclass
import tiktoken
from anthropic import Anthropic, BadRequestError
from config import ANTHROPIC_API_KEY, LLM_MODEL

_client = Anthropic(api_key=ANTHROPIC_API_KEY)
_enc = tiktoken.get_encoding("cl100k_base")

# Leave headroom for system prompt + answer
_MAX_CONTEXT_TOKENS = 140_000

_SYSTEM = """You are a financial analyst assistant specialising in SEC filings.

Rules:
1. Answer ONLY from the provided context. Do not use outside knowledge.
2. After every factual claim, cite the source inline using this exact format: [TICKER, FILING_TYPE, DATE, Page N]
   Example: Apple's revenue was $391B [AAPL, 10-K, 2024-11-01, Page 45].
3. If the context is insufficient, say: "The provided filings do not contain enough information to answer this question."
4. Be precise with numbers — quote exact figures from the filings.
5. For cross-company comparisons, address each company separately before comparing."""


@dataclass
class Citation:
    ticker: str
    company_name: str
    filing_type: str
    filing_date: str
    filename: str
    page_num: int
    excerpt: str


@dataclass
class GeneratorResponse:
    answer: str
    citations: list[Citation]


def _format_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    total_tokens = 0
    for i, chunk in enumerate(chunks, 1):
        p = chunk["payload"]
        part = (
            f"[Source {i}] {p['ticker']} {p['filing_type']} {p['filing_date']} Page {p['page_num']}\n"
            f"{p['parent_text']}"
        )
        part_tokens = len(_enc.encode(part))
        if total_tokens + part_tokens > _MAX_CONTEXT_TOKENS:
            break
        parts.append(part)
        total_tokens += part_tokens
    return "\n\n---\n\n".join(parts)


def generate(query: str, chunks: list[dict]) -> GeneratorResponse:
    context = _format_context(chunks)
    user_msg = f"Context from SEC filings:\n\n{context}\n\nQuestion: {query}"

    try:
        response = _client.messages.create(
            model=LLM_MODEL,
            max_tokens=2048,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
    except BadRequestError:
        # Retry with fewer chunks (top half only)
        context = _format_context(chunks[: max(1, len(chunks) // 2)])
        user_msg = f"Context from SEC filings:\n\n{context}\n\nQuestion: {query}"
        response = _client.messages.create(
            model=LLM_MODEL,
            max_tokens=2048,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

    answer = response.content[0].text.strip()

    seen: set[tuple] = set()
    citations: list[Citation] = []
    for chunk in chunks:
        p = chunk["payload"]
        key = (p["filename"], p["page_num"])
        if key not in seen:
            seen.add(key)
            citations.append(Citation(
                ticker=p["ticker"],
                company_name=p["company_name"],
                filing_type=p["filing_type"],
                filing_date=p["filing_date"],
                filename=p["filename"],
                page_num=p["page_num"],
                excerpt=p["parent_text"][:500],   # increased from 200 for better faithfulness judging
            ))

    return GeneratorResponse(answer=answer, citations=citations)
