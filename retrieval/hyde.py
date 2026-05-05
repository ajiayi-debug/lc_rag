"""
HyDE — Hypothetical Document Embeddings.

Instead of embedding the raw query, generate a short hypothetical SEC filing
excerpt that would answer the question, then embed that. Document-style language
matches the embedding space better than question-style language, improving recall
for balance sheets, tables, and specific financial metrics.

BM25 still uses the original query (keyword matching benefits from query terms).
"""

from anthropic import Anthropic
from config import ANTHROPIC_API_KEY, LLM_MODEL

_client = Anthropic(api_key=ANTHROPIC_API_KEY)

_SYSTEM = (
    "You are an expert financial analyst who has read thousands of SEC filings. "
    "Generate realistic SEC filing excerpts using precise financial terminology."
)


def generate_hypothetical_doc(query: str) -> str:
    """
    Generate a short hypothetical SEC filing excerpt that would directly answer
    the query. The excerpt is embedded (not shown to the user) to find real chunks.
    """
    prompt = (
        f"Question: {query}\n\n"
        "Write a 2-4 sentence excerpt from an SEC filing (10-K, 10-Q, or 8-K) that "
        "would directly answer this question. Use specific numbers, filing-style language, "
        "and financial terminology. If the question involves multiple companies, write "
        "one sentence per company. Output only the excerpt, no preamble."
    )
    resp = _client.messages.create(
        model=LLM_MODEL,
        max_tokens=150,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()
