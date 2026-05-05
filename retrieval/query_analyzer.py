"""
Analyzes a user query to extract:
  - ticker symbols mentioned
  - filing types (10-K, 10-Q, 8-K)
  - whether the query is multi-part (needs decomposition)
  - sub-questions if multi-part
"""

import json
import re
from anthropic import Anthropic
from config import ANTHROPIC_API_KEY, LLM_MODEL, TICKER_TO_COMPANY

_client = Anthropic(api_key=ANTHROPIC_API_KEY)

_ALL_TICKERS = set(TICKER_TO_COMPANY.keys())
_COMPANY_TO_TICKER = {v.lower(): k for k, v in TICKER_TO_COMPANY.items()}

_SYSTEM = """You are a financial query analyzer. Given a user question about SEC filings, output a JSON object with:
- "tickers": list of company ticker symbols explicitly or implicitly referenced (e.g. ["MSFT", "GOOG"]). Empty list if none.
- "filing_types": list of filing types needed, subset of ["10-K", "10-Q", "8-K"]. Empty list means search all.
- "is_complex": true if the query has multiple distinct sub-questions or requires comparing 2+ companies.
- "sub_queries": if is_complex=true, list of focused sub-questions to answer independently. Otherwise empty list.

Known tickers: """ + ", ".join(sorted(_ALL_TICKERS)) + """

Output only valid JSON, no markdown fences."""


def analyze_query(query: str) -> dict:
    response = _client.messages.create(
        model=LLM_MODEL,
        max_tokens=512,
        system=_SYSTEM,
        messages=[{"role": "user", "content": query}],
    )
    raw = response.content[0].text.strip()

    # Strip markdown fences if the model adds them anyway
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"tickers": [], "filing_types": [], "is_complex": False, "sub_queries": []}

    # Validate tickers against known set
    result["tickers"] = [t.upper() for t in result.get("tickers", []) if t.upper() in _ALL_TICKERS]
    result["filing_types"] = [ft for ft in result.get("filing_types", []) if ft in {"10-K", "10-Q", "8-K"}]
    result.setdefault("is_complex", False)
    result.setdefault("sub_queries", [])

    # If complex but no sub_queries provided, fall back to original query
    if result["is_complex"] and not result["sub_queries"]:
        result["sub_queries"] = [query]

    return result
