"""
Reranker — Cohere primary, flashrank fallback.

Cohere rerank-english-v3.0 significantly outperforms local T5-flan on
precision@k for domain-specific financial retrieval.
"""

import time
import cohere
from cohere.errors import TooManyRequestsError
from flashrank import Ranker, RerankRequest

from config import COHERE_API_KEY, RERANK_MODEL, TOP_K_RERANK, DISABLE_COHERE

_cohere: cohere.ClientV2 | None = None
if COHERE_API_KEY and not DISABLE_COHERE:
    _cohere = cohere.ClientV2(api_key=COHERE_API_KEY)

_flashrank: Ranker | None = None

# Phase 1: exponential backoff (10s, 20s, 40s, 80s)
_COHERE_EXP_RETRIES = 4
_COHERE_RETRY_BASE = 10
# Phase 2: after exponential retries exhausted, keep retrying every 60s
_COHERE_STEADY_WAIT = 60


def _get_flashrank() -> Ranker:
    global _flashrank
    if _flashrank is None:
        _flashrank = Ranker(model_name=RERANK_MODEL)
    return _flashrank


def rerank(query: str, candidates: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
    if not candidates:
        return []
    if _cohere is not None:
        return _rerank_cohere(query, candidates, top_k)
    return _rerank_flashrank(query, candidates, top_k)


def _rerank_cohere(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    docs = [c["payload"]["text"] for c in candidates]
    attempt = 0
    while True:
        try:
            response = _cohere.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=docs,
                top_n=top_k,
            )
            return [
                {**candidates[hit.index], "rerank_score": hit.relevance_score}
                for hit in response.results
            ]
        except TooManyRequestsError:
            if attempt < _COHERE_EXP_RETRIES:
                wait = _COHERE_RETRY_BASE * (2 ** attempt)
            else:
                wait = _COHERE_STEADY_WAIT
            print(f"  [reranker] Cohere rate limit, waiting {wait}s (attempt {attempt + 1})...")
            time.sleep(wait)
            attempt += 1
        except Exception:
            raise


def _rerank_flashrank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    passages = [{"id": i, "text": c["payload"]["text"]} for i, c in enumerate(candidates)]
    request = RerankRequest(query=query, passages=passages)
    results = _get_flashrank().rerank(request)
    return [
        {**candidates[r["id"]], "rerank_score": r["score"]}
        for r in results[:top_k]
    ]
