import pickle

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from config import COLLECTION_NAME, QDRANT_PATH, BM25_INDEX_PATH, TOP_K_RETRIEVAL
from ingestion.embedder import embed_query, embed_texts


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridSearcher:
    def __init__(self):
        self.client = QdrantClient(path=QDRANT_PATH)
        self._load_bm25()

    def _load_bm25(self) -> None:
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.bm25_ids: list[str] = data["ids"]
        self.ticker_to_indices: dict[str, list[int]] = data["ticker_to_indices"]
        self.filing_type_to_indices: dict[str, list[int]] = data["filing_type_to_indices"]

    def _qdrant_filter(self, filters: dict) -> Filter | None:
        conditions = []
        if filters.get("tickers"):
            conditions.append(FieldCondition(key="ticker", match=MatchAny(any=filters["tickers"])))
        if filters.get("filing_types"):
            conditions.append(FieldCondition(key="filing_type", match=MatchAny(any=filters["filing_types"])))
        return Filter(must=conditions) if conditions else None

    def _valid_mask(self, filters: dict) -> np.ndarray:
        n = len(self.bm25_ids)
        mask = np.ones(n, dtype=bool)
        if filters.get("tickers"):
            ticker_mask = np.zeros(n, dtype=bool)
            for t in filters["tickers"]:
                for idx in self.ticker_to_indices.get(t, []):
                    ticker_mask[idx] = True
            mask &= ticker_mask
        if filters.get("filing_types"):
            ft_mask = np.zeros(n, dtype=bool)
            for ft in filters["filing_types"]:
                for idx in self.filing_type_to_indices.get(ft, []):
                    ft_mask[idx] = True
            mask &= ft_mask
        return mask

    def dense_search(self, query: str, filters: dict, top_k: int, hyde_doc: str | None = None) -> list[dict]:
        # HyDE: embed the hypothetical document instead of the raw query for dense search
        vec = embed_texts([hyde_doc], input_type="document")[0] if hyde_doc else embed_query(query)
        q_filter = self._qdrant_filter(filters)
        result = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            query_filter=q_filter,
            limit=top_k,
            with_payload=True,
        )
        return [{"id": str(h.id), "score": h.score, "payload": h.payload} for h in result.points]

    def bm25_search(self, query: str, filters: dict, top_k: int) -> list[dict]:
        scores = np.array(self.bm25.get_scores(_tokenize(query)))

        # Apply filter mask
        mask = self._valid_mask(filters)
        scores = scores * mask

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            chunk_id = self.bm25_ids[idx]
            # Fetch payload from Qdrant (batch would be ideal but one-by-one is fine for top_k)
            points = self.client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[chunk_id],
                with_payload=True,
            )
            if points:
                results.append({"id": chunk_id, "score": float(scores[idx]), "payload": points[0].payload})
        return results

    def _rrf(self, dense: list[dict], bm25: list[dict], k: int = 60) -> list[dict]:
        rrf_scores: dict[str, float] = {}
        all_by_id: dict[str, dict] = {}

        for rank, r in enumerate(dense):
            rrf_scores[r["id"]] = rrf_scores.get(r["id"], 0.0) + 1.0 / (k + rank + 1)
            all_by_id[r["id"]] = r

        for rank, r in enumerate(bm25):
            rrf_scores[r["id"]] = rrf_scores.get(r["id"], 0.0) + 1.0 / (k + rank + 1)
            all_by_id[r["id"]] = r

        ranked = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [{"id": rid, "score": rrf_scores[rid], "payload": all_by_id[rid]["payload"]} for rid in ranked]

    def search(self, query: str, filters: dict | None = None, top_k: int = TOP_K_RETRIEVAL, hyde_doc: str | None = None) -> list[dict]:
        filters = filters or {}
        dense = self.dense_search(query, filters, top_k, hyde_doc=hyde_doc)
        bm25 = self.bm25_search(query, filters, top_k)  # BM25 always uses original query
        return self._rrf(dense, bm25)[:top_k]
