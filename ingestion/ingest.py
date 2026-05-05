"""
Main ingestion pipeline — with per-file checkpointing.

Crash/resume behaviour:
  - Progress is saved to ingest_progress.json after every PDF is fully upserted.
  - On restart, already-finished files are skipped automatically.
  - BM25 index is rebuilt from Qdrant at the end (or via --rebuild-bm25 alone).

Usage:
    python -m ingestion.ingest                       # ingest all PDFs (resumes if interrupted)
    python -m ingestion.ingest --no-summarize        # skip table summarisation (faster)
    python -m ingestion.ingest --recreate            # wipe everything and re-index from scratch
    python -m ingestion.ingest --rebuild-bm25        # rebuild BM25 from existing Qdrant data only
    python -m ingestion.ingest --dir /path/to/pdfs   # custom PDF folder
"""

import argparse
import json
import pickle
from pathlib import Path

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

from config import (
    BM25_INDEX_PATH,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    QDRANT_PATH,
    ROOT_DIR,
    SEC_FILINGS_DIR,
)
from ingestion.chunker import chunk_document
from ingestion.embedder import embed_texts
from ingestion.pdf_parser import parse_pdf

PROGRESS_PATH = ROOT_DIR / "ingest_progress.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_client() -> QdrantClient:
    return QdrantClient(path=QDRANT_PATH)


def _setup_collection(client: QdrantClient, recreate: bool = False) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        if recreate:
            client.delete_collection(COLLECTION_NAME)
        else:
            return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


def _load_progress() -> set[str]:
    if PROGRESS_PATH.exists():
        try:
            return set(json.loads(PROGRESS_PATH.read_text()))
        except (json.JSONDecodeError, ValueError):
            print("  [warn] ingest_progress.json corrupted — starting fresh")
            return set()
    return set()


def _save_progress(done: set[str]) -> None:
    PROGRESS_PATH.write_text(json.dumps(sorted(done), indent=2))


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


# ── BM25 rebuild (from Qdrant — no in-memory state needed) ───────────────────

def rebuild_bm25(client: QdrantClient) -> None:
    print("Rebuilding BM25 index from Qdrant...")
    all_ids: list[str] = []
    all_texts: list[str] = []
    ticker_to_indices: dict[str, list[int]] = {}
    filing_type_to_indices: dict[str, list[int]] = {}

    offset = None
    idx = 0
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            p = point.payload
            all_ids.append(str(point.id))
            all_texts.append(p.get("text", ""))
            ticker_to_indices.setdefault(p.get("ticker", ""), []).append(idx)
            filing_type_to_indices.setdefault(p.get("filing_type", ""), []).append(idx)
            idx += 1
        if offset is None:
            break

    print(f"  Building BM25 over {len(all_texts)} chunks...")
    bm25 = BM25Okapi([_tokenize(t) for t in all_texts])

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "ids": all_ids,
            "ticker_to_indices": ticker_to_indices,
            "filing_type_to_indices": filing_type_to_indices,
        }, f)

    print(f"  BM25 index saved → {BM25_INDEX_PATH}")


# ── Main ingestion ────────────────────────────────────────────────────────────

def run_ingestion(
    filings_dir: str | Path | None = None,
    summarize_tables: bool = True,
    recreate: bool = False,
) -> None:
    filings_dir = Path(filings_dir or SEC_FILINGS_DIR)
    pdf_files = sorted(filings_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {filings_dir}")

    client = _get_client()

    if recreate:
        PROGRESS_PATH.unlink(missing_ok=True)

    _setup_collection(client, recreate=recreate)

    done_files = _load_progress()
    pending = [p for p in pdf_files if p.name not in done_files]

    if not pending:
        print("All PDFs already indexed. Run with --recreate to start fresh.")
    else:
        if done_files:
            print(f"Resuming: {len(done_files)} done, {len(pending)} remaining.")

        # ── Per-file: parse → chunk → embed → upsert → checkpoint ────────
        CONSECUTIVE_FAIL_LIMIT = 3
        consecutive_failures = 0

        for pdf_path in tqdm(pending, desc="Ingesting"):
            try:
                tqdm.write(f"\n→ {pdf_path.name}")

                tqdm.write("  parsing...")
                doc = parse_pdf(pdf_path)
                n_tables = sum(len(p.tables) for p in doc.pages)
                tqdm.write(f"  {len(doc.pages)} pages, {n_tables} tables")

                tqdm.write(f"  chunking{f' + summarizing {n_tables} tables' if summarize_tables and n_tables else ''}...")
                chunks = chunk_document(doc, summarize_tables=summarize_tables)
                tqdm.write(f"  {len(chunks)} chunks ({sum(1 for c in chunks if c.chunk_type == 'table_summary')} table summaries)")

                if not chunks:
                    done_files.add(pdf_path.name)
                    _save_progress(done_files)
                    continue

                tqdm.write("  embedding...")
                embeddings = embed_texts([c.text for c in chunks], input_type="document")

                points = [
                    PointStruct(id=c.id, vector=emb, payload=c.to_payload())
                    for c, emb in zip(chunks, embeddings)
                ]
                client.upsert(collection_name=COLLECTION_NAME, points=points)

                # Mark file as done only after successful upsert
                done_files.add(pdf_path.name)
                _save_progress(done_files)
                consecutive_failures = 0  # reset on success

            except Exception as e:
                if "credit balance" in str(e).lower():
                    print(
                        f"\n\n  *** FATAL: Anthropic credit balance too low ***\n"
                        f"  Top up at console.anthropic.com/settings/billing\n"
                        f"  Progress saved up to this point — re-run to resume.\n"
                    )
                    return

                consecutive_failures += 1
                print(f"\n  [error] {pdf_path.name}: {e}")

                if consecutive_failures >= CONSECUTIVE_FAIL_LIMIT:
                    tqdm.write(
                        f"\n  Aborted: {consecutive_failures} consecutive failures — "
                        "likely an API or network issue, not a bad PDF.\n"
                        "  Progress saved. Re-run the same command to resume once the issue is fixed."
                    )
                    return

                tqdm.write("  Skipping — will retry on next run.")

    total = client.count(COLLECTION_NAME).count
    print(f"Qdrant collection: {total} chunks")

    rebuild_bm25(client)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=None, help="Path to PDF folder")
    parser.add_argument("--no-summarize", action="store_true", help="Skip table summarisation")
    parser.add_argument("--recreate", action="store_true", help="Wipe everything and re-index")
    parser.add_argument("--rebuild-bm25", action="store_true", help="Rebuild BM25 from existing Qdrant data")
    args = parser.parse_args()

    if args.rebuild_bm25:
        rebuild_bm25(_get_client())
    else:
        run_ingestion(
            filings_dir=args.dir,
            summarize_tables=not args.no_summarize,
            recreate=args.recreate,
        )
