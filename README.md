# SEC Filings RAG

Agentic RAG system for answering investor questions over SEC filings (10-K, 10-Q, 8-K) from the top 50 US companies by market cap.

**Architecture:** hybrid dense + BM25 retrieval → Cohere reranker → LangGraph agent (query analysis → retrieval → verification → generation) → Claude Haiku with inline citations.

**[Full Technical Report](https://docs.google.com/document/d/1YfCLOSIR7DLpoXwhxcJDd2a6m02rfKyCProjFm9S4XQ/edit?usp=sharing)** — system design, evaluation results, and experiment analysis.

---

## Prerequisites

You need API keys for three services:

| Service | Key name in `.env` | Get it at |
|---|---|---|
| Anthropic (Claude) | `claude` | console.anthropic.com |
| Voyage AI | `voyage` | dash.voyageai.com |
| Cohere | `cohere` | dashboard.cohere.com |

Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/) (for local setup), or Docker (for containerised setup).

---

## Quick Start — Docker (recommended)

```bash
# 1. Clone the repo
git clone <repo-url>
cd lc_rag

# 2. Create .env with your API keys
cat > .env <<EOF
claude="sk-ant-..."
voyage="pa-..."
cohere="..."
EOF

# 3. Add your PDFs
unzip the provided file or dump the folder sec_fillings_pdf in root

# 4. Build the image and start the container
docker compose up --build -d

# 5. Build both indexes (Qdrant + BM25) — required before the app can answer questions
docker exec -it sec_rag python -m ingestion.ingest

# 6. Open http://localhost:8501
```

> The indexes (`qdrant_storage/` and `bm25_index.pkl`) are not included in the repo and must be built on first run. Step 5 handles both. Once built they persist on the host via volume mounts — you only need to run ingestion again if you add new files.

On first run the image build takes ~2–3 minutes (installing system libs and Python deps). Subsequent `docker compose up` reuses the cached image and starts in seconds.

**Useful commands:**

```bash
# Run in the foreground (see logs live)
docker compose up --build

# Check it's healthy
docker ps

# Stream live logs
docker compose logs -f

# Stop the container
docker compose down

# Rebuild from scratch (e.g. after changing dependencies)
docker compose up --build
```

The container mounts `qdrant_storage/`, `bm25_index.pkl`, `sec_filings_pdf/`, `evaluation/`, and `.env` as volumes — all data and index changes persist on the host across restarts.

---

## Quick Start — Local

```bash
# 1. Install dependencies
uv sync

# 2. Create .env
cat > .env <<EOF
claude="sk-ant-..."
voyage="pa-..."
cohere="..."
EOF

# 3. Ingest PDFs (place them in sec_filings_pdf/<TICKER>/ first either by unzipping provided zip file or placing in root sec_fillings_pdf)
uv run python -m ingestion.ingest

# 4. Launch Streamlit UI
uv run streamlit run streamlit_app.py
# → http://localhost:8501
```

---

## PDF Naming Convention

Filings should follow `{TICKER}_{FILING_TYPE}_{YYYYMMDD}.pdf`:

```
sec_filings_pdf/
├── AAPL/
│   ├── AAPL_10-K_20251031.pdf
│   └── AAPL_10-Q_20250628.pdf
├── MSFT/
│   └── MSFT_10-K_20240630.pdf
└── NVDA/
    └── NVDA_10-K_20260126.pdf
```

The ingestion pipeline auto-detects ticker and filing type from the filename. The Streamlit UI also lets you upload PDFs directly and override the ticker if the name doesn't match the convention.

---

## Streamlit Interface

Open `http://localhost:8501` and use the four tabs:

| Tab | Description |
|---|---|
| **💬 Ask** | Ask natural-language questions about SEC filings. Returns grounded answers with inline citations (ticker, filing type, date, page number). Sidebar has example queries. |
| **📤 Ingest Data** | Upload new PDF filings or re-ingest all existing ones. Shows live ingestion progress streamed from the indexing pipeline. |
| **📋 Evaluation** | Compare experiment results, drill into per-question metrics, filter by category, and download results CSVs. Also shows the full test set with ground truth answers. |
| **⚙️ Settings** | Enter and save API keys (Anthropic, Voyage AI, Cohere). Keys are written to `.env` and applied immediately without restarting the server. |

If API keys are not configured, the Ask tab is disabled and the sidebar shows a warning indicating which keys are missing.

---

## Building the Indexes (Qdrant + BM25)

Neither `qdrant_storage/` nor `bm25_index.pkl` are committed to the repo. Both must be built locally from the source PDFs before the app is usable.

**One command builds both:**

```bash
# Place PDFs in sec_filings_pdf/<TICKER>/ first, then:
uv run python -m ingestion.ingest
```

This parses every PDF, chunks it, embeds with voyage-finance-2, upserts to Qdrant, and writes `bm25_index.pkl` at the end. Both indexes are created in a single run.

The pipeline is resumable — if interrupted, re-run the same command and it skips already-indexed files (`ingest_progress.json` tracks progress).

**Other useful flags:**

```bash
# Wipe everything and re-index from scratch
uv run python -m ingestion.ingest --recreate

# Skip table summarisation (faster, slightly lower recall on tables)
uv run python -m ingestion.ingest --no-summarize

# Rebuild only the BM25 index from existing Qdrant data (no re-embedding needed)
uv run python -m ingestion.ingest --rebuild-bm25
```

Once both indexes exist the app is fully functional with no further setup.

---

## Example Queries

- "What was NVIDIA's data center revenue in FY2025?"
- "How has Apple's gross margin trended over the past 3 fiscal years?"
- "Break down Amazon's revenue by segment for the last fiscal year."
- "Compare R&D spending as a percentage of revenue between Google and Microsoft."
- "What are Tesla's main risk factors in its most recent 10-K?"
- "How are financial sector companies discussing AI adoption?"
- "Compare Visa and Mastercard operating margins."

---

## Ingestion Pipeline

The ingestion pipeline (`ingestion/ingest.py`) processes PDFs in the following steps:

1. **Parse** — pdfplumber extracts text and tables from every page
2. **Chunk** — parent-child chunking (300-token child chunks for retrieval, ±150-token parent for generation)
3. **Table dual-indexing** — each table is indexed twice: raw markdown (for BM25 keyword recall) and an LLM-generated prose summary (for dense semantic retrieval)
4. **Embed** — voyage-finance-2 embeds all child chunks and table summaries
5. **Index** — vectors are upserted to Qdrant; BM25 index is rebuilt and saved to `bm25_index.pkl`

Re-ingesting is idempotent — existing chunks for a filing are overwritten, not duplicated.

---

## Running Evaluation

```bash
# Run a named experiment (exp02 = production config: Cohere reranker, no HyDE, k=30)
uv run python -m evaluation.run_experiment --name exp02_cohere_reranker --disable-hyde --top-k-retrieval 30

# Re-run an existing experiment (overwrites results)
uv run python -m evaluation.run_experiment --redo exp02_cohere_reranker --disable-hyde --top-k-retrieval 30

# Compare all experiments side-by-side
uv run python -m evaluation.run_experiment --compare
```

Results are saved to `evaluation/experiments/<exp_name>/`:
- `results.csv` — per-question metrics (P@k, R@k, faithfulness, accuracy, LLM-only accuracy)
- `summary.json` — aggregated metrics
- `config.json` — experiment configuration snapshot

The test set is at `evaluation/test_set.json` (24 questions: 8 easy, 7 medium, 9 hard). Ground truth verification records are at `evaluation/gt_manual_checklist.txt`.

---

## Project Structure

```
sec_rag/
├── agent/
│   └── graph.py          # LangGraph pipeline (query analysis → retrieval → verify → generate)
├── ingestion/
│   └── ingest.py         # PDF parsing, chunking, embedding, indexing
├── retrieval/
│   ├── dense.py          # Qdrant dense search (voyage-finance-2)
│   ├── bm25.py           # BM25 sparse search
│   └── reranker.py       # Cohere / FlashRank reranker
├── generation/
│   └── generator.py      # Claude Haiku answer synthesis with citations
├── evaluation/
│   ├── test_set.json      # 27-question test set with ground truths
│   ├── run_experiment.py  # Evaluation runner (metrics, LLM-as-judge)
│   └── experiments/       # Per-experiment results
├── streamlit_app.py       # Streamlit UI (4 tabs)
├── config.py              # Paths, model names, ticker→company mapping
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── REPORT.pdf             # Technical report (also at the link above)
```

---

## Using a Custom Dataset

The system works with **any collection of PDFs**, not just SEC filings. To point it at a different document set:

### 1. Organise your PDFs

Follow the naming convention `{TICKER}_{FILING_TYPE}_{YYYYMMDD}.pdf` and place files under `sec_filings_pdf/<TICKER>/`:

```
sec_filings_pdf/
├── MYCO/
│   ├── MYCO_10-K_20250101.pdf
│   └── MYCO_report_20240601.pdf
└── OTHERCO/
    └── OTHERCO_annual_20250301.pdf
```

The ticker and filing type are extracted from the filename automatically. Any string works for `FILING_TYPE` (e.g. `annual`, `report`, `slide`). The date must be `YYYYMMDD`.

### 2. Add a company name mapping (optional)

Open `config.py` and add your tickers to the `TICKER_TO_COMPANY` dict so the system displays full company names in answers:

```python
TICKER_TO_COMPANY = {
    ...
    "MYCO": "My Company Inc.",
}
```

### 3. Run ingestion

```bash
# Ingest all PDFs in sec_filings_pdf/ (resumes automatically if interrupted)
uv run python -m ingestion.ingest

# Or point to a different folder
uv run python -m ingestion.ingest --dir /path/to/your/pdfs
```

The pipeline parses each PDF, chunks it, generates table summaries, embeds with voyage-finance-2, and upserts to Qdrant. Progress is checkpointed per file — if the run is interrupted, re-run the same command to resume.

### 4. Query

The Ask tab in Streamlit works immediately after ingestion. For best results, include the ticker in your question so the query analyser applies metadata pre-filtering (e.g. "What was MYCO's revenue in 2025?").

> **Note:** voyage-finance-2 is trained on financial documents and works well on any structured financial text. For non-financial domains, consider swapping the embedding model in `config.py` (`EMBEDDING_MODEL`) to a general-purpose embedder like `voyage-large-2`.

---

## Environment Variables

All keys are read from `.env` in the project root (or from the ⚙️ Settings tab in the UI):

```env
claude="sk-ant-api03-..."   # Anthropic API key (Claude Haiku for generation + LLM-as-judge)
voyage="pa-..."             # Voyage AI key (voyage-finance-2 embeddings)
cohere="..."                # Cohere key (rerank-english-v3.0)
```
