"""
SEC Filings RAG — Streamlit demo UI.

Run:
    streamlit run streamlit_app.py
"""

import json
import os
import queue as _queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SEC Filings RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.citation-badge {
    display: inline-block;
    background: #4f8ef7;
    color: #ffffff;
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
}
.section-header {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    opacity: 0.5;
    margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ── API key helpers ───────────────────────────────────────────────────────────

ENV_PATH = Path(__file__).parent / ".env"
API_KEY_VARS = {
    "claude":  "Anthropic (Claude)",
    "voyage":  "Voyage AI",
    "cohere":  "Cohere",
}


def _load_env_file() -> dict[str, str]:
    """Parse .env file into a dict."""
    result = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                result[k.strip()] = v.strip().strip('"').strip("'")
    return result


def _save_env_file(keys: dict[str, str]) -> None:
    """Write key=value pairs to .env, merging with existing content."""
    existing = _load_env_file()
    existing.update({k: v for k, v in keys.items() if v})
    lines = [f'{k}="{v}"' for k, v in existing.items()]
    ENV_PATH.write_text("\n".join(lines) + "\n")


def _apply_keys_to_process(keys: dict[str, str]) -> None:
    """Set env vars in the current process so config.py picks them up immediately."""
    for k, v in keys.items():
        if v:
            os.environ[k] = v


def _keys_configured() -> dict[str, bool]:
    env = _load_env_file()
    return {k: bool(env.get(k) or os.environ.get(k)) for k in API_KEY_VARS}


# ── Index / config helpers ────────────────────────────────────────────────────

def _get_config():
    from config import TICKER_TO_COMPANY, QDRANT_PATH, BM25_INDEX_PATH
    return TICKER_TO_COMPANY, QDRANT_PATH, BM25_INDEX_PATH


def _index_ready() -> bool:
    _, qdrant_path, bm25_path = _get_config()
    return Path(qdrant_path).exists() and Path(bm25_path).exists()


def _detect_ticker(filename: str) -> str | None:
    m = re.match(r"^([A-Z\-]+)_", filename)
    return m.group(1) if m else None


# ── Trace renderer ────────────────────────────────────────────────────────────

_STEP_ICONS = {
    "query_analysis": "🧠",
    "retrieval": "🔎",
    "verification": "✅",
    "generation": "✍️",
}


def _render_trace_step(t: dict) -> None:
    """Render a single trace step (called live during streaming)."""
    step = t["step"]
    icon = _STEP_ICONS.get(step, "•")

    if step == "query_analysis":
        st.markdown(f"**{icon} Query Analysis**")
        cols = st.columns(3)
        cols[0].metric("Tickers", ", ".join(t["tickers"]) or "all")
        cols[1].metric("Filing types", ", ".join(t["filing_types"]) or "all")
        cols[2].metric("Complex query", "Yes" if t["is_complex"] else "No")
        if t["is_complex"] and t["sub_queries"]:
            st.markdown("Sub-queries decomposed:")
            for i, sq in enumerate(t["sub_queries"], 1):
                st.markdown(f"&nbsp;&nbsp;&nbsp;**{i}.** {sq}")

    elif step == "retrieval":
        attempt = t["attempt"]
        label = f"**{icon} Retrieval** (attempt {attempt})" if attempt > 1 else f"**{icon} Retrieval**"
        st.markdown(label)
        for sq, count in t["chunks_per_query"].items():
            st.markdown(f"&nbsp;&nbsp;&nbsp;`{sq[:80]}` → **{count}** chunks after reranking")
        st.caption(f"Total unique chunks in context: {t['total_chunks']}")

    elif step == "verification":
        if t["passed"]:
            st.markdown(f"**{icon} Verification** — ✅ {t['reason']}")
        else:
            st.markdown(f"**⚠️ Verification** — ❌ {t['reason']}")
            st.markdown("Broadened queries:")
            for sq in t.get("broadened_queries", []):
                st.markdown(f"&nbsp;&nbsp;&nbsp;→ {sq}")

    elif step == "generation":
        st.markdown(f"**{icon} Generation** — used {t['chunks_used']} chunks → {t['citations']} cited sources")

    st.divider()


def _render_trace(trace: list[dict]) -> None:
    for t in trace:
        _render_trace_step(t)


def _clean_excerpt(text: str) -> str:
    text = re.sub(r"^\[.*?\]\s*\n?", "", text.strip())
    if "|" in text[:100]:
        lines = []
        for line in text.splitlines():
            line = re.sub(r"\|[-: ]+\|.*", "", line)
            line = re.sub(r"\s*\|\s*", " · ", line).strip(" ·")
            if line:
                lines.append(line)
        text = "  ".join(lines[:4])
    return text[:300]


# ── Apply any saved keys to the process on every page load ───────────────────
_apply_keys_to_process(_load_env_file())

TICKER_TO_COMPANY, QDRANT_PATH, BM25_INDEX_PATH = _get_config()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 SEC Filings RAG")
    st.caption("Top 50 US companies · 10-K, 10-Q, 8-K filings")
    st.divider()

    # API key status
    key_status = _keys_configured()
    all_set = all(key_status.values())
    if all_set:
        st.success("API keys configured")
    else:
        missing = [API_KEY_VARS[k] for k, ok in key_status.items() if not ok]
        st.warning(f"Missing keys: {', '.join(missing)}\nGo to ⚙️ Settings tab.")

    st.divider()

    if not _index_ready():
        st.error("Index not built. Upload PDFs and run ingestion in the 📤 Ingest tab.")
    else:
        st.success(f"Index ready · {len(TICKER_TO_COMPANY)} companies")

    st.divider()
    st.markdown("**Example queries**")
    examples = [
        "What was Microsoft's total revenue in FY2024?",
        "Break down Amazon's revenue by segment for the last fiscal year.",
        "How has Apple's gross margin trended over the past 3 fiscal years?",
        "Compare R&D spending as % of revenue between Google and Microsoft.",
        "Summarize how semiconductor companies are discussing AI-related demand.",
        "Give me an overview of NVIDIA's recent performance.",
        "How are financial sector companies discussing AI adoption?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state["pending_query"] = ex

    st.divider()
    st.markdown("**Companies in knowledge base**")
    st.markdown("  ".join(f"`{t}`" for t in sorted(TICKER_TO_COMPANY.keys())))


def _run_ingestion(uploaded_files, overrides):
    """Save uploaded files and run ingestion with live log streaming."""
    if uploaded_files:
        saved = []
        for uf in uploaded_files:
            ticker = overrides[uf.name]
            dest_dir = Path("sec_filings_pdf") / ticker
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / uf.name
            dest.write_bytes(uf.read())
            saved.append(dest.name)
        st.success(f"Saved {len(saved)} file(s): {', '.join(saved)}")

    # Release Qdrant lock so ingestion subprocess can open it
    from agent.graph import reset_searcher
    reset_searcher()

    st.markdown("#### Ingestion Progress")
    status_box = st.empty()
    progress_bar = st.progress(0, text="Starting...")
    log_area = st.empty()

    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, "-m", "ingestion.ingest"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(Path(__file__).parent),
        env=env,
    )

    lines = []
    total_files = 0
    done_files = 0

    for raw_line in proc.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        lines.append(line)

        # Parse progress hints from ingest.py output
        if "files to process" in line.lower() or "pdfs found" in line.lower():
            m = re.search(r"(\d+)", line)
            if m:
                total_files = int(m.group(1))
        if line.strip().startswith("✓") or "done" in line.lower() or "upserted" in line.lower():
            done_files += 1
            if total_files:
                pct = min(done_files / total_files, 1.0)
                progress_bar.progress(pct, text=f"{done_files}/{total_files} files processed")

        log_area.code("\n".join(lines[-40:]), language=None)

    proc.wait()
    if proc.returncode == 0:
        progress_bar.progress(1.0, text="Complete!")
        status_box.success("Ingestion complete. New filings are now searchable.")
    else:
        progress_bar.empty()
        status_box.error("Ingestion failed. Check the log above.")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_query, tab_ingest, tab_eval, tab_settings = st.tabs([
    "💬 Ask", "📤 Ingest Data", "📋 Evaluation", "⚙️ Settings"
])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Query
# ══════════════════════════════════════════════════════════════════════════════
with tab_query:
    st.markdown("### Ask anything about SEC filings")
    st.caption("Answers retrieved directly from 10-K, 10-Q, and 8-K filings with citations.")

    if not all_set:
        st.warning("Configure your API keys in the ⚙️ Settings tab before querying.")

    if "pending_query" in st.session_state:
        st.session_state["query_text"] = st.session_state.pop("pending_query")

    is_streaming = st.session_state.get("streaming", False)

    query = st.text_area(
        "Your question",
        key="query_text",
        height=80,
        placeholder="e.g. What was NVIDIA's data center revenue in FY2025?",
        disabled=is_streaming,
    )

    col_btn, col_stop, col_clear = st.columns([1, 1, 5])
    with col_btn:
        submitted = st.button(
            "Ask", type="primary", use_container_width=True,
            disabled=not (_index_ready() and all_set) or is_streaming,
        )
    with col_stop:
        stop_clicked = st.button("⏹ Stop", use_container_width=True, disabled=not is_streaming)
    with col_clear:
        if st.button("Clear history"):
            for k in ("history", "trace_steps", "stream_resp", "stream_history_added"):
                st.session_state.pop(k, None)

    # Handle stop
    if stop_clicked and st.session_state.get("stop_event"):
        st.session_state["stop_event"].set()
        st.session_state["streaming"] = False

    # Start new query in background thread
    if submitted and query.strip():
        stop_ev = threading.Event()
        rq: _queue.Queue = _queue.Queue()

        def _run(q_str, out_q, stop):
            try:
                from agent.graph import stream_query
                for node, output in stream_query(q_str):
                    if stop.is_set():
                        out_q.put(("stopped",))
                        return
                    out_q.put(("step", node, output))
                out_q.put(("done",))
            except Exception as exc:
                out_q.put(("error", exc))

        threading.Thread(target=_run, args=(query.strip(), rq, stop_ev), daemon=True).start()
        st.session_state.update({
            "streaming": True,
            "stream_queue": rq,
            "stop_event": stop_ev,
            "trace_steps": [],
            "stream_resp": None,
            "stream_history_added": False,
        })
        st.rerun()

    # Drain queue while streaming
    if st.session_state.get("streaming"):
        rq = st.session_state["stream_queue"]
        while True:
            try:
                item = rq.get_nowait()
            except _queue.Empty:
                break
            kind = item[0]
            if kind == "step":
                _, _node, output = item
                for step in output.get("trace", []):
                    st.session_state["trace_steps"].append(step)
                if output.get("response") is not None:
                    st.session_state["stream_resp"] = output["response"]
            else:
                st.session_state["streaming"] = False
                if kind == "error":
                    st.error(f"Error: {item[1]}")
                break

    # Render live trace + answer
    running = st.session_state.get("streaming", False)
    trace_steps = st.session_state.get("trace_steps", [])
    resp = st.session_state.get("stream_resp")

    if running or trace_steps or resp:
        st.markdown("---")
        with st.expander(
            f"🔍 Agent trace {'(running…)' if running else '— done'}",
            expanded=running,
        ):
            _render_trace(trace_steps)

        if resp is not None and not running:
            st.markdown('<p class="section-header">Answer</p>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(resp.answer)

            if resp.citations:
                st.markdown('<p class="section-header">Sources</p>', unsafe_allow_html=True)
                badges_html = " ".join(
                    f'<span class="citation-badge">{c.ticker} {c.filing_type} {c.filing_date[:7]} p.{c.page_num}</span>'
                    for c in resp.citations
                )
                st.markdown(badges_html, unsafe_allow_html=True)
                st.write("")
                for i, cit in enumerate(resp.citations, 1):
                    with st.expander(f"[{i}] {cit.company_name} · {cit.filing_type} · {cit.filing_date} · Page {cit.page_num}"):
                        st.caption(f"📄 {cit.filename}")
                        st.markdown(_clean_excerpt(cit.excerpt))

            if not running and not st.session_state.get("stream_history_added"):
                st.session_state["stream_history_added"] = True
                if "history" not in st.session_state:
                    st.session_state["history"] = []
                st.session_state["history"].insert(0, {
                    "query": st.session_state.get("query_text", ""),
                    "answer": resp.answer,
                    "citations": resp.citations,
                })
                st.session_state["history"] = st.session_state["history"][:10]

    elif submitted and not query.strip():
        st.warning("Please enter a question.")

    # Poll every 0.5s while streaming
    if st.session_state.get("streaming"):
        time.sleep(0.5)
        st.rerun()

    if not running and st.session_state.get("history"):
        st.markdown("---")
        st.markdown('<p class="section-header">Previous queries this session</p>', unsafe_allow_html=True)
        for item in st.session_state["history"][1:]:
            with st.expander(f"Q: {item['query'][:100]}{'...' if len(item['query']) > 100 else ''}"):
                st.markdown(item["answer"])
                if item["citations"]:
                    st.markdown(" ".join(
                        f'<span class="citation-badge">{c.ticker} {c.filing_type} p.{c.page_num}</span>'
                        for c in item["citations"]
                    ), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Ingest Data
# ══════════════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.markdown("### Ingest SEC Filings")
    st.caption("Upload PDF filings to add them to the knowledge base.")

    with st.expander("📋 Filename format", expanded=False):
        st.markdown(
            "Files must follow the naming convention below so the ticker and filing type are auto-detected:\n\n"
            "```\n{TICKER}_{FILING_TYPE}_{YYYYMMDD}.pdf\n```\n\n"
            "| Example | Ticker | Filing | Date |\n"
            "|---|---|---|---|\n"
            "| `AAPL_10-K_20251031.pdf` | AAPL | 10-K | 2025-10-31 |\n"
            "| `MSFT_10-Q_20250430.pdf` | MSFT | 10-Q | 2025-04-30 |\n"
            "| `NVDA_8-K_20260122.pdf` | NVDA | 8-K | 2026-01-22 |\n\n"
            "If the filename doesn't match, the ticker column will show **—** and you can type it in the Override column."
        )

    # ── File upload section ───────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Select PDF filings",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if uploaded_files:
        st.markdown("**Files to ingest:**")
        cols = st.columns([3, 1, 1])
        cols[0].markdown("**Filename**")
        cols[1].markdown("**Detected ticker**")
        cols[2].markdown("**Override**")
        overrides = {}
        for uf in uploaded_files:
            auto = _detect_ticker(uf.name)
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.text(uf.name)
            c2.text(auto or "—")
            override = c3.text_input("", key=f"tick_{uf.name}", label_visibility="collapsed",
                                     placeholder="e.g. AAPL")
            overrides[uf.name] = override.strip().upper() if override.strip() else auto

        missing = [uf.name for uf in uploaded_files if not overrides.get(uf.name)]
        if missing:
            st.warning(f"Ticker unknown for: {', '.join(missing)}. Enter ticker overrides above.")

        can_ingest = not missing and all_set
        if not all_set:
            st.info("Configure API keys in ⚙️ Settings before ingesting.")

        if st.button("💾 Save & Ingest", type="primary", disabled=not can_ingest, key="ingest_btn"):
            _run_ingestion(uploaded_files, overrides)

    else:
        st.info("No files selected yet. Upload one or more PDFs above. See the filename format guide above.")

    st.divider()

    # ── Sync index ────────────────────────────────────────────────────────────
    st.markdown("#### Sync Index")
    st.caption(
        "Picks up any PDFs in `sec_filings_pdf/` that are not yet indexed. "
        "Already-indexed files are skipped automatically — only new files are processed."
    )

    pdf_count = sum(1 for _ in Path("sec_filings_pdf").rglob("*.pdf")) if Path("sec_filings_pdf").exists() else 0
    done_count = len(json.loads(Path("ingest_progress.json").read_text())) if Path("ingest_progress.json").exists() else 0
    pending_count = max(pdf_count - done_count, 0)
    st.markdown(
        f"`sec_filings_pdf/` — **{pdf_count} PDFs** total, "
        f"**{done_count} indexed**, **{pending_count} pending**"
        if Path("sec_filings_pdf").exists() else "`sec_filings_pdf/` not found."
    )

    if st.button("🔄 Sync New Files", disabled=not all_set, key="reingest_all_btn"):
        _run_ingestion(uploaded_files=None, overrides=None)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### Evaluation")
    import pandas as pd

    experiments_dir = Path("evaluation/experiments")

    exp_dirs = sorted(experiments_dir.glob("exp[0-9]*")) if experiments_dir.exists() else []
    exp_summaries = []
    for ed in exp_dirs:
        s_path = ed / "summary.json"
        c_path = ed / "config.json"
        if not s_path.exists():
            continue
        s = json.loads(s_path.read_text())
        c = json.loads(c_path.read_text()) if c_path.exists() else {}
        ret = c.get("retrieval", {})
        arch_parts = [ret.get("reranker", "?")[:35]]
        if ret.get("hyde"):
            arch_parts.append("HyDE")
        exp_summaries.append({
            "Experiment": ed.name,
            "Architecture": " + ".join(arch_parts),
            "P@k": s["retrieval"]["precision_k"],
            "R@k": s["retrieval"]["recall_k"],
            "F1@k": s["retrieval"]["f1_k"],
            "Faithfulness": s["generation"]["faithfulness"],
            "Accuracy": s["generation"].get("accuracy"),
            "RAG lift": s["generation"].get("rag_improvement"),
            "_dir": str(ed),
            "_has_csv": (ed / "results.csv").exists(),
        })

    if exp_summaries:
        st.markdown("#### Experiment Comparison")
        display_df = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in exp_summaries])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("#### Drill Down")
        exp_options = [r["Experiment"] for r in exp_summaries]
        selected = st.selectbox("Select experiment to inspect", exp_options, index=len(exp_options) - 1)
        selected_meta = next(r for r in exp_summaries if r["Experiment"] == selected)
        selected_dir = Path(selected_meta["_dir"])

        if selected_meta["_has_csv"]:
            df = pd.read_csv(selected_dir / "results.csv")
            valid = df[~df["rag_answer"].str.startswith("ERROR:", na=False)]

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Questions", len(df))
            m2.metric("Precision@k", f"{valid['precision_k'].mean():.3f}")
            m3.metric("Recall@k", f"{valid['recall_k'].mean():.3f}" if valid["recall_k"].notna().any() else "N/A")
            m4.metric("Faithfulness", f"{valid['faithfulness'].mean():.3f}" if valid['faithfulness'].notna().any() else "N/A")
            if valid["accuracy"].notna().any():
                rag_acc = valid["accuracy"].mean()
                llm_acc = valid["llm_only_accuracy"].mean() if "llm_only_accuracy" in valid.columns and valid["llm_only_accuracy"].notna().any() else None
                delta = rag_acc - llm_acc if llm_acc is not None else None
                m5.metric("RAG Accuracy", f"{rag_acc:.3f}", delta=f"{delta:+.3f}" if delta is not None else None)
            else:
                m5.metric("RAG Accuracy", "N/A")

            st.markdown("**Per-question results**")
            display_cols = ["id", "category", "query", "precision_k", "recall_k", "faithfulness", "accuracy", "llm_only_accuracy"]
            display_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(
                df[display_cols].rename(columns={
                    "id": "ID", "category": "Category", "query": "Question",
                    "precision_k": "P@k", "recall_k": "R@k",
                    "faithfulness": "Faith", "accuracy": "RAG Acc", "llm_only_accuracy": "LLM Acc",
                }),
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("Download results CSV"):
                st.download_button(
                    f"Download {selected}_results.csv",
                    data=(selected_dir / "results.csv").read_bytes(),
                    file_name=f"{selected}_results.csv",
                    mime="text/csv",
                )
        else:
            st.info("No results.csv for this experiment yet.")

        with st.expander("Config"):
            if (selected_dir / "config.json").exists():
                st.json(json.loads((selected_dir / "config.json").read_text()))
    else:
        st.info("No experiments yet. Run:\n```\nuv run python -m evaluation.run_experiment --name 'my_exp'\n```")

    st.divider()

    # ── Test set viewer ───────────────────────────────────────────────────────
    test_set_path = Path("evaluation/test_set.json")
    if test_set_path.exists():
        data = json.loads(test_set_path.read_text())
        questions = [q for q in data["questions"] if q.get("ground_truth_answer", "TBD") != "TBD"]
        st.markdown(f"#### Test Set · {len(questions)} questions")

        cat_filter = st.selectbox("Filter by category", ["all", "easy", "medium", "hard"])
        filtered = questions if cat_filter == "all" else [q for q in questions if q["category"] == cat_filter]

        for q in filtered:
            with st.expander(f"[{q['id']}] {q['query']}"):
                st.markdown(f"**Category:** {q['category']}  |  **Tickers:** {', '.join(q['expected_tickers'])}")
                st.markdown("**Ground truth:**")
                st.markdown(q.get("ground_truth_answer", ""))
                src = q.get("ground_truth_source", "")
                if src:
                    st.caption(f"Source: {src}")
    else:
        st.warning("test_set.json not found.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: Settings
# ══════════════════════════════════════════════════════════════════════════════
with tab_settings:
    st.markdown("### API Key Configuration")
    st.caption("Keys are saved to `.env` in the project root and applied immediately. They persist across container restarts.")

    current_env = _load_env_file()

    st.markdown("#### Enter API Keys")
    new_keys = {}
    for var, label in API_KEY_VARS.items():
        current_val = current_env.get(var) or os.environ.get(var, "")
        new_keys[var] = st.text_input(
            f"{label}  (`{var}`)",
            type="password",
            placeholder="Already configured — paste to update" if current_val else "Paste your key here",
            key=f"key_input_{var}",
        )

    col_save, col_status = st.columns([1, 3])
    with col_save:
        if st.button("💾 Save Keys", type="primary", use_container_width=True):
            to_save = {k: v for k, v in new_keys.items() if v.strip()}
            if to_save:
                _save_env_file(to_save)
                _apply_keys_to_process(to_save)
                st.success(f"{len(to_save)} key(s) saved.")
            else:
                st.warning("No keys entered.")

    st.divider()
    st.markdown("#### Current Status")
    status_cols = st.columns(len(API_KEY_VARS))
    for col, (var, label) in zip(status_cols, API_KEY_VARS.items()):
        val = current_env.get(var) or os.environ.get(var, "")
        if val:
            col.success(f"**{label}**  \nConfigured ✓")
        else:
            col.error(f"**{label}**  \nNot set")

    st.divider()
    st.markdown("#### Environment")
    st.info(
        f"`.env` path: `{ENV_PATH}`  \n"
        f"Python: `{sys.executable}`  \n"
        f"Working dir: `{Path.cwd()}`"
    )
