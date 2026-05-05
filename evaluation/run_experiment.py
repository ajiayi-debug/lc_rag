#!/usr/bin/env -S uv run python3
"""
Experiment runner — wraps run_eval in a subprocess with config overrides.

Usage:
    # New experiment with current defaults
    uv run python -m evaluation.run_experiment --name "cohere_reranker"

    # Override specific settings
    uv run python -m evaluation.run_experiment --name "baseline" --disable-cohere --disable-hyde --top-k-retrieval 20 --no-chunk-cap

    # Redo an existing experiment (add/replace results.csv without renumbering)
    uv run python -m evaluation.run_experiment --redo exp01_baseline_flashrank --disable-cohere --disable-hyde --top-k-retrieval 20 --no-chunk-cap

    # Compare all experiments
    uv run python -m evaluation.run_experiment --compare

Each run saves to:
    evaluation/experiments/exp{N:02d}_{name}/
        config.json      — effective config + git hash
        summary.json     — aggregate metrics
        results.csv      — per-question results
        test_set.json    — snapshot of test set used
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
TEST_SET_PATH = Path(__file__).parent / "test_set.json"
CHECKPOINT_PATH = Path(__file__).parent / "eval_checkpoint.json"


def _next_exp_dir(name: str) -> Path:
    existing = sorted(EXPERIMENTS_DIR.glob("exp[0-9][0-9]*"))
    n = len(existing) + 1
    slug = name.lower().replace(" ", "_")
    return EXPERIMENTS_DIR / f"exp{n:02d}_{slug}"


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _build_env(overrides: dict) -> dict:
    """Merge current env with experiment-specific overrides."""
    env = os.environ.copy()
    for k, v in overrides.items():
        env[k] = str(v)
    return env


def _build_config_snapshot(exp_dir: Path, name: str, notes: str, category: str | None, overrides: dict) -> dict:
    """Build config.json capturing effective values used in this experiment."""
    top_k_retrieval = int(overrides.get("TOP_K_RETRIEVAL", os.getenv("TOP_K_RETRIEVAL", "30")))
    top_k_rerank = int(overrides.get("TOP_K_RERANK", os.getenv("TOP_K_RERANK", "6")))
    max_chunks = overrides.get("MAX_CHUNKS_TOTAL", os.getenv("MAX_CHUNKS_TOTAL", "12"))
    disable_cohere = overrides.get("DISABLE_COHERE", "0") == "1"
    disable_hyde = overrides.get("DISABLE_HYDE", "0") == "1"

    # Determine reranker label
    cohere_key = os.getenv("cohere")
    if disable_cohere or not cohere_key:
        reranker = "flashrank rank-T5-flan (local)"
    else:
        reranker = "Cohere rerank-english-v3.0"

    return {
        "experiment": exp_dir.name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "git_hash": _git_hash(),
        "name": name,
        "notes": notes,
        "category_filter": category,
        "retrieval": {
            "embedding_model": "voyage-finance-2",
            "reranker": reranker,
            "hyde": not disable_hyde,
            "TOP_K_RETRIEVAL": top_k_retrieval,
            "TOP_K_RERANK": top_k_rerank,
            "MAX_CHUNKS_TOTAL": None if max_chunks == "0" else int(max_chunks),
        },
        "generation": {
            "model": os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
        },
    }


def _build_summary(exp_dir: Path, df: pd.DataFrame) -> dict:
    valid = df[~df["rag_answer"].str.startswith("ERROR:", na=False)]
    summary = {
        "experiment": exp_dir.name,
        "questions": len(df),
        "errors": len(df) - len(valid),
        "retrieval": {
            "precision_k": round(valid["precision_k"].mean(), 3),
            "recall_k": round(valid["recall_k"].mean(), 3),
            "f1_k": round(valid["f1_k"].mean(), 3),
        },
        "generation": {
            "faithfulness": round(valid["faithfulness"].mean(), 3),
            "accuracy": round(valid["accuracy"].dropna().mean(), 3) if valid["accuracy"].notna().any() else None,
            "llm_only_accuracy": round(valid["llm_only_accuracy"].dropna().mean(), 3) if "llm_only_accuracy" in valid and valid["llm_only_accuracy"].notna().any() else None,
        },
        "per_category": {},
    }
    for cat in ["easy", "medium", "hard"]:
        sub = valid[valid["category"] == cat]
        if sub.empty:
            continue
        summary["per_category"][cat] = {
            "precision_k": round(sub["precision_k"].mean(), 3),
            "recall_k": round(sub["recall_k"].mean(), 3),
            "f1_k": round(sub["f1_k"].mean(), 3),
            "faithfulness": round(sub["faithfulness"].mean(), 3),
            "accuracy": round(sub["accuracy"].dropna().mean(), 3) if sub["accuracy"].notna().any() else None,
        }
    if summary["generation"].get("accuracy") and summary["generation"].get("llm_only_accuracy"):
        summary["generation"]["rag_improvement"] = round(
            summary["generation"]["accuracy"] - summary["generation"]["llm_only_accuracy"], 3
        )
    return summary


def run_experiment(
    name: str,
    notes: str = "",
    category: str | None = None,
    ablation: bool = True,
    redo_errors: bool = False,
    overrides: dict | None = None,
    target_dir: Path | None = None,
) -> Path:
    overrides = overrides or {}
    exp_dir = target_dir or _next_exp_dir(name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExperiment: {exp_dir.name}")

    config = _build_config_snapshot(exp_dir, name, notes, category, overrides)
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    shutil.copy(TEST_SET_PATH, exp_dir / "test_set.json")

    results_path = exp_dir / "results.csv"
    CHECKPOINT_PATH.unlink(missing_ok=True)

    # Run eval as subprocess so env var overrides take effect cleanly
    cmd = [
        "uv", "run", "python", "-m", "evaluation.run_eval",
        "--out", str(results_path),
    ]
    if category:
        cmd += ["--category", category]
    if not ablation:
        cmd += ["--no-ablation"]
    if redo_errors:
        cmd += ["--redo-errors"]

    env = _build_env(overrides)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\n[ERROR] Eval subprocess failed (exit {result.returncode})")
        return exp_dir

    shutil.copy(results_path, Path(__file__).parent / "results.csv")

    df = pd.read_csv(results_path)
    summary = _build_summary(exp_dir, df)
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nSaved to {exp_dir}/")
    return exp_dir


def compare_experiments() -> None:
    rows = []
    for exp_dir in sorted(EXPERIMENTS_DIR.glob("exp[0-9]*")):
        s_path = exp_dir / "summary.json"
        c_path = exp_dir / "config.json"
        if not s_path.exists():
            continue
        s = json.loads(s_path.read_text())
        c = json.loads(c_path.read_text()) if c_path.exists() else {}
        ret = c.get("retrieval", {})
        arch = ret.get("reranker", "?")[:30]
        if ret.get("hyde"):
            arch += " + HyDE"
        rows.append({
            "Experiment": exp_dir.name,
            "Architecture": arch,
            "P@k": s["retrieval"]["precision_k"],
            "R@k": s["retrieval"]["recall_k"],
            "F1@k": s["retrieval"]["f1_k"],
            "Faith": s["generation"]["faithfulness"],
            "Acc": s["generation"].get("accuracy"),
            "RAG lift": s["generation"].get("rag_improvement"),
        })
    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
    else:
        print("No experiments found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None, help="Short name for this experiment")
    parser.add_argument("--notes", default="", help="What changed in this experiment")
    parser.add_argument("--category", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--redo-errors", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Print comparison table of all experiments")
    parser.add_argument("--redo", metavar="DIRNAME", default=None,
                        help="Redo into existing exp folder (e.g. exp01_baseline_flashrank)")

    # Config overrides
    parser.add_argument("--disable-cohere", action="store_true", help="Force flashrank (no Cohere reranker)")
    parser.add_argument("--disable-hyde", action="store_true", help="Disable HyDE query expansion")
    parser.add_argument("--top-k-retrieval", type=int, default=None, metavar="N")
    parser.add_argument("--top-k-rerank", type=int, default=None, metavar="N")
    parser.add_argument("--no-chunk-cap", action="store_true", help="Disable MAX_CHUNKS_TOTAL cap")

    args = parser.parse_args()

    if args.compare:
        compare_experiments()
    else:
        overrides = {}
        if args.disable_cohere:
            overrides["DISABLE_COHERE"] = "1"
        if args.disable_hyde:
            overrides["DISABLE_HYDE"] = "1"
        if args.top_k_retrieval is not None:
            overrides["TOP_K_RETRIEVAL"] = str(args.top_k_retrieval)
        if args.top_k_rerank is not None:
            overrides["TOP_K_RERANK"] = str(args.top_k_rerank)
        if args.no_chunk_cap:
            overrides["MAX_CHUNKS_TOTAL"] = "999"

        target_dir = None
        if args.redo:
            target_dir = EXPERIMENTS_DIR / args.redo
            if not target_dir.exists():
                parser.error(f"Experiment dir not found: {target_dir}")
            name = args.name or args.redo
        elif args.name:
            name = args.name
        else:
            parser.error("--name is required unless --compare or --redo is specified")

        run_experiment(
            name=name,
            notes=args.notes,
            category=args.category,
            ablation=not args.no_ablation,
            redo_errors=args.redo_errors,
            overrides=overrides,
            target_dir=target_dir,
        )
