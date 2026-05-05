"""
Evaluation runner — called as subprocess by run_experiment.py.

Usage:
    uv run python -m evaluation.run_eval --out results.csv
    uv run python -m evaluation.run_eval --out results.csv --category easy
    uv run python -m evaluation.run_eval --out results.csv --no-ablation
    uv run python -m evaluation.run_eval --out results.csv --redo-errors
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from tqdm import tqdm

from config import ANTHROPIC_API_KEY, LLM_MODEL
from agent.graph import query as rag_query
from evaluation.metrics import retrieval_metrics, faithfulness, answer_accuracy

_client = Anthropic(api_key=ANTHROPIC_API_KEY)

TEST_SET_PATH = Path(__file__).parent / "test_set.json"
CHECKPOINT_PATH = Path(__file__).parent / "eval_checkpoint.json"


def _llm_only(user_query: str) -> str:
    resp = _client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=(
            "You are an expert financial analyst. Answer questions about SEC filings "
            "from publicly traded companies based on your knowledge. Be precise with figures."
        ),
        messages=[{"role": "user", "content": user_query}],
    )
    return resp.content[0].text.strip()


def _load_checkpoint(out_path: Path) -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text())
    if out_path.exists():
        df = pd.read_csv(out_path)
        return {row["id"]: dict(row) for _, row in df.iterrows()}
    return {}


def _save_checkpoint(results: dict) -> None:
    CHECKPOINT_PATH.write_text(json.dumps(results))


def run_eval(
    out_path: Path,
    category: str | None = None,
    ablation: bool = True,
    redo_errors: bool = False,
) -> None:
    test_set = json.loads(TEST_SET_PATH.read_text())
    questions = test_set["questions"]
    if category:
        questions = [q for q in questions if q["category"] == category]

    checkpoint = _load_checkpoint(out_path) if redo_errors else {}

    rows = []
    for q in tqdm(questions, desc="Evaluating"):
        qid = q["id"]

        # Skip already-done rows (unless redo-errors and this was an error)
        if qid in checkpoint:
            prev = checkpoint[qid]
            rag_ans = str(prev.get("rag_answer", ""))
            if not redo_errors or not rag_ans.startswith("ERROR:"):
                rows.append(prev)
                continue

        try:
            result = rag_query(q["query"])
            rag_answer = result.response.answer

            context = "\n\n".join(
                c["payload"].get("parent_text") or c["payload"].get("text", "")
                for c in result.retrieved_chunks
            )

            ret = retrieval_metrics(
                query=q["query"],
                retrieved_chunks=result.retrieved_chunks,
                candidate_chunks=result.candidate_chunks,
            )

            faith = faithfulness(rag_answer, context)
            acc = answer_accuracy(rag_answer, q.get("ground_truth_answer", ""))

            row = {
                "id": qid,
                "category": q["category"],
                "query": q["query"],
                "rag_answer": rag_answer,
                "precision_k": ret["precision"],
                "recall_k": ret["recall"],
                "f1_k": ret["f1"],
                "chunks_retrieved": ret["k"],
                "chunks_relevant": ret["relevant"],
                "faithfulness": faith,
                "accuracy": acc,
                "llm_only_answer": "",
                "llm_only_accuracy": None,
            }

            if ablation:
                llm_ans = _llm_only(q["query"])
                llm_acc = answer_accuracy(llm_ans, q.get("ground_truth_answer", ""))
                row["llm_only_answer"] = llm_ans
                row["llm_only_accuracy"] = llm_acc

        except Exception as e:
            row = {
                "id": qid,
                "category": q["category"],
                "query": q["query"],
                "rag_answer": f"ERROR: {e}",
                "precision_k": 0.0,
                "recall_k": 0.0,
                "f1_k": 0.0,
                "chunks_retrieved": 0,
                "chunks_relevant": 0,
                "faithfulness": None,
                "accuracy": None,
                "llm_only_answer": "",
                "llm_only_accuracy": None,
            }

        rows.append(row)
        checkpoint[qid] = row
        _save_checkpoint(checkpoint)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    # Print quick summary
    valid = df[~df["rag_answer"].str.startswith("ERROR:", na=False)]
    print(f"P@k={valid['precision_k'].mean():.3f}  R@k={valid['recall_k'].mean():.3f}  "
          f"F1@k={valid['f1_k'].mean():.3f}  Faith={valid['faithfulness'].mean():.3f}  "
          f"Acc={valid['accuracy'].dropna().mean():.3f}")
    if "llm_only_accuracy" in valid and valid["llm_only_accuracy"].notna().any():
        rag_acc = valid["accuracy"].dropna().mean()
        llm_acc = valid["llm_only_accuracy"].dropna().mean()
        print(f"RAG lift: {rag_acc - llm_acc:+.3f}  (RAG={rag_acc:.3f}, LLM-only={llm_acc:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--category", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--redo-errors", action="store_true")
    args = parser.parse_args()

    run_eval(
        out_path=Path(args.out),
        category=args.category,
        ablation=not args.no_ablation,
        redo_errors=args.redo_errors,
    )
