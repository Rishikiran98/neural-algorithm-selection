"""Small cross-domain pilot for the LOCAL v0.10 API.

Most natural-language tasks are intentionally UNGRADED. This measures system
behavior and retains answers for human review; it does not establish general
intelligence or correctness on arbitrary tasks.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from time import time

from .client import CPUReasoningClient
from .datasets import split_of, FrozenRegistryError
from cpu_first.core import parse_number

MAX_ITEMS = 500


def _numeric(response: object):
    if not isinstance(response, str):
        return None
    text = response.strip().replace(",", "")
    m = re.search(r"(?:final answer|answer)\s*:\s*([+-]?\d+(?:\.\d+)?)\b", text, re.I)
    value = m.group(1) if m else text
    try:
        return parse_number(value)
    except (ValueError, TypeError):
        return None


def score(row: dict, result: dict) -> tuple[str, str]:
    """Result (grade, reason). Only exact numeric tasks are auto-scored."""
    if result.get("status") != "ok":
        return "error", "request failed or no answer"
    kind = row.get("grading", "manual")
    if kind == "manual":
        return "unscored", "human assessment required"
    if kind != "numeric_exact":
        raise ValueError("Unknown grading kind")
    expected = parse_number(row["expected"])
    actual = _numeric(result.get("answer"))
    if actual is None:
        return "unscored", "numeric final answer unavailable in accepted extraction format"
    return ("correct", "exact numeric match") if actual == expected else ("incorrect", "numeric mismatch")


def run(dataset: str, output: str, *, url: str="http://127.0.0.1:8070",
        api_key: str="", limit: int=20, thinking: bool=False, strategy: str="direct") -> dict:
    if not 1 <= limit <= MAX_ITEMS:
        raise ValueError("limit must be 1..500")
    path = Path(dataset)
    # Fail closed. This development pilot must never consume the frozen eval set
    # or a copied subset with changed identifiers. No API calls until classified.
    split, _ = split_of(path)
    if split == "eval":
        raise FrozenRegistryError("Frozen evaluation material cannot be used in the development pilot")
    output_file = Path(output)
    if path.resolve() == output_file.resolve():
        raise ValueError("Output cannot overwrite the input dataset")
    items = []
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not all(k in row for k in ("id", "question", "domain", "grading")):
            raise ValueError("Each row requires id, question, domain, grading")
        if not isinstance(row["id"], str) or row["id"] in ids:
            raise ValueError("Missing or duplicate dataset ID")
        ids.add(row["id"])
        if row["grading"] not in ("manual", "numeric_exact"):
            raise ValueError("Unknown grading type")
        if row["grading"] == "numeric_exact":
            parse_number(row.get("expected"))
        items.append(row)
    if not items:
        raise ValueError("Empty dataset")
    client = CPUReasoningClient(url, api_key, timeout=180)
    records = []
    for row in items[:limit]:
        try:
            result = client.ask(row["question"], domain=row["domain"],
                                strategy=strategy, thinking=thinking)
        except Exception as exc:
            result = {"status": "error", "error": type(exc).__name__ + ": " + str(exc)[:100]}
        grade, reason = score(row, result)
        records.append({"id": row["id"], "domain": row["domain"], "grading": row["grading"],
                        "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
                        "grade": grade, "grade_reason": reason, "result": result})
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = dict(Counter(r["grade"] for r in records))
    summary = {"version": "0.10.0", "status": "development_pilot_not_general_intelligence_claim",
               "dataset": str(path), "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "timestamp": time(), "model_url": url, "thinking": thinking, "strategy": strategy,
               "total": len(records), "grades": counts,
               "num_exact_scored": counts.get("correct", 0) + counts.get("incorrect", 0),
               "exact_accuracy_on_scored": (counts.get("correct", 0) /
                   (counts.get("correct", 0) + counts.get("incorrect", 0))
                   if counts.get("correct", 0) + counts.get("incorrect", 0) else None),
               "unscored_and_errors_are_not_successes": True,
               "total_model_calls": sum(r["result"].get("usage", {}).get("model_calls", 0) for r in records)}
    summary_path = output_file.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Cross-domain local Q&A smoke test, not proof of reasoning quality")
    parser.add_argument("--dataset", default="data/general_smoke.jsonl")
    parser.add_argument("--output", default="results/general_smoke_v010.jsonl")
    parser.add_argument("--url", default="http://127.0.0.1:8070")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--strategy", choices=("direct", "check_and_revise"), default="direct")
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.output, url=args.url, api_key=args.api_key,
                         limit=args.limit, thinking=args.thinking, strategy=args.strategy), indent=2))

if __name__ == "__main__":
    main()
