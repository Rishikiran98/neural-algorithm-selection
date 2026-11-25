"""Experiment 002: paired API evaluation of baseline, hybrid, adaptive.

Requires the local CPU platform API and (for actual inference) a local model.
No energy data is inferred or manufactured. Failing to reach the service fails the run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
from typing import Any

from cpu_first.benchmark import load_questions
from cpu_first.core import gsm8k_gold
from .client import CPUReasoningClient

MODES = ("baseline", "hybrid", "adaptive")


def summarize(rows: list[dict]) -> dict:
    """Paired task accuracy and cost proxies. Missing cases are never dropped."""
    if not rows:
        raise ValueError("No benchmark records")
    keyed: dict[str, dict[str, dict]] = {}
    for row in rows:
        bucket = keyed.setdefault(row["id"], {})
        if row["mode"] in bucket:
            raise ValueError("Duplicate mode for example " + row["id"])
        bucket[row["mode"]] = row
    if any(set(parts) != set(MODES) for parts in keyed.values()):
        raise ValueError("Incomplete paired benchmark: each task must have all modes")

    metrics = {}
    for mode in MODES:
        filtered = [r for r in rows if r["mode"] == mode]
        correct = sum(int(r["correct"]) for r in filtered)
        calls = sum(r["model_calls"] for r in filtered)
        latency = sum(r["latency_ms"] for r in filtered)
        timings = sorted(r["latency_ms"] for r in filtered)
        metrics[mode] = {
            "examples": len(filtered),
            "correct": correct,
            "accuracy": round(correct / len(filtered), 4),
            "valid_output_rate": round(sum(r["status"] == "ok" for r in filtered) / len(filtered), 4),
            "total_model_calls": calls,
            "mean_model_calls_per_task": round(calls / len(filtered), 3),
            "total_latency_ms": round(latency, 2),
            "mean_latency_ms": round(statistics.mean(timings), 2),
            "p95_latency_ms": round(timings[math.ceil(.95 * len(timings)) - 1], 2),
            "model_calls_per_correct": round(calls / correct, 3) if correct else None,
            "latency_ms_per_correct": round(latency / correct, 2) if correct else None,
            "prompt_tokens": sum(r["prompt_tokens"] for r in filtered),
            "completion_tokens": sum(r["completion_tokens"] for r in filtered),
            "stop_reasons": {reason: sum(r.get("stop_reason") == reason for r in filtered)
                             for reason in sorted(set(r.get("stop_reason") for r in filtered))},
        }
    ordered = sorted(keyed)
    intervals = {}
    for first, second in (("adaptive", "hybrid"), ("adaptive", "baseline"), ("hybrid", "baseline")):
        diff = [int(keyed[id][first]["correct"]) - int(keyed[id][second]["correct"]) for id in ordered]
        rng = random.Random(2002)
        bootstrap = sorted(
            statistics.mean(rng.choice(diff) for _ in diff) for _ in range(2000)
        )
        intervals[f"{first}_minus_{second}"] = {
            "accuracy_difference": round(statistics.mean(diff), 4),
            "exploratory_paired_bootstrap_95pct_interval": [round(bootstrap[49], 4), round(bootstrap[1949], 4)],
            "discordant_first_only": sum(d == 1 for d in diff),
            "discordant_second_only": sum(d == -1 for d in diff),
        }
    return {
        "n_paired_problems": len(keyed),
        "modes": metrics,
        "paired_differences": intervals,
        "interpretation": "Accuracy, token and latency comparisons only. No energy or CPU-seconds measured; budgets not compute-matched."
    }


def run_benchmark(client: Any, questions: list[dict], out: Path) -> dict:
    """Run all three modes on all questions, with rotating execution order."""
    out.parent.mkdir(exist_ok=True, parents=True)
    rows = []
    with out.open("w", encoding="utf-8") as fh:
        for i, item in enumerate(questions):
            gold = gsm8k_gold(item["answer"])
            for mode in MODES[i % 3:] + MODES[:i % 3]:
                result = client.reason(item["question"], mode=mode,
                                       max_calls=2 if mode == "adaptive" else 1)
                if "http_status" in result and result["http_status"] != 422:
                    raise RuntimeError(f"API failed for example {i}, mode {mode}: "
                                       f"HTTP {result['http_status']} ({result.get('error')})")
                if result.get("status") not in ("ok", "invalid"):
                    raise RuntimeError(f"Unexpected response on example {i}, {mode}: {result.get('status')}")
                prediction = result.get("answer")
                try:
                    correct = result["status"] == "ok" and prediction is not None and Fraction(prediction) == gold
                except (ValueError, ZeroDivisionError, TypeError):
                    correct = False
                usage = result.get("usage") or {}
                row = {
                    "id": str(item.get("id", i)),
                    "mode": mode, "status": result["status"], "correct": bool(correct),
                    "answer": prediction, "gold": str(gold),
                    "arithmetic_verified": bool(result.get("arithmetic_verified")),
                    "numeric_grounding_passed": result.get("numeric_grounding_passed"),
                    "semantically_verified": bool(result.get("semantically_verified")),
                    "stop_reason": result.get("stop_reason"),
                    "model_calls": usage.get("model_calls", 0),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "latency_ms": usage.get("total_latency_ms", 0),
                }
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"{i+1}/{len(questions)} {mode:<9} correct={row['correct']} "
                      f"calls={row['model_calls']} latency={row['latency_ms']:.0f}ms")
    result = summarize(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description="Paired CPU API reasoning benchmark, three methods")
    parser.add_argument("--dataset", default="data/smoke.jsonl")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--api-url", default="http://127.0.0.1:8070")
    parser.add_argument("--output", default="results/experiment2.jsonl")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.limit < 30:
        print("CAUTION: fewer than 30 paired tasks: exploratory/debugging only")
    questions = load_questions(Path(args.dataset), args.limit)
    client = CPUReasoningClient(args.api_url, api_key=os.environ.get("CPU_REASONING_API_KEY", ""))
    report = run_benchmark(client, questions, Path(args.output))
    summary = {
        "version": "0.2.0", "run_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset, "limit": args.limit,
        "api_url": args.api_url, "platform": platform.platform(),
        "cpu": platform.processor(), "cpu_count": os.cpu_count(),
        "benchmark": report,
        "important_limits": [
            "Smoke data and small sample results are not publishable evidence",
            "Actual model revision and CPU-only execution must be captured separately from llama.cpp logs",
            "Model-generated expressions are not semantically verified",
            "Grounding only checks lexical numeric membership with a few conversion constants",
            "Neither energy nor CPU-seconds are measured",
            "Additional adaptive calls are not cost-matched to a stronger model or matched compute budget",
            "Experiment writes output files incrementally; interrupted runs are incomplete",
        ],
    }
    path = Path(args.output).with_suffix(".summary.json")
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Summary:", path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
