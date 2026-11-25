"""Experiment 003: four-mode paired evaluation with explicit proof validity.

All models must be evaluated on the same fixed dataset and CPU setup.
A syntactically valid graph is never treated as semantically correct without
matching a separate known ground truth.
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

from cpu_first.benchmark import load_questions
from cpu_first.core import gsm8k_gold
from .client import CPUReasoningClient

MODES = ("baseline", "hybrid", "adaptive", "proof")


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No benchmark records")
    keyed: dict[str, dict] = {}
    for row in rows:
        bucket = keyed.setdefault(row["id"], {})
        if row["mode"] in bucket:
            raise ValueError("Duplicate mode for task")
        bucket[row["mode"]] = row
    if any(set(parts) != set(MODES) for parts in keyed.values()):
        raise ValueError("Every task must contain each of the four modes")
    report = {}
    for mode in MODES:
        subset = [r[mode] for r in keyed.values()]
        n = len(subset)
        correct = sum(r["correct"] for r in subset)
        latencies = sorted(r["latency_ms"] for r in subset)
        calls = sum(r["model_calls"] for r in subset)
        report[mode] = {
            "tasks": n,
            "correct": correct,
            "accuracy": round(correct/n, 4),
            "success_output_rate": round(sum(r["status"] == "ok" for r in subset)/n, 4),
            "model_calls_total": calls,
            "calls_per_correct": round(calls/correct, 3) if correct else None,
            "mean_calls_per_task": round(calls/n, 3),
            "mean_latency_ms": round(statistics.mean(latencies), 2),
            "p95_latency_ms": round(latencies[math.ceil(n*.95)-1], 2),
            "ms_per_correct": round(sum(latencies)/correct, 2) if correct else None,
            "verified_step_total": sum(r["proof_steps"] for r in subset),
            "proof_pass_rate": round(sum(r["proof_verified"] for r in subset)/n, 4),
            "prompt_tokens": sum(r["prompt_tokens"] for r in subset),
            "completion_tokens": sum(r["completion_tokens"] for r in subset),
        }
    # Fixed-seed bootstrap is exploratory; task IDs are paired.
    deltas = {}
    for challenger in ("hybrid", "adaptive", "proof"):
        diffs = [int(pair[challenger]["correct"]) - int(pair["baseline"]["correct"])
                 for _, pair in sorted(keyed.items())]
        rng = random.Random(3003)
        boots = sorted(sum(rng.choice(diffs) for _ in diffs)/len(diffs) for _ in range(2000))
        deltas[f"{challenger}_minus_baseline"] = {
            "difference": round(statistics.mean(diffs), 4),
            "exploratory_bootstrap_95pct_interval": [round(boots[49], 4), round(boots[1949], 4)],
        }
    return {
        "paired_tasks": len(keyed), "modes": report, "accuracy_differences": deltas,
        "note": "No matched-compute guarantee, no energy measurements. Proof passes mean exact arithmetic replay, NOT semantic proof.",
    }


def run_benchmark(client, questions: list[dict], path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with path.open("w", encoding="utf-8") as stream:
        for i, item in enumerate(questions):
            gold = gsm8k_gold(item["answer"])
            # Rotate the request order to reduce systematic mode-order bias.
            for mode in MODES[i % 4:] + MODES[:i % 4]:
                result = client.reason(item["question"], mode=mode,
                                       max_calls=2 if mode in ("proof", "adaptive") else 1)
                if result.get("http_status") not in (None, 422):
                    raise RuntimeError(f"HTTP failure for task {i} mode {mode}: {result.get('error')}")
                if result.get("status") not in ("ok", "invalid"):
                    raise RuntimeError("Model server failed or malformed response")
                try:
                    correct = (result["status"] == "ok" and result.get("answer") is not None
                               and Fraction(result["answer"]) == gold)
                except (TypeError, ValueError, ZeroDivisionError):
                    correct = False
                usage = result.get("usage") or {}
                row = {
                    "id": str(item.get("id", i)), "mode": mode, "status": result["status"],
                    "correct": bool(correct), "answer": result.get("answer"), "gold": str(gold),
                    "model_calls": usage.get("model_calls", 0),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "latency_ms": usage.get("total_latency_ms", 0),
                    "proof_verified": bool(result.get("proof_verified")),
                    "proof_steps": len(result.get("proof_trace") or []),
                    "semantic_verification": bool(result.get("semantically_verified")),
                    "stop_reason": result.get("stop_reason"),
                }
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(f"{i+1}/{len(questions)} {mode:<9} correct={correct} calls={row['model_calls']}")
    return summarize(rows)


def main():
    parser = argparse.ArgumentParser(description="v0.3 paired CPU-only reasoning comparison")
    parser.add_argument("--dataset", default="data/smoke.jsonl")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--api-url", default="http://127.0.0.1:8070")
    parser.add_argument("--output", default="results/experiment3.jsonl")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    questions = load_questions(Path(args.dataset), args.limit)
    if len(questions) < 30:
        print("WARNING: Small dataset for smoke testing only, not valid for research claims")
    client = CPUReasoningClient(args.api_url, api_key=os.environ.get("CPU_REASONING_API_KEY", ""))
    report = run_benchmark(client, questions, Path(args.output))
    summary = {"release": "0.3.0", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "dataset": args.dataset, "hardware": {"os": platform.platform(),
               "cpu": platform.processor(), "logical_cpus": os.cpu_count()},
               "results": report,
               "caveats": ["Model ID, quantization, prompt version and CPU-only flags must be logged separately",
                           "Arithmetic proof validation is not problem-semantic verification",
                           "All four modes use different inference budgets", "No watts or CPU-seconds measured"]}
    output = Path(args.output).with_suffix(".summary.json")
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Saved:", output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
