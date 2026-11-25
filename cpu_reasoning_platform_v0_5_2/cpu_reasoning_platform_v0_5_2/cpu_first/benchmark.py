"""Paired baseline-vs-symbolic benchmark against a local CPU-only llama server."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import threading
import time

from .core import LocalLlama, formatted_fraction, gsm8k_gold, parse_number, solve


def load_questions(filename: Path, limit: int) -> list[dict]:
    rows = []
    with filename.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                if "question" not in item or "answer" not in item:
                    raise ValueError(f"Line {i+1} missing question/answer")
                rows.append(item)
            if len(rows) >= limit:
                break
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


class RSSSampler:
    """Optional approximate peak RSS for server + descendants, using psutil."""
    def __init__(self, pid: int | None):
        self.pid = pid
        self.peak_mb = None
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        if self.pid is None:
            return self
        try:
            import psutil
        except ImportError:
            print("WARN: pip install psutil for approximate server RSS measurement", file=sys.stderr)
            return self
        try:
            process = psutil.Process(self.pid)
        except psutil.Error as exc:
            print(f"WARN: can't access server PID: {exc}", file=sys.stderr)
            return self

        def sample():
            while not self.stop.is_set():
                try:
                    procs = [process] + process.children(recursive=True)
                    total = sum(p.memory_info().rss for p in procs if p.is_running()) / 1024**2
                    self.peak_mb = max(self.peak_mb or 0, total)
                except psutil.Error:
                    pass
                self.stop.wait(0.05)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1)


def summarize(records: list[dict]) -> dict:
    by_mode = {}
    for mode in ("baseline", "hybrid"):
        filtered = [row for row in records if row["mode"] == mode]
        n = len(filtered)
        num_correct = sum(bool(row["correct"]) for row in filtered)
        by_mode[mode] = {
            "total": n,
            "correct": num_correct,
            "accuracy": round(num_correct / n, 4),
            "valid_output_rate": round(sum(row["output_valid"] for row in filtered) / n, 4),
            "average_latency_s": round(statistics.mean(row["latency_s"] for row in filtered), 3),
            "total_latency_s": round(sum(row["latency_s"] for row in filtered), 2),
            "completion_tokens": sum(row["completion_tokens"] or 0 for row in filtered),
            "prompt_tokens": sum(row["prompt_tokens"] or 0 for row in filtered),
            "inference_calls": n,
        }
        by_mode[mode]["completion_tokens_per_correct"] = (
            round(by_mode[mode]["completion_tokens"] / num_correct, 1) if num_correct else None
        )
    # Paired bootstrap over problems, NOT independent marginal intervals.
    pairs = {}
    for row in records:
        pairs.setdefault(row["id"], {})[row["mode"]] = int(row["correct"])
    differences = [p["hybrid"] - p["baseline"] for p in pairs.values()
                   if "hybrid" in p and "baseline" in p]
    rng = random.Random(2026)
    samples = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(2000)
    ) if differences else []
    return {
        "modes": by_mode,
        "hybrid_minus_baseline_accuracy": round(statistics.mean(differences), 4) if differences else None,
        "paired_bootstrap_95pct_interval": (
            [round(samples[50], 4), round(samples[1949], 4)] if samples else None
        ),
        "note": "The bootstrap interval is exploratory; results are not independently validated.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/smoke.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", default="results/pilot.jsonl")
    parser.add_argument("--server-pid", type=int, default=None)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    questions = load_questions(Path(args.dataset), args.limit)
    client = LocalLlama(args.url)
    out = Path(args.output)
    out.parent.mkdir(exist_ok=True, parents=True)
    records = []
    with RSSSampler(args.server_pid) as sampler, out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(questions):
            gold = gsm8k_gold(row["answer"])
            # Alternate order to partly control for warmup and caching order effects.
            modes = ("baseline", "hybrid") if i % 2 == 0 else ("hybrid", "baseline")
            for mode in modes:
                try:
                    result = solve(client, row["question"], mode)
                except Exception as exc:
                    # Fail visibly; do not silently convert outages to low accuracy.
                    raise RuntimeError(f"Inference failed on problem {i} ({mode}): {exc}") from exc
                result.update({
                    "id": str(row.get("id", i)),
                    "correct": result["output_valid"] and parse_number(result["prediction"]) == gold
                    if result["output_valid"] and "/" not in result["prediction"] else
                    result["output_valid"] and result["prediction"] == formatted_fraction(gold),
                    "gold": formatted_fraction(gold),
                })
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                records.append(result)
                print(f"{i+1}/{len(questions)} {mode:8s} correct={result['correct']} "
                      f"latency={result['latency_s']:.2f}s")
    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "model": client.model,
        "server": client.base_url,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "approx_peak_server_rss_mb": round(sampler.peak_mb, 1) if sampler.peak_mb is not None else None,
        "benchmarks": summarize(records),
        "limitations": [
            "No independently measured energy consumption",
            "RSS sampled when --server-pid and psutil supplied; includes server descendants, not page cache",
            "No verification that natural language was translated correctly to arithmetic",
            "CPU-only execution must be ensured with llama.cpp --n-gpu-layers 0 and no external inference",
            "No frontier reasoning claim: initial benchmark is narrow arithmetic",
        ],
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n", json.dumps(summary["benchmarks"], indent=2))
    print("Output:", out, "and", summary_path)


if __name__ == "__main__":
    main()
