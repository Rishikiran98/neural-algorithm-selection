"""V0.4 paired CPU inference benchmark and reproducibility manifest.

A measured real model is required for research results. No accuracy is inferred
from the offline fixture tests. Template support is treated as coverage, not a
successful answer. Server/model energy is NOT measured by this harness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess

from cpu_first.benchmark import load_questions, RSSSampler
from cpu_first.core import gsm8k_gold
from .client import CPUReasoningClient
from . import __version__

MODES = ("baseline", "hybrid", "adaptive", "proof", "semantic")


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256() -> str:
    """Fingerprint code without depending on a git checkout or python caches."""
    root = Path(__file__).resolve().parent.parent
    h = hashlib.sha256()
    for folder in ("cpu_first", "cpu_platform"):
        for file in sorted((root / folder).rglob("*.py")):
            h.update(str(file.relative_to(root)).encode("utf-8"))
            h.update(file.read_bytes())
    return h.hexdigest()


def memory_total_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            state = MEMORYSTATUSEX()
            state.dwLength = ctypes.sizeof(state)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
                return state.ullTotalPhys
        except (AttributeError, OSError):
            pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ValueError, OSError):
        pass
    return None


def manifest(dataset: Path, *, model_id: str, quantization: str, threads: int | None,
             model_file: Path | None, modes: tuple[str, ...], seed: int,
             repeats: int, gpu_layers: int, context_size: int | None) -> dict:
    if gpu_layers != 0:
        raise ValueError("v0.4 enforces a CPU-only inference declaration: --gpu-layers 0")
    if model_file and not model_file.is_file():
        raise FileNotFoundError(model_file)
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                  timeout=2, cwd=Path(__file__).resolve().parent.parent, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        revision = None
    return {
        "release": __version__, "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "logical_cpus": os.cpu_count(), "system_ram_bytes": memory_total_bytes()},
        "dataset": {"file": str(dataset), "sha256": sha256_file(dataset)},
        "source": {"sha256": source_sha256(), "git_revision": revision},
        "model": {"id": model_id or None, "quantization": quantization or None,
                  "weights_sha256": sha256_file(model_file) if model_file else None,
                  "cpu_threads_declared": threads, "context_size_declared": context_size,
                  "gpu_layers_declared": gpu_layers, "cpu_only_execution_attested": False},
        "experiment": {"modes": modes, "seed": seed, "repeats": repeats,
                       "comparison": "paired problems, rotating mode order; different compute budgets"},
        "limitations": [
            "A zero GPU-layer launch declaration alone does not verify actual hardware execution.",
            "Accuracy requires independently specified ground truth; fixture/mocked tests are not model performance evidence.",
            "Server CPU time and energy are not measured; model latency is wall-clock proxy only.",
            "Semantic coverage is confined to fully matched templates; unknown wording must abstain.",
            "The modes use different inference budgets, so any accuracy gain is not compute matched.",
        ],
    }


def summarize(rows: list[dict], modes: tuple[str, ...]) -> dict:
    if not rows or not modes or len(set(modes)) != len(modes):
        raise ValueError("Missing results or duplicate modes")
    paired: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = (row["id"], row["repeat"])
        if row["mode"] in paired.setdefault(key, {}):
            raise ValueError("Duplicate mode for paired task")
        paired[key][row["mode"]] = row
    if any(set(pair) != set(modes) for pair in paired.values()):
        raise ValueError("Every paired task must have exactly the requested modes")
    results = {}
    for mode in modes:
        group = [p[mode] for p in paired.values()]
        count = len(group)
        correct = sum(r["correct"] for r in group)
        lat = sorted(r["latency_ms"] for r in group)
        calls = sum(r["model_calls"] for r in group)
        supported = sum(r["semantic_supported"] for r in group)
        results[mode] = {
            "tasks": count, "correct": correct, "accuracy": round(correct / count, 4),
            "abstentions_or_invalid": sum(r["status"] != "ok" for r in group),
            "semantic_coverage": round(supported / count, 4) if mode == "semantic" else None,
            "semantically_checked_correct": sum(r["correct"] and r["semantically_verified"] for r in group),
            "model_calls_total": calls, "model_calls_per_correct": round(calls / correct, 3) if correct else None,
            "prompt_tokens": sum(r["prompt_tokens"] for r in group),
            "completion_tokens": sum(r["completion_tokens"] for r in group),
            "mean_latency_ms": round(statistics.mean(lat), 2),
            "p95_latency_ms": round(lat[math.ceil(.95 * count) - 1], 2),
            "wall_ms_per_correct": round(sum(lat) / correct, 2) if correct else None,
        }
    deltas = {}
    if "baseline" in modes:
        for challenger in modes:
            if challenger == "baseline":
                continue
            diffs = [int(p[challenger]["correct"]) - int(p["baseline"]["correct"])
                     for _, p in sorted(paired.items())]
            rng = random.Random(4004)
            boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs)
                           for _ in range(2000))
            deltas[f"{challenger}_minus_baseline"] = {
                "accuracy_difference": round(statistics.mean(diffs), 4),
                "exploratory_bootstrap_95pct_interval": [round(boots[49], 4), round(boots[1949], 4)],
            }
    return {"paired_observations": len(paired), "modes": results,
            "paired_accuracy_differences": deltas,
            "caution": "Raw task repeats are not independent examples. Bootstrap intervals are exploratory."}


def run_benchmark(client, questions: list[dict], out: Path, *, modes: tuple[str, ...] = MODES,
                  seed: int = 404, repeats: int = 1, warmup: int = 0) -> dict:
    if not questions or repeats < 1 or warmup < 0 or not modes or any(m not in MODES for m in modes):
        raise ValueError("Invalid dataset, modes, repeats or warmup")
    if len(set(modes)) != len(modes):
        raise ValueError("Duplicate modes")
    for _ in range(warmup):
        # Explicitly label warmups; avoid contaminating recorded results.
        warmed = client.reason("What is 2 plus 2?", mode="baseline", max_calls=1)
        if warmed.get("http_status") or warmed.get("status") not in ("ok", "invalid"):
            raise RuntimeError("Warmup request failed")
    work = [(str(item.get("id", i)), repeat, item) for repeat in range(repeats)
            for i, item in enumerate(questions)]
    random.Random(seed).shuffle(work)
    rows = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream:
        for index, (task_id, repeat, item) in enumerate(work):
            gold = gsm8k_gold(item["answer"])
            order = modes[index % len(modes):] + modes[:index % len(modes)]
            for mode in order:
                result = client.reason(item["question"], mode=mode,
                                       max_calls=2 if mode in ("adaptive", "proof", "semantic") else 1)
                if result.get("http_status") not in (None, 422):
                    raise RuntimeError(f"HTTP failure for {task_id} {mode}: {result.get('error')}")
                if result.get("status") not in ("ok", "invalid"):
                    raise RuntimeError(f"Invalid API status for {task_id} {mode}")
                try:
                    correct = (result["status"] == "ok" and result.get("answer") is not None
                               and Fraction(result["answer"]) == gold)
                except (ValueError, TypeError, ZeroDivisionError):
                    correct = False
                usage = result.get("usage") or {}
                row = {"id": task_id, "repeat": repeat, "mode": mode,
                       "status": result["status"], "correct": bool(correct),
                       "gold": str(gold), "answer": result.get("answer"),
                       "semantic_supported": result.get("stop_reason") != "unsupported_template"
                           if mode == "semantic" else False,
                       "semantically_verified": bool(result.get("semantically_verified")),
                       "stop_reason": result.get("stop_reason"),
                       "model_calls": usage.get("model_calls", 0),
                       "prompt_tokens": usage.get("prompt_tokens", 0),
                       "completion_tokens": usage.get("completion_tokens", 0),
                       "latency_ms": usage.get("total_latency_ms", 0)}
                rows.append(row)
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                stream.flush()
                print(f"{task_id} repeat={repeat} {mode}: correct={correct} calls={row['model_calls']}")
    return summarize(rows, modes)


def main():
    parser = argparse.ArgumentParser(description="v0.4 CPU-only paired real-model evaluation")
    parser.add_argument("--dataset", default="data/smoke.jsonl")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--api-url", default="http://127.0.0.1:8070")
    parser.add_argument("--output", default="results/experiment4.jsonl")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--quantization", default="")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--model-file", type=Path, help="Optional GGUF weights to fingerprint")
    parser.add_argument("--model-pid", type=int, help="Optional llama.cpp PID to sample RSS via psutil")
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--launch-manifest", type=Path,
                        help="results/launch_<port>.json from cpu_platform.launch; fills threads, "
                             "context size, model id and PID, and is embedded in the summary")
    args = parser.parse_args()
    launch_info = None
    if args.launch_manifest is not None and args.launch_manifest.is_file():
        launch_info = json.loads(args.launch_manifest.read_text(encoding="utf-8"))
        settings = launch_info.get("settings", {})
        args.threads = args.threads or settings.get("threads")
        args.context_size = args.context_size or settings.get("ctx_size")
        args.model_id = args.model_id or (launch_info.get("model") or {}).get("label", "")
        args.model_pid = args.model_pid or (launch_info.get("llama_cpp") or {}).get("pid")
    elif args.launch_manifest is not None:
        print(f"NOTE: launch manifest {args.launch_manifest} not found; declared settings only.")
    modes = tuple(m.strip() for m in args.modes.split(","))
    if args.limit < 1 or args.repeats < 1 or args.warmup < 0 or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        parser.error("Invalid limit, repeats, warmup, or modes")
    if args.gpu_layers != 0:
        parser.error("CPU-only benchmark requires --gpu-layers 0")
    dataset = Path(args.dataset)
    questions = load_questions(dataset, args.limit)
    meta = manifest(dataset, model_id=args.model_id, quantization=args.quantization,
                    threads=args.threads, model_file=args.model_file, modes=modes,
                    seed=args.seed, repeats=args.repeats, gpu_layers=args.gpu_layers,
                    context_size=args.context_size)
    client = CPUReasoningClient(args.api_url, api_key=os.environ.get("CPU_REASONING_API_KEY", ""))
    with RSSSampler(args.model_pid) as monitor:
        scores = run_benchmark(client, questions, Path(args.output), modes=modes,
                               seed=args.seed, repeats=args.repeats, warmup=args.warmup)
    meta["measurement"] = {"model_pid_peak_rss_mb": monitor.peak_mb,
                           "rss_measurement_note": "Sampling is optional, approximate and needs psutil; PID is user-supplied."}
    meta["scores"] = scores
    meta["launch_manifest"] = launch_info
    meta["warning"] = ("SMOKE TEST ONLY; cannot establish model quality" if len(questions) < 30
                       else "Exploratory model benchmark, not a frontier-level claim")
    summary = Path(args.output).with_suffix(".summary.json")
    summary.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Summary:", summary)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
