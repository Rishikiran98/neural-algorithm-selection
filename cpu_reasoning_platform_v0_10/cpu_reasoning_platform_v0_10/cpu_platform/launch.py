"""One cross-platform, hardware-aware CPU-only llama.cpp launcher.

Replaces per-OS hardcoded settings. It detects the machine, picks threads and
context size, hides every GPU from the llama.cpp process, binds to loopback,
keeps the server log, and writes a launch manifest that records what was
actually started plus any GPU activity found in the server's own log.

    python -m cpu_platform.launch --model qwen3-0.6b               # port 8080
    python -m cpu_platform.launch --model qwen3-4b --port 8081     # comparison model
    python -m cpu_platform.launch --list-models
    python -m cpu_platform.launch --model qwen3-4b --dry-run

The GPU check is evidence from logs and environment, NOT hardware attestation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

from .hardware import GB, MODEL_PRESETS, detect, model_fits

ROOT = Path(__file__).resolve().parent.parent

# Environment variables that hide devices from each GPU backend llama.cpp can be built with.
GPU_HIDING_ENV = {
    "CUDA_VISIBLE_DEVICES": "-1",       # NVIDIA CUDA
    "HIP_VISIBLE_DEVICES": "-1",        # AMD ROCm/HIP
    "ROCR_VISIBLE_DEVICES": "-1",       # AMD ROCm runtime
    "ONEAPI_DEVICE_SELECTOR": "opencl:cpu",  # Intel SYCL: CPU-only selector
    "GGML_VK_VISIBLE_DEVICES": "",      # Vulkan (llama.cpp-specific)
}

# Strong evidence that a GPU was visible to, or used by, the server.
GPU_USE_PATTERNS = [
    r"offloaded [1-9]\d*/\d+ layers to GPU",
    r"found [1-9]\d* (?:CUDA|ROCm|HIP|Vulkan|SYCL) devices?",
    r"using device (?:CUDA|ROCm|Vulkan|SYCL|Metal|MTL)\d*",
    r"ggml_metal_init: (?:found device|picking default device)",
    r"ggml_vulkan: Found [1-9]\d* Vulkan devices?",
    r"(?:CUDA|ROCm|Vulkan|Metal|SYCL)\d* (?:model )?buffer size\s*=\s*[1-9]",
]
GPU_BACKEND_MENTION = r"\b(?:CUDA|ROCm|HIP|Vulkan|Metal|SYCL|cuBLAS)\b"


def find_llama(explicit: str | None = None) -> list[str] | None:
    """Return the argv prefix that starts llama.cpp's HTTP server."""
    candidates: list[list[str]] = []
    if explicit:
        candidates.append([explicit])
    if os.environ.get("LLAMA_SERVER"):
        candidates.append([os.environ["LLAMA_SERVER"]])
    exe = ".exe" if os.name == "nt" else ""
    vendor = Path(os.environ.get("LLAMA_DIR", ROOT / "vendor" / "llama.cpp"))
    candidates += [
        [str(ROOT / "vendor" / "llama-cpu" / f"llama-server{exe}")],    # unpacked CPU-only release
        [str(vendor / "build-cpu" / "bin" / f"llama-server{exe}")],  # scripts/build_cpu_llama.sh
        [str(vendor / "build-cpu" / "bin" / "Release" / f"llama-server{exe}")],
    ]
    if shutil.which("llama-server"):
        candidates.append([shutil.which("llama-server")])
    if shutil.which("llama"):
        candidates.append([shutil.which("llama"), "serve"])  # newer unified CLI
    for cand in candidates:
        if Path(cand[0]).is_file():
            return cand
    return None


def server_help(prefix: list[str]) -> str:
    try:
        out = subprocess.run(prefix + ["--help"], capture_output=True, text=True, timeout=30)
        return (out.stdout or "") + (out.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return ""


def server_version(prefix: list[str]) -> str | None:
    try:
        out = subprocess.run(prefix + ["--version"], capture_output=True, text=True, timeout=30)
        text = ((out.stdout or "") + (out.stderr or "")).strip()
        return text.splitlines()[-1][:200] if text else None
    except (OSError, subprocess.SubprocessError):
        return None


def supports(help_text: str, flag: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", help_text) is not None


def build_command(prefix: list[str], help_text: str, *, model_source: tuple[str, str],
                  port: int, threads: int, ctx_size: int, allow_gpu: bool,
                  is_qwen3: bool) -> tuple[list[str], list[str]]:
    """Return (argv, notes). Optional flags are added only when this build lists them."""
    kind, value = model_source
    cmd = list(prefix)
    cmd += (["--hf-repo", value] if kind == "hf" else ["--model", value])
    cmd += ["--n-gpu-layers", "0", "--threads", str(threads), "--ctx-size", str(ctx_size),
            "--host", "127.0.0.1", "--port", str(port)]
    notes = []
    if supports(help_text, "--threads-batch"):
        cmd += ["--threads-batch", str(threads)]
    if supports(help_text, "--parallel"):
        cmd += ["--parallel", "1"]
    if not allow_gpu:
        if supports(help_text, "--device"):
            # Stops prompt processing from silently using a GPU even with 0 offloaded layers.
            cmd += ["--device", "none"]
        else:
            notes.append("This llama.cpp build has no --device flag; relying on env hiding and log checks.")
    if is_qwen3:
        # v0.8: thinking is decided PER REQUEST (chat_template_kwargs.enable_thinking, plus a
        # per-request reasoning_budget_tokens), so the server must not force it off. Older
        # launchers passed '--reasoning off'; the client still sends /no_think as a fallback.
        if supports(help_text, "--jinja"):
            cmd += ["--jinja"]
        if supports(help_text, "--reasoning"):
            cmd += ["--reasoning", "auto"]
        elif not supports(help_text, "--reasoning-budget"):
            notes.append("This llama.cpp build has no reasoning-budget support; thinking arms "
                         "(small_think*) need a newer build.")
    return cmd, notes


def scan_log_for_gpu(text: str) -> dict:
    evidence = sorted({m.group(0) for p in GPU_USE_PATTERNS for m in re.finditer(p, text)})
    mentions = sorted({line.strip()[:160] for line in text.splitlines()
                       if re.search(GPU_BACKEND_MENTION, line)})[:20]
    verdict = ("gpu_visible_or_used" if evidence else
               "gpu_backend_mentioned_not_used" if mentions else "no_gpu_backend_seen")
    return {"verdict": verdict, "evidence": evidence, "backend_mentions": mentions}


def wait_ready(port: int, proc: subprocess.Popen, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(1)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hardware-aware CPU-only llama.cpp launcher")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", default="qwen3-0.6b", help="preset name (see --list-models)")
    source.add_argument("--hf", help="custom Hugging Face REPO:QUANT")
    source.add_argument("--model-file", help="local .gguf file")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--threads", type=int, help="override detected physical cores")
    parser.add_argument("--ctx-size", type=int, help="override detected context size")
    parser.add_argument("--llama-bin", help="path to llama-server (default: auto-detect)")
    parser.add_argument("--allow-gpu", action="store_true",
                        help="do NOT hide GPUs; the manifest then records a non-CPU-only run")
    parser.add_argument("--force", action="store_true", help="launch even if the model may not fit in RAM")
    parser.add_argument("--ready-timeout", type=int, default=1800, help="seconds to wait (includes download)")
    parser.add_argument("--dry-run", action="store_true", help="print the command and manifest, then exit")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args(argv)

    hw = detect(measure_bandwidth=not args.dry_run)
    rec = hw["recommended"]
    total_ram = hw["memory"]["total_bytes"]

    if args.list_models:
        for p in MODEL_PRESETS.values():
            fit = "fits" if model_fits(p, total_ram, rec["ctx_size"]) else "too large for this RAM"
            est = (rec["token_rate_estimate_per_s"] or {}).get(p.name)
            print(f"{p.name:12s} {p.hf:32s} ~{p.gb:5.2f} GB  {fit}"
                  + (f"  ~{est} tok/s est." if est else ""))
        print(f"\nDetected: {hw['cpu']['physical_cores']} physical cores, {hw['memory']['total_gb']} GB RAM,"
              f" GPUs: {', '.join(g['name'] for g in hw['gpus']) or 'none'}")
        return 0

    if not 1 <= args.port <= 65535:
        parser.error("invalid port")
    threads = args.threads or rec["threads"]
    ctx_size = args.ctx_size or rec["ctx_size"]
    preset = None
    if args.model_file:
        if not Path(args.model_file).is_file():
            parser.error(f"model file not found: {args.model_file}")
        model_source, label = ("file", str(Path(args.model_file).resolve())), Path(args.model_file).name
        size = Path(args.model_file).stat().st_size
        if total_ram and size + GB > 0.6 * total_ram and not args.force:
            parser.error("model file is too large for 60% of RAM; use --force to override")
    elif args.hf:
        model_source, label = ("hf", args.hf), args.hf
    else:
        preset = MODEL_PRESETS.get(args.model)
        if preset is None:
            parser.error(f"unknown preset {args.model!r}; choose from {', '.join(MODEL_PRESETS)}")
        if not model_fits(preset, total_ram, ctx_size) and not args.force:
            parser.error(f"{preset.name} (~{preset.gb} GB) may not fit in this machine's RAM "
                         f"({hw['memory']['total_gb']} GB); pick a smaller model or pass --force")
        model_source, label = ("hf", preset.hf), preset.name
    is_qwen3 = "qwen3" in label.lower()

    prefix = find_llama(args.llama_bin)
    if prefix is None:
        print("llama.cpp server not found. Install it (e.g. 'winget install llama.cpp' on Windows,\n"
              "'brew install llama.cpp' on macOS) or build a CPU-only one with scripts/build_cpu_llama.sh,\n"
              "or pass --llama-bin / set LLAMA_SERVER.", file=sys.stderr)
        if not args.dry_run:
            return 2
        prefix = ["llama-server"]
    help_text = server_help(prefix) if Path(prefix[0]).is_file() else ""
    cmd, notes = build_command(prefix, help_text, model_source=model_source, port=args.port,
                               threads=threads, ctx_size=ctx_size, allow_gpu=args.allow_gpu,
                               is_qwen3=is_qwen3)
    if not help_text:
        notes = ["llama.cpp not found or --help unreadable: build-dependent flags "
                 "(--device none, --reasoning off, --threads-batch, --parallel) are omitted here."]
    env_overrides = {} if args.allow_gpu else dict(GPU_HIDING_ENV)
    results = Path(args.results_dir)
    log_path = results / f"llama_server_{args.port}.log"
    manifest_path = results / f"launch_{args.port}.json"
    manifest = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"label": label, "source": model_source[0], "value": model_source[1],
                  "preset": preset.name if preset else None,
                  "params_b": preset.params_b if preset else None,
                  "approx_bytes": preset.approx_bytes if preset else None},
        "llama_cpp": {"argv": cmd, "version": server_version(prefix) if help_text else None,
                      "port": args.port, "log_file": str(log_path)},
        "settings": {"threads": threads, "ctx_size": ctx_size, "parallel_slots": 1},
        "cpu_only": {"requested": not args.allow_gpu, "n_gpu_layers": 0,
                     "device_none_flag": "--device" in cmd and not args.allow_gpu,
                     "gpu_env_hidden": sorted(env_overrides), "log_scan": None,
                     "hardware_attested": False},
        "hardware": hw,
        "notes": notes,
    }
    print("Detected:", hw["cpu"]["name"], f"| {hw['cpu']['physical_cores']} physical cores |",
          f"{hw['memory']['total_gb']} GB RAM | GPUs:", ", ".join(g["name"] for g in hw["gpus"]) or "none")
    if hw["gpus"] and not args.allow_gpu:
        print("GPUs present: hiding them from llama.cpp (env + --device none when supported).")
    print("Command:", " ".join(cmd))
    for note in notes:
        print("Note:", note)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    results.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **env_overrides}
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"Started llama.cpp (PID {proc.pid}); log: {log_path}. Waiting for http://127.0.0.1:{args.port}/health ...")
    manifest["llama_cpp"]["pid"] = proc.pid
    ready = wait_ready(args.port, proc, args.ready_timeout)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    scan = scan_log_for_gpu(log_text)
    manifest["cpu_only"]["log_scan"] = scan
    manifest["ready"] = ready
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not ready:
        print("Server did not become ready. Last log lines:", file=sys.stderr)
        print("\n".join(log_text.splitlines()[-25:]), file=sys.stderr)
        if proc.poll() is None:
            proc.terminate()
        return 1
    print(f"Ready. Manifest: {manifest_path}")
    if scan["verdict"] == "gpu_visible_or_used" and not args.allow_gpu:
        print("WARNING: the server log shows GPU visibility/use:", *scan["evidence"], sep="\n  ", file=sys.stderr)
        print("This run must not be reported as CPU-only. Use a CPU-only build (scripts/build_cpu_llama.sh).",
              file=sys.stderr)
    else:
        print(f"GPU log check: {scan['verdict']}. Also confirm GPU utilisation stays at 0 in your OS monitor.")
    print(f"Server PID {proc.pid} (experiments read it from {manifest_path} to measure CPU-seconds).")
    print(BANNER.format(port=args.port))
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        print("\nCtrl+C received: stopping the llama.cpp server ...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("llama.cpp server stopped. Anything still running against "
              f"port {args.port} will now fail with 'connection refused'.")
        return 0
    print(f"llama.cpp server EXITED on its own (exit code {code}). Last log lines:", file=sys.stderr)
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
    print("\n".join(tail), file=sys.stderr)
    return code or 1


BANNER = """
================================================================================
  MODEL SERVER RUNNING on http://127.0.0.1:{port}
  KEEP THIS WINDOW OPEN. Run experiments in a SECOND terminal window.
  Pressing Ctrl+C here STOPS the model server.
================================================================================"""


if __name__ == "__main__":
    sys.exit(main())
