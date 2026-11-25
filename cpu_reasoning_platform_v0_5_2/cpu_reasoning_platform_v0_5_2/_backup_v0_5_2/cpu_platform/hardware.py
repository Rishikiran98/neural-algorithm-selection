"""Portable hardware detection and a recommended CPU-only inference profile.

Standard library only. Works on Windows, Linux and macOS; every probe is optional
and failures degrade to None instead of raising. Nothing here is hardware
attestation: it records what the operating system reports.

    python -m cpu_platform.hardware            # print the detected profile as JSON
    python -m cpu_platform.hardware --no-bandwidth
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

GB = 1024 ** 3


@dataclass(frozen=True)
class ModelPreset:
    name: str
    hf: str                # llama.cpp --hf-repo value, REPO:QUANT
    params_b: float        # parameters, billions
    approx_bytes: int      # approximate GGUF file size; verify against the actual download

    @property
    def gb(self) -> float:
        return round(self.approx_bytes / GB, 2)


# Sizes are approximate published GGUF sizes. Repos/quantizations can change upstream;
# pass --hf or --model-file to the launcher to use anything else.
MODEL_PRESETS: dict[str, ModelPreset] = {p.name: p for p in (
    ModelPreset("qwen3-0.6b", "Qwen/Qwen3-0.6B-GGUF:Q8_0", 0.6, int(0.64 * GB)),
    ModelPreset("qwen3-1.7b", "Qwen/Qwen3-1.7B-GGUF:Q8_0", 1.7, int(1.83 * GB)),
    ModelPreset("qwen3-4b", "Qwen/Qwen3-4B-GGUF:Q4_K_M", 4.0, int(2.50 * GB)),
    ModelPreset("qwen3-8b", "Qwen/Qwen3-8B-GGUF:Q4_K_M", 8.2, int(5.03 * GB)),
    ModelPreset("qwen3-14b", "Qwen/Qwen3-14B-GGUF:Q4_K_M", 14.8, int(9.00 * GB)),
)}
SMALL_DEFAULT = "qwen3-0.6b"


def _run(cmd: list[str], timeout: float = 8.0) -> str | None:
    """Run a probe command; return stdout or None. Never raises."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return out.stdout if out.returncode == 0 else None


def _powershell(script: str) -> str | None:
    exe = "powershell" if shutil.which("powershell") else "pwsh"
    return _run([exe, "-NoProfile", "-NonInteractive", "-Command", script], timeout=15)


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return None


# ---------------------------------------------------------------- CPU

def cpu_name() -> str | None:
    system = platform.system()
    if system == "Linux":
        text = _read("/proc/cpuinfo") or ""
        m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
        if m:
            return m.group(1).strip()
    elif system == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out and out.strip():
            return out.strip()
    elif system == "Windows":
        out = _powershell("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name")
        if out and out.strip():
            return out.strip()
    return platform.processor() or None


def parse_proc_cpuinfo_cores(text: str) -> int | None:
    """Count unique (physical id, core id) pairs; None if the fields are absent."""
    cores, phys, core = set(), None, None
    for line in text.splitlines() + [""]:
        if not line.strip():
            if core is not None:
                cores.add((phys, core))
            phys, core = None, None
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "physical id":
            phys = value.strip()
        elif key == "core id":
            core = value.strip()
    return len(cores) or None


def physical_cores() -> tuple[int | None, str]:
    """Return (physical core count, source). Hyperthreads are excluded where possible."""
    system = platform.system()
    if system == "Linux":
        n = parse_proc_cpuinfo_cores(_read("/proc/cpuinfo") or "")
        if n:
            return n, "proc_cpuinfo"
        out = _run(["lscpu", "-p=CORE,SOCKET"])
        if out:
            pairs = {line for line in out.splitlines() if line and not line.startswith("#")}
            if pairs:
                return len(pairs), "lscpu"
    elif system == "Darwin":
        out = _run(["sysctl", "-n", "hw.physicalcpu"])
        if out and out.strip().isdigit():
            return int(out.strip()), "sysctl"
    elif system == "Windows":
        out = _powershell("(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum")
        if out and out.strip().isdigit():
            return int(out.strip()), "cim"
    return None, "unknown"


# ---------------------------------------------------------------- memory

def memory_bytes() -> tuple[int | None, int | None]:
    """Return (total, available) physical RAM in bytes."""
    system = platform.system()
    if system == "Windows":
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
                return int(state.ullTotalPhys), int(state.ullAvailPhys)
        except (AttributeError, OSError):
            pass
        return None, None
    if system == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        total = int(out.strip()) if out and out.strip().isdigit() else None
        return total, None
    text = _read("/proc/meminfo") or ""
    values = {k: int(v) * 1024 for k, v in re.findall(r"^(\w+):\s+(\d+) kB", text, re.M)}
    return values.get("MemTotal"), values.get("MemAvailable")


def measure_copy_bandwidth(size_mb: int = 256, repeats: int = 3) -> float | None:
    """Single-thread memory copy bandwidth in bytes/s (read + write traffic).

    Both buffers are pre-faulted so page faults are not timed; the best run is kept.
    One thread usually reaches well under the multi-core peak that llama.cpp uses,
    so treat this as a LOWER bound for order-of-magnitude estimates only.
    """
    try:
        size = size_mb * 1024 * 1024
        source = bytearray(b"\x01") * size
        dest = bytearray(b"\x02") * size
        best = None
        for _ in range(repeats):
            start = time.perf_counter()
            dest[:] = source
            elapsed = time.perf_counter() - start
            if elapsed > 0:
                rate = 2 * size / elapsed
                best = rate if best is None else max(best, rate)
        del source, dest
        return best
    except MemoryError:
        return None


# ---------------------------------------------------------------- GPUs

VENDOR_IDS = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x8086": "Intel", "0x106b": "Apple"}


def gpus() -> list[dict]:
    """Every GPU the OS reports. Any of them may be usable by a llama.cpp build
    (CUDA, ROCm/HIP, Vulkan, SYCL or Metal), including integrated graphics."""
    found: list[dict] = []
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if smi:
        for line in smi.strip().splitlines():
            name, _, mem = line.partition(",")
            found.append({"name": name.strip(), "vendor": "NVIDIA",
                          "memory_mb": int(mem.strip()) if mem.strip().isdigit() else None,
                          "source": "nvidia-smi"})
    system = platform.system()
    if system == "Windows":
        out = _powershell("Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }")
        for name in (out or "").splitlines():
            name = name.strip()
            if not name or any(g["name"] == name for g in found):
                continue
            if re.search(r"basic display|remote display|virtual|parsec|idd", name, re.I):
                continue
            vendor = next((v for v in ("NVIDIA", "AMD", "Intel") if v.lower() in name.lower()), None)
            if vendor is None and "radeon" in name.lower():
                vendor = "AMD"
            found.append({"name": name, "vendor": vendor, "memory_mb": None, "source": "cim"})
    elif system == "Linux":
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
            if "-" in card.name:
                continue  # connector entries such as card0-HDMI-A-1
            vendor_id = (_read(str(card / "device" / "vendor")) or "").strip()
            vendor = VENDOR_IDS.get(vendor_id)
            if vendor is None or (vendor == "NVIDIA" and smi):
                continue
            found.append({"name": f"{vendor} GPU ({card.name})", "vendor": vendor,
                          "memory_mb": None, "source": "sysfs"})
    elif system == "Darwin" and platform.machine() == "arm64":
        found.append({"name": "Apple Silicon GPU (Metal)", "vendor": "Apple", "memory_mb": None,
                      "source": "platform"})
    return found


# ---------------------------------------------------------------- profile

def model_fits(preset: ModelPreset, total_ram: int | None, ctx_size: int) -> bool:
    """Weights + KV cache/runtime headroom must fit in 60% of RAM (OS and API need the rest)."""
    if total_ram is None:
        return preset.name == SMALL_DEFAULT
    overhead = int(0.5 * GB) + ctx_size * 160 * 1024  # generous KV-cache allowance
    return preset.approx_bytes + overhead <= 0.6 * total_ram


def recommend(total_ram: int | None, cores: int | None, logical: int | None,
              bandwidth: float | None) -> dict:
    threads = cores or max(1, (logical or 2) // 2)
    threads = max(1, min(threads, 64))
    ctx = 4096 if (total_ram or 0) >= 7.5 * GB else 2048
    feasible = [p for p in MODEL_PRESETS.values() if model_fits(p, total_ram, ctx)]
    # The comparison model is the largest feasible preset up to 8B: larger ones are
    # usually too slow on laptop CPUs for a 100-question benchmark.
    large = max((p for p in feasible if p.params_b <= 8.5 and p.name != SMALL_DEFAULT),
                key=lambda p: p.params_b, default=None)
    estimates = {}
    if bandwidth:
        for p in MODEL_PRESETS.values():
            # Token generation streams every weight once per token, so tokens/s is roughly
            # bandwidth / size. The bandwidth input is a single-thread lower bound.
            estimates[p.name] = round(bandwidth / p.approx_bytes, 1)
    return {
        "threads": threads,
        "threads_batch": threads,
        "ctx_size": ctx,
        "max_inflight": 1,
        "parallel_slots": 1,
        "small_model": SMALL_DEFAULT,
        "large_model": large.name if large else None,
        "feasible_models": [p.name for p in feasible],
        "token_rate_estimate_per_s": estimates or None,
        "notes": [
            "threads = physical cores; hyperthreads usually slow down token generation",
            "token_rate_estimate = single-thread copy bandwidth / model size: order of magnitude only, not a measurement",
        ],
    }


def detect(measure_bandwidth: bool = True) -> dict:
    total, available = memory_bytes()
    cores, cores_source = physical_cores()
    logical = os.cpu_count()
    bandwidth = measure_copy_bandwidth() if measure_bandwidth else None
    gpu_list = gpus()
    return {
        "os": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu": {"name": cpu_name(), "physical_cores": cores, "physical_cores_source": cores_source,
                "logical_cpus": logical},
        "memory": {"total_bytes": total, "available_bytes": available,
                   "total_gb": round(total / GB, 1) if total else None,
                   "single_thread_copy_bandwidth_gb_s": round(bandwidth / 1e9, 1) if bandwidth else None},
        "gpus": gpu_list,
        "gpu_present": bool(gpu_list),
        "recommended": recommend(total, cores, logical, bandwidth),
    }


def main():
    parser = argparse.ArgumentParser(description="Detect hardware and print a CPU-only inference profile")
    parser.add_argument("--no-bandwidth", action="store_true", help="skip the ~1 s memory copy test")
    args = parser.parse_args()
    print(json.dumps(detect(measure_bandwidth=not args.no_bandwidth), indent=2))


if __name__ == "__main__":
    main()
