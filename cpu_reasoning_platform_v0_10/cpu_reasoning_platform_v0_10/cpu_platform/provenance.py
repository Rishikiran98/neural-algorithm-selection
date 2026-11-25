"""v0.9 run provenance: what exactly ran, and what was NOT measured.

Reads llama.cpp's GET /props (model_path, build_info, model_ftype, chat_template; see
tools/server/server-context.cpp) and hashes the GGUF file actually loaded. Hashes are
cached by (path, size, mtime) so a 5 GB model is hashed once, not on every run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

NOT_RUN = "NOT RUN"


def server_props(base_url: str, timeout: float = 10) -> dict | None:
    try:
        with urlopen(base_url.rstrip("/") + "/props", timeout=timeout) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def file_sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def model_sha256(path: str | None, cache_file: Path) -> tuple[str | None, str]:
    """(sha256 or None, note). Uses a (path, size, mtime) cache."""
    if not path:
        return None, "server did not report model_path"
    p = Path(path)
    if not p.is_file():
        return None, f"model_path {path} not readable from this process (different machine?)"
    stat = p.stat()
    key = f"{p.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    if key in cache:
        return cache[key], "cached"
    print(f"Hashing model weights {p.name} ({stat.st_size / 1e9:.1f} GB), once ...", flush=True)
    digest = file_sha256(p)
    cache[key] = digest
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
    return digest, "computed"


def describe_server(base_url: str, cache_file: Path) -> dict:
    props = server_props(base_url)
    if props is None:
        return {"props": NOT_RUN, "model_sha256": NOT_RUN, "build_info": NOT_RUN}
    digest, note = model_sha256(props.get("model_path"), cache_file)
    template = props.get("chat_template") or ""
    return {
        "model_path": props.get("model_path"),
        "model_sha256": digest or NOT_RUN,
        "model_sha256_note": note,
        "model_ftype": props.get("model_ftype"),
        "build_info": props.get("build_info") or NOT_RUN,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest() if template else NOT_RUN,
        "total_slots": props.get("total_slots"),
        "n_ctx": ((props.get("default_generation_settings") or {}).get("n_ctx")),
    }


def evidence_table(*, cost_unit: str, servers: dict, manifests: dict, split: str | None,
                   peak_rss_mb: dict) -> dict:
    """Each item is MEASURED / RECORDED / NOT RUN, never silently 'PASS'."""
    def all_recorded(key):
        values = [s.get(key) for s in servers.values()]
        return "RECORDED" if values and all(v and v != NOT_RUN for v in values) else NOT_RUN
    verdicts = [((m or {}).get("cpu_only") or {}).get("log_scan") or {} for m in manifests.values()]
    verdicts = [v.get("verdict") for v in verdicts if v]
    if len(verdicts) != len(manifests) or not verdicts:
        gpu = NOT_RUN
    elif any(v == "gpu_visible_or_used" for v in verdicts):
        gpu = "FAILED: GPU visible or used (" + ", ".join(verdicts) + ")"
    else:
        gpu = "RECORDED: " + ", ".join(verdicts) + " (log evidence, not hardware attestation)"
    return {
        "cost_metering": "MEASURED: model-server CPU-seconds (psutil)" if cost_unit == "cpu_seconds"
                         else "PARTIAL: wall-clock seconds only (CPU-seconds NOT RUN)",
        "model_weights_sha256": all_recorded("model_sha256"),
        "llama_cpp_build": all_recorded("build_info"),
        "cpu_only_log_check": gpu,
        "peak_model_rss": ("MEASURED: " + ", ".join(f"{k} {v:.0f} MB" for k, v in peak_rss_mb.items())
                           if peak_rss_mb and all(v is not None for v in peak_rss_mb.values()) else NOT_RUN),
        "held_out_evaluation": "RUN on frozen eval split" if split == "eval"
                               else f"{NOT_RUN} (this run used the {split or 'unregistered'} data)",
        "energy_joules": NOT_RUN,
        "thermal_throttling": NOT_RUN,
        "semantic_verification": "NOT CLAIMED: the checker verifies arithmetic and number provenance only",
    }
