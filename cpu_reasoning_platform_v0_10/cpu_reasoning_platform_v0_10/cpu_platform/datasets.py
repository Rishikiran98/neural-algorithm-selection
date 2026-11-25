"""v0.9 dataset registry: frozen, disjoint development and evaluation splits.

    python -m cpu_platform.datasets freeze      # build data/splits/{dev,eval}.jsonl + manifest
    python -m cpu_platform.datasets validate    # dry-run check of every question, answer and hash
    python -m cpu_platform.datasets status      # what exists, and how often eval has been run

Rules this module enforces:
- dev = the questions already used for pilots (data/gsm8k_test.jsonl), so earlier runs
  stay valid as development data. Tune prompts, budgets and checker policies ONLY on dev.
- eval = a seeded sample from the remaining questions, excluding exact duplicates, near
  duplicates (also near duplicates of any dev question) and anything echoed in a prompt.
  It is frozen: re-freezing needs --force, and every experiment run on it is logged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys

from cpu_first.core import gsm8k_gold

SPLITS_DIR = Path("data/splits")
MANIFEST = "manifest.json"
EVAL_LOG = "eval_runs.jsonl"
NEAR_DUPLICATE_JACCARD = 0.7
PROMPT_ECHO_JACCARD = 0.5
SHINGLE = 3


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9$%.]+", " ", text.lower())).strip()


def question_sha256(question: str) -> str:
    """Stable fingerprint of question text, independent of row ID and whitespace/case."""
    return hashlib.sha256(normalize(question).encode("utf-8")).hexdigest()


class FrozenRegistryError(ValueError):
    """A frozen split cannot be trusted. Refuse to classify a dataset as safe."""


def shingles(text: str, n: int = SHINGLE) -> set[tuple[str, ...]]:
    words = normalize(text).split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                if "question" not in row or "answer" not in row:
                    raise ValueError(f"{path}:{number}: missing question/answer")
                row.setdefault("id", str(number - 1))
                row["id"] = str(row["id"])
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps({"id": row["id"], "question": row["question"], "answer": row["answer"]},
                               ensure_ascii=False) + "\n")


def prompt_texts() -> list[str]:
    """Every built-in prompt (their worked examples must not appear in evaluation)."""
    from . import constrained, engine, search
    from cpu_first import core
    texts = [v for v in search.PROMPTS.values()]
    texts += [engine.PROOF_PROMPT, engine.REPAIR_PROMPT, core.BASELINE_PROMPT, core.HYBRID_PROMPT,
              constrained.CONSTRAINED_PLAN_PROMPT]
    return texts


def clean(rows: list[dict]) -> tuple[list[dict], dict]:
    """Drop exact duplicates, near duplicates and prompt echoes; return (kept, removed ids)."""
    removed = {"exact_duplicates": [], "near_duplicates": [], "prompt_echo": [], "invalid_gold": []}
    prompts = [(normalize(p), shingles(p)) for p in prompt_texts()]
    seen_exact: dict[str, str] = {}
    kept: list[dict] = []
    kept_shingles: list[set] = []
    for row in rows:
        try:
            gsm8k_gold(row["answer"])
        except ValueError:
            removed["invalid_gold"].append(row["id"])
            continue
        norm = normalize(row["question"])
        if norm in seen_exact:
            removed["exact_duplicates"].append(row["id"])
            continue
        sh = shingles(row["question"])
        if any(norm in p_norm or jaccard(sh, p_sh) >= PROMPT_ECHO_JACCARD
               or len(sh & p_sh) >= max(3, len(sh) // 2) for p_norm, p_sh in prompts):
            removed["prompt_echo"].append(row["id"])
            continue
        if any(jaccard(sh, other) >= NEAR_DUPLICATE_JACCARD for other in kept_shingles):
            removed["near_duplicates"].append(row["id"])
            continue
        seen_exact[norm] = row["id"]
        kept.append(row)
        kept_shingles.append(sh)
    return kept, removed


def freeze(source: Path, dev_file: Path | None, out_dir: Path, *, eval_size: int, dev_size: int,
           seed: int, force: bool = False) -> dict:
    manifest_path = out_dir / MANIFEST
    if manifest_path.exists() and not force:
        raise SystemExit(f"{manifest_path} exists: the evaluation split is already frozen. "
                         f"Re-freezing changes the held-out set; pass --force only if you mean it.")
    rows = read_jsonl(source)
    wanted = ([r["id"] for r in read_jsonl(dev_file)]
              if dev_file is not None and dev_file.is_file() else None)
    if wanted:
        # Pilot questions were already run: clean them FIRST so that when two questions are
        # duplicates, the one already used is kept and the other is dropped.
        first = set(wanted)
        rows = [r for r in rows if r["id"] in first] + [r for r in rows if r["id"] not in first]
    kept, removed = clean(rows)
    by_id = {r["id"]: r for r in kept}
    if wanted:
        dev = [by_id[i] for i in wanted if i in by_id]
        dev_origin = f"ids of {dev_file} (questions already used for pilots)"
    else:
        pool = sorted(by_id)
        dev = [by_id[i] for i in random.Random(seed).sample(pool, min(dev_size, len(pool)))]
        dev_origin = f"seeded sample (seed {seed})"
    dev_ids = {r["id"] for r in dev}
    dev_sh = [shingles(r["question"]) for r in dev]
    candidates = [r for r in kept if r["id"] not in dev_ids
                  and not any(jaccard(shingles(r["question"]), s) >= NEAR_DUPLICATE_JACCARD for s in dev_sh)]
    if len(candidates) < eval_size:
        raise SystemExit(f"Only {len(candidates)} clean questions remain for eval; lower --eval-size.")
    rng = random.Random(seed + 1)
    eval_rows = sorted(rng.sample(candidates, eval_size), key=lambda r: int(r["id"]) if r["id"].isdigit() else r["id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "dev.jsonl", dev)
    write_jsonl(out_dir / "eval.jsonl", eval_rows)
    source_info = None
    info_path = source.parent / "gsm8k_source.json"
    if info_path.is_file():
        source_info = json.loads(info_path.read_text(encoding="utf-8"))
    manifest = {
        "schema": "cpu-reasoning-splits/2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"file": str(source), "sha256": sha256_text(source), "rows": len(rows), "provenance": source_info},
        "cleaning": {"near_duplicate_jaccard": NEAR_DUPLICATE_JACCARD, "prompt_echo_jaccard": PROMPT_ECHO_JACCARD,
                     "shingle_words": SHINGLE, "removed": removed,
                     "removed_count": sum(len(v) for v in removed.values())},
        "seed": seed,
        "splits": {
            "dev": {"file": str(out_dir / "dev.jsonl"), "sha256": sha256_text(out_dir / "dev.jsonl"),
                    "count": len(dev), "ids": [r["id"] for r in dev],
                    "question_sha256": [question_sha256(r["question"]) for r in dev],
                    "origin": dev_origin,
                    "use": "tune prompts, budgets, checker policy; look as often as needed"},
            "eval": {"file": str(out_dir / "eval.jsonl"), "sha256": sha256_text(out_dir / "eval.jsonl"),
                     "count": len(eval_rows), "ids": [r["id"] for r in eval_rows],
                     "question_sha256": [question_sha256(r["question"]) for r in eval_rows],
                     "frozen": True,
                     "use": "final comparison only, with settings fixed in advance; every run is logged"},
        },
        "contamination_note": ("GSM8K (2021) is public and may be in the models' training data. "
                               "Results on it are evidence about this pipeline, not proof of reasoning "
                               "on unseen problems; confirm on fresh or generated problems."),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_manifest(directory: Path = SPLITS_DIR) -> dict | None:
    path = directory / MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def validate(directory: Path = SPLITS_DIR) -> list[str]:
    """Return a list of problems (empty = valid). Reads every question and gold answer."""
    manifest = load_manifest(directory)
    if manifest is None:
        return [f"no {directory / MANIFEST}; run: python -m cpu_platform.datasets freeze"]
    problems = []
    ids = {}
    fingerprint_sets = {}
    for name, split in manifest["splits"].items():
        path = Path(split["file"])
        if not path.is_file():
            problems.append(f"{name}: missing {path}")
            continue
        if sha256_text(path) != split["sha256"]:
            problems.append(f"{name}: {path} was modified after freezing (sha256 mismatch)")
        try:
            rows = read_jsonl(path)
        except (ValueError, KeyError) as exc:   # includes json.JSONDecodeError
            problems.append(f"{name}: unreadable {path} ({exc})")
            continue
        if [r["id"] for r in rows] != split["ids"]:
            problems.append(f"{name}: ids differ from the manifest")
        if len({r["id"] for r in rows}) != len(rows):
            problems.append(f"{name}: duplicate ids")
        actual_fingerprints = [question_sha256(r["question"]) for r in rows]
        if "question_sha256" in split and actual_fingerprints != split["question_sha256"]:
            problems.append(f"{name}: question fingerprints differ from the manifest")
        if len(set(actual_fingerprints)) != len(rows):
            problems.append(f"{name}: duplicate normalized questions")
        fingerprint_sets[name] = set(actual_fingerprints)
        for r in rows:
            try:
                gsm8k_gold(r["answer"])
            except ValueError as exc:
                problems.append(f"{name}/{r['id']}: bad gold answer ({exc})")
            if not r["question"].strip():
                problems.append(f"{name}/{r['id']}: empty question")
        ids[name] = {r["id"] for r in rows}
    if "dev" in ids and "eval" in ids and ids["dev"] & ids["eval"]:
        problems.append(f"dev and eval overlap: {sorted(ids['dev'] & ids['eval'])[:10]}")
    if "dev" in fingerprint_sets and "eval" in fingerprint_sets:
        overlap = fingerprint_sets["dev"] & fingerprint_sets["eval"]
        if overlap:
            problems.append(f"dev and eval share {len(overlap)} normalized question(s)")
    return problems


def _trusted_fingerprints(manifest: dict, name: str) -> set[str]:
    """Read a pinned split, including legacy v1 manifests, and fail closed if altered.

    v1 manifests did not include per-question fingerprints. Compute them from the
    original, SHA256-pinned JSONL instead of requiring a dangerous re-freeze.
    """
    split = manifest["splits"][name]
    path = Path(split["file"])
    if not path.is_file():
        raise FrozenRegistryError(f"frozen {name} split missing: {path}; restore the original file")
    if sha256_text(path) != split["sha256"]:
        raise FrozenRegistryError(f"frozen {name} split modified: {path}; restore the original file")
    try:
        rows = read_jsonl(path)
        actual = [question_sha256(row["question"]) for row in rows]
        recorded = split.get("question_sha256")
        if recorded is not None and actual != recorded:
            raise FrozenRegistryError(f"frozen {name} question fingerprints differ from manifest")
        if [row["id"] for row in rows] != split["ids"]:
            raise FrozenRegistryError(f"frozen {name} question ids differ from manifest")
    except (ValueError, KeyError, TypeError) as exc:
        raise FrozenRegistryError(f"frozen {name} split unreadable or invalid: {exc}") from exc
    return set(actual)


def split_of(dataset: Path) -> tuple[str | None, dict | None]:
    """Classify a dataset using question content, not caller-supplied IDs.

    Any single normalized frozen evaluation question in a copied, relabelled,
    reordered or merged dataset counts as evaluation. Only the original frozen
    eval file is permitted by the experiment runner, with explicit confirmation.
    Dev is recognized only when *all* questions occur in dev.

    Raises FrozenRegistryError if the frozen reference files have changed: an
    invalid registry must never turn evaluation into apparently safe development.
    """
    if not dataset.is_file():
        return None, None
    dev_manifest = None
    for directory in dict.fromkeys([dataset.parent, dataset.parent / "splits", SPLITS_DIR]):
        try:
            manifest = load_manifest(directory)
        except (ValueError, KeyError, TypeError) as exc:
            raise FrozenRegistryError(f"unreadable frozen manifest in {directory}: {exc}") from exc
        if not manifest:
            continue
        # Verify the pinned reference material before making any decision. The
        # original v1 manifest is also protected without modifying its files.
        try:
            eval_hashes = _trusted_fingerprints(manifest, "eval")
            dev_hashes = _trusted_fingerprints(manifest, "dev")
        except (KeyError, TypeError) as exc:
            raise FrozenRegistryError(f"invalid frozen manifest in {directory}: {exc}") from exc
        try:
            candidate_hashes = {question_sha256(r["question"]) for r in read_jsonl(dataset)}
        except (ValueError, KeyError, TypeError) as exc:
            raise FrozenRegistryError(f"dataset cannot be checked against frozen splits: {exc}") from exc
        if candidate_hashes & eval_hashes:
            return "eval", manifest
        if candidate_hashes and candidate_hashes <= dev_hashes:
            dev_manifest = manifest
    # Inspect every available registry before declaring a dataset to be dev.
    # Evaluation overlap always has precedence over a dev match elsewhere.
    return ("dev", dev_manifest) if dev_manifest is not None else (None, None)


def eval_runs(directory: Path = SPLITS_DIR) -> list[dict]:
    path = directory / EVAL_LOG
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def log_eval_run(entry: dict, directory: Path = SPLITS_DIR) -> int:
    """Append one eval run to the log; returns how many eval runs exist including this one."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / EVAL_LOG).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return len(eval_runs(directory))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Frozen dev/eval splits for CPU reasoning experiments")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze", help="build dev/eval splits and the manifest")
    f.add_argument("--source", default="data/gsm8k_test_full.jsonl")
    f.add_argument("--dev", default="data/gsm8k_test.jsonl",
                   help="questions already used for pilots become the dev split")
    f.add_argument("--dev-size", type=int, default=200, help="only if --dev does not exist")
    f.add_argument("--eval-size", type=int, default=500)
    f.add_argument("--seed", type=int, default=909)
    f.add_argument("--out", default=str(SPLITS_DIR))
    f.add_argument("--force", action="store_true")
    v = sub.add_parser("validate", help="dry-run check of splits")
    v.add_argument("--dir", default=str(SPLITS_DIR))
    s = sub.add_parser("status")
    s.add_argument("--dir", default=str(SPLITS_DIR))
    args = p.parse_args(argv)

    if args.cmd == "freeze":
        source = Path(args.source)
        if not source.is_file():
            print(f"ERROR: {source} not found. Run: py scripts/fetch_gsm8k.py", file=sys.stderr)
            return 2
        m = freeze(source, Path(args.dev), Path(args.out), eval_size=args.eval_size,
                   dev_size=args.dev_size, seed=args.seed, force=args.force)
        print(f"dev:  {m['splits']['dev']['count']} questions ({m['splits']['dev']['origin']})")
        print(f"eval: {m['splits']['eval']['count']} questions, FROZEN")
        print(f"removed while cleaning: {m['cleaning']['removed_count']} "
              + str({k: len(v) for k, v in m['cleaning']['removed'].items()}))
        print(f"manifest: {Path(args.out) / MANIFEST}")
        return 0
    directory = Path(args.dir)
    if args.cmd == "validate":
        problems = validate(directory)
        for line in problems:
            print("PROBLEM:", line)
        print("OK: splits valid" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    manifest = load_manifest(directory)
    if manifest is None:
        print("No frozen splits yet.")
        return 0
    for name, split in manifest["splits"].items():
        print(f"{name}: {split['count']} questions  {split['file']}")
    runs = eval_runs(directory)
    print(f"eval runs so far: {len(runs)}" + (" (each extra look weakens the held-out claim)" if len(runs) > 1 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
