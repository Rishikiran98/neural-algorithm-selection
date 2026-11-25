"""GSM8K test-set downloader (needs: pip install datasets).

Writes:
  data/gsm8k_test.jsonl        seeded random sample (default 200) for pilots / development
  data/gsm8k_test_full.jsonl   all 1319 test questions (v0.9: source for frozen splits)
  data/gsm8k_source.json       provenance: dataset id, config, split, fingerprint, versions

Question ids are the row index in the official test split, so ids are stable across
files. Then freeze disjoint dev/eval splits with:  python -m cpu_platform.datasets freeze
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/gsm8k_test.jsonl")
    parser.add_argument("--full-output", default="data/gsm8k_test_full.jsonl")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    try:
        import datasets
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install first: pip install datasets") from exc
    test = load_dataset("openai/gsm8k", "main", split="test")
    rows = [{"id": str(i), "question": test[i]["question"], "answer": test[i]["answer"]}
            for i in range(len(test))]

    full = Path(args.full_output)
    full.parent.mkdir(exist_ok=True, parents=True)
    with full.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    dest = Path(args.output)
    if dest.exists():
        # Never silently replace the sample a pilot already ran on: it defines the dev split.
        print(f"Kept existing {dest} (delete it yourself to draw a new sample).")
    else:
        indices = random.Random(args.seed).sample(range(len(rows)), min(args.limit, len(rows)))
        with dest.open("w", encoding="utf-8") as f:
            for i in indices:
                f.write(json.dumps(rows[i]) + "\n")
        print(f"Wrote {len(indices)} sampled items to {dest}")

    source = {
        "dataset": "openai/gsm8k", "config": "main", "split": "test", "rows": len(rows),
        "license": "MIT (github.com/openai/grade-school-math)",
        "fingerprint": getattr(test, "_fingerprint", None),
        "datasets_version": getattr(datasets, "__version__", None),
        "dataset_version": str(getattr(getattr(test, "info", None), "version", "") or "") or None,
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "sample": {"file": str(dest), "seed": args.seed, "limit": args.limit},
    }
    Path(full.parent / "gsm8k_source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote all {len(rows)} test items to {full} and provenance to {full.parent / 'gsm8k_source.json'}")


if __name__ == "__main__":
    main()
