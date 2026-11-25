"""Optional GSM8K test-set downloader. Installs datasets separately."""
import argparse
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/gsm8k_test.jsonl")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install first: pip install datasets") from exc
    test = load_dataset("openai/gsm8k", "main", split="test")
    indices = random.Random(args.seed).sample(range(len(test)), min(args.limit, len(test)))
    dest = Path(args.output)
    dest.parent.mkdir(exist_ok=True, parents=True)
    with dest.open("w", encoding="utf-8") as f:
        for i in indices:
            row = test[int(i)]
            f.write(json.dumps({"id": str(i), "question": row["question"], "answer": row["answer"]}) + "\n")
    print(f"Wrote {len(indices)} items to {dest}")


if __name__ == "__main__":
    main()
