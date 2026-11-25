#!/usr/bin/env bash
set -euo pipefail
python3 -m cpu_platform.experiment6 --large-url none --arms small_cot_vote,small_cot_checked --output results/experiment7.jsonl "$@"
