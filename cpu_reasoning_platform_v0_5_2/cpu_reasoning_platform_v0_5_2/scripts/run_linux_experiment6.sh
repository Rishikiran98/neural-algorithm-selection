#!/usr/bin/env bash
# Experiment 006 (see scripts/run_windows_experiment6.ps1 for the two-server setup).
#   bash scripts/serve_cpu.sh --model qwen3-0.6b --port 8080 &
#   bash scripts/serve_cpu.sh --model qwen3-4b  --port 8081 &
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python3 -m cpu_platform.experiment6 --output results/experiment6.jsonl "$@"
