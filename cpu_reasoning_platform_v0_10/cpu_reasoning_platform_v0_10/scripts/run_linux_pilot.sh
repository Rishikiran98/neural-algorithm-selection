#!/usr/bin/env bash
# v0.8 pilot (see scripts/run_windows_pilot.ps1 for the two-server setup).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python3 -m cpu_platform.experiment6 --dataset data/gsm8k_test.jsonl --limit 100 --output results/pilot_v08.jsonl "$@"
