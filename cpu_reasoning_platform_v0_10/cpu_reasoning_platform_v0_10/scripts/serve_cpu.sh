#!/usr/bin/env bash
# CPU-only llama.cpp server on Linux/macOS, configured from detected hardware.
# Uses vendor/llama.cpp/build-cpu (from scripts/build_cpu_llama.sh) if present, else llama-server on PATH.
#   bash scripts/serve_cpu.sh                          # qwen3-0.6b on 8080
#   bash scripts/serve_cpu.sh --model qwen3-4b --port 8081
#   bash scripts/serve_cpu.sh --list-models
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python3 -m cpu_platform.launch "$@"
