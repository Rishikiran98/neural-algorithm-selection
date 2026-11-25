#!/usr/bin/env bash
# Use the same model, quantization and fixed dataset as Windows tests.
set -euo pipefail
python3 -m cpu_platform.experiment4 \
    --dataset data/smoke.jsonl --limit 5 \
    --output results/experiment4_smoke.jsonl \
    --model-id Qwen3-0.6B-GGUF --quantization Q8_0 --launch-manifest results/launch_8080.json \
    --gpu-layers 0 \
    --seed 404 --repeats 1
