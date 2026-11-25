#!/usr/bin/env bash
# Build an explicitly CPU-only llama.cpp server on macOS or Linux.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$PROJECT_ROOT/vendor/llama.cpp}"
if ! command -v git >/dev/null || ! command -v cmake >/dev/null; then
  echo 'Requires git, cmake, and a C/C++ compiler. Install those first.' >&2
  exit 1
fi
mkdir -p "$(dirname "$LLAMA_DIR")"
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build-cpu" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_HIP=OFF \
  -DGGML_VULKAN=OFF -DGGML_SYCL=OFF
cmake --build "$LLAMA_DIR/build-cpu" --config Release --target llama-server --parallel 4
printf '\nCPU-only server built at: %s\n' "$LLAMA_DIR/build-cpu/bin/llama-server"
printf 'For reproducible comparisons, record the llama.cpp git commit: '
git -C "$LLAMA_DIR" rev-parse HEAD
