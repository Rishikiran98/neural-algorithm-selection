# Hardware detection and the CPU-only launcher (v0.5.2)

Nothing is tuned for a particular machine any more. `cpu_platform.hardware` inspects
the computer it runs on, and `cpu_platform.launch` starts llama.cpp from that profile,
the same way on Windows, Linux and macOS.

```bash
python -m cpu_platform.hardware              # what was detected + recommended settings
python -m cpu_platform.launch --list-models  # presets, whether each fits in RAM, speed estimate
python -m cpu_platform.launch --model qwen3-0.6b               # serve on 127.0.0.1:8080
python -m cpu_platform.launch --model qwen3-4b --port 8081     # second model for experiment 6
python -m cpu_platform.launch --model qwen3-4b --dry-run       # print command + manifest only
```

`scripts/serve_windows_cpu.ps1` and `scripts/serve_cpu.sh` are thin wrappers that pass
their arguments to the launcher.

## What is detected

| Item | How | Used for |
| --- | --- | --- |
| Physical cores | `/proc/cpuinfo`/`lscpu`, `sysctl hw.physicalcpu`, `Win32_Processor.NumberOfCores` | `--threads`, `--threads-batch` (hyperthreads usually slow token generation) |
| RAM | `/proc/meminfo`, `sysctl hw.memsize`, `GlobalMemoryStatusEx` | context size; which models fit (weights + headroom ≤ 60% of RAM) |
| GPUs | `nvidia-smi`, `Win32_VideoController`, `/sys/class/drm`, Apple Silicon | hiding every GPU, including integrated ones (Vulkan can use them) |
| Copy bandwidth | ~1 s single-thread copy of 256 MB | rough tokens/s estimate (bandwidth / model size) |

The tokens/s figure is an order-of-magnitude estimate, not a measurement. Generating a
token on a CPU streams all the weights through memory once, so bandwidth / model size is
what makes a small model so much cheaper per token. Experiment 6 is built on that.

## How "CPU-only" is enforced and recorded

1. `--n-gpu-layers 0` (no layers offloaded).
2. `--device none` when the build supports it. This stops prompt processing from quietly
   using a GPU in CUDA/Vulkan builds even with 0 offloaded layers.
3. Environment variables that hide devices from every backend: `CUDA_VISIBLE_DEVICES`,
   `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `ONEAPI_DEVICE_SELECTOR`, `GGML_VK_VISIBLE_DEVICES`.
4. The server log is saved to `results/llama_server_<port>.log` and scanned for GPU
   evidence ("found 1 CUDA devices", "using device CUDA0", "offloaded 12/29 layers to GPU").
   The verdict goes into `results/launch_<port>.json`:
   - `no_gpu_backend_seen`: a CPU-only build.
   - `gpu_backend_mentioned_not_used`: a GPU-capable build with no device visible or used. Acceptable, but say so when reporting.
   - `gpu_visible_or_used`: do **not** report the run as CPU-only. Use a CPU-only build (`scripts/build_cpu_llama.sh`).

This is evidence, not hardware attestation. Also watch GPU utilisation in your OS monitor
(Task Manager → Performance → GPU on Windows) during a run. `--allow-gpu` turns all of this
off, and the manifest then records `"requested": false`.

## Qwen3 thinking mode

Qwen3 models "think" before answering by default, and that can use up the whole token
budget before any JSON appears, so every mode would score near zero. The launcher passes
`--reasoning off` (or `--reasoning-budget 0` on older builds). The client also appends
`/no_think` for Qwen3 models and strips any `<think>` block. All modes are affected equally.

## The launch manifest

`results/launch_<port>.json` records the exact argv, llama.cpp version, threads, context
size, model source, PID, the full hardware profile and the GPU log scan. Experiments 4 and
6 read it: `--launch-manifest` (experiment 4), or automatically by port (experiment 6),
which also uses the PID to measure the server's CPU-seconds when `psutil` is installed.

## Model presets

`qwen3-0.6b` (Q8_0, ~0.64 GB), `qwen3-1.7b` (Q8_0, ~1.8 GB), `qwen3-4b` (Q4_K_M, ~2.5 GB),
`qwen3-8b` (Q4_K_M, ~5.0 GB), `qwen3-14b` (Q4_K_M, ~9.0 GB). Sizes are approximate and the
upstream repositories can change, so use `--hf REPO:QUANT` or `--model-file x.gguf` for
anything else. The launcher refuses a model that won't fit in RAM unless you pass `--force`.
