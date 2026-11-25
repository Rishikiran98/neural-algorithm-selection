# Experiment 004: reproducible CPU-only comparison

## Objective

Compare the original `baseline`, `hybrid`, `adaptive`, `proof` and new `semantic` modes on the same CPU-hosted model and question set. Measure *ground-truth accuracy*, latency, API-reported token counts, model calls, semantic coverage and cost proxies. This experiment does **not** show frontier reasoning or GPU-free pretraining.

## Prerequisites

- Python 3.11+ and API running on `127.0.0.1:8070`.
- llama.cpp CPU-only model server on `127.0.0.1:8080` with **no GPU offload**. Use the existing scripts to start it with `-ngl 0` (Windows) or `--n-gpu-layers 0` (Linux).
- For a larger, meaningful evaluation, provision a dataset with independently supplied answers and set `--dataset`/`--limit` accordingly. The five bundled GSM8K-style smoke problems are **software checks, not scientific results**. Review dataset terms before use.

## Commands

Windows PowerShell, from the project root:

```powershell
powershell -NoProfile -File .\scripts\run_windows_experiment4.ps1
```

Linux:

```bash
bash scripts/run_linux_experiment4.sh
```

For a larger run, first fetch a dataset using the existing `scripts/fetch_gsm8k.py` helper (it needs the optional `datasets` package and an internet connection). Then:

```bash
python -m cpu_platform.experiment4 --dataset data/gsm8k_test.jsonl --limit 100 \
  --output results/gsm8k_100.jsonl --seed 404 --repeats 1 --warmup 2 \
  --model-id Qwen3-0.6B-GGUF --quantization Q8_0 --threads 4 --context-size 2048 \
  --gpu-layers 0 --model-file /path/to/downloaded/weights.gguf
```

Check the actual output name of the dataset fetch helper, as it may differ from the example. `--model-file` fingerprints the downloaded model weights but does not load them or verify that the running server uses those weights. `--model-pid` optionally samples RSS via `psutil`. Omit it if you do not have model-server process access. `--modes baseline,semantic` runs a smaller paired comparison.

## What is recorded

Each output record contains task ID, repeat index, requested mode, separate correct/invalid status, independent gold answer, model calls, prompt/completion token counts and request wall-clock latency. It also records template support and whether the stronger restricted semantic check passed.

The `.summary.json` contains operating system, CPU name, logical CPU count, physical RAM where detectable, dataset SHA-256, Python source-tree fingerprint, model identity supplied by operator, optionally model-weight SHA-256, GPU-layer launch declaration, computation budgets, mean/p95 latency, accuracy, accuracy difference and exploratory paired bootstrap intervals.

## Rules for interpretation

1. Do not call these results an AI breakthrough unless they come from *actual model inference*, not mock models or hardcoded template answers.
2. Keep **unsupported template questions in the denominator**. Unsupported questions are abstentions, not successes.
3. The semantic mode uses a limited grammar. Report general coverage separately from conditional correctness.
4. Different inference budgets mean results **are not compute-matched**. Compare latency and calls alongside accuracy.
5. Repeated runs of the same task are not independent data points. Do not confuse larger repeat counts with dataset diversity.
6. `--gpu-layers 0` is a declared configuration, not hardware attestation. Verify the llama.cpp startup log and monitor the actual host yourself.
7. The API and model server run in separate processes. Request wall-clock timings are not CPU-seconds or energy usage. Optional RSS is an approximate observed peak for a *user-selected* process and descendants only.
8. Use a stable, held-out evaluation set and consider contamination, statistical uncertainty and distribution shift.

This harness runs only when the API and model are available. **No real-model inference benchmark is supplied or claimed as part of the release.**
