# Release notes: v0.5.2

**Focus:** hardware-agnostic CPU-only setup, and the first experiment that directly tests the project thesis.

## Added

- `cpu_platform/hardware.py`: portable detection (Windows, Linux, macOS; standard library only) of physical cores, RAM, GPUs and single-thread memory bandwidth, plus a recommended profile: threads = physical cores, context size and feasible models by RAM, and a rough tokens/s estimate per model. `python -m cpu_platform.hardware`.
- `cpu_platform/launch.py`: one cross-platform llama.cpp launcher that replaces the hardcoded per-OS settings. It picks model presets (`qwen3-0.6b` to `qwen3-14b`, or `--hf`/`--model-file`), refuses models that don't fit in RAM, and hides every GPU (`--device none` when the build supports it, plus CUDA/ROCm/SYCL/Vulkan environment variables). It binds to loopback, saves the server log, scans it for GPU use, and writes `results/launch_<port>.json` (argv, version, PID, settings, hardware, GPU verdict). Optional flags are added only when that build's `--help` lists them (flag names checked against current llama.cpp source).
- `cpu_platform/search.py`: verifier-guided test-time search. Draws N seeded samples, checks plan candidates with exact replay, majority-votes (ties go to the first seen), and scores every prefix. Includes a psutil CPU-seconds meter for the model server process.
- `cpu_platform/experiment6.py`: paired arms `small_greedy`, `large_greedy`, `small_vote` (sampling without the checker) and `small_verified`. Produces accuracy-vs-cost curves, a matched-compute comparison against the larger model, and the verifier's effect at an equal number of samples. Cost is server CPU-seconds when metered, otherwise wall-clock, never mixed.
- `docs/hardware.md`, `docs/benchmark6.md`, `scripts/run_*_experiment6`.
- Tests: `tests/test_v052.py` (20 tests) and `tests/fake_llama.py`, an offline stand-in for llama.cpp's HTTP API that runs the real client code end to end. Full suite: 141 tests. Also fixes a race in the v0.4 `test_metrics_no_prompt_content` test that failed intermittently under load.

## Changed

- `scripts/serve_windows_cpu.ps1` and `scripts/serve_cpu.sh` now delegate to the launcher (previously `-t 4 -c 2048` on Windows and no thread setting on Linux).
- Product and API start scripts use `--max-inflight 1` (one llama.cpp process should serve one request at a time).
- Experiment 4 scripts read threads and context size from the launch manifest (`--launch-manifest`) instead of hardcoding `--threads 4 --context-size 2048`. The manifest is embedded in the summary.
- **Qwen3 thinking is disabled for every mode**: `--reasoning off` (or `--reasoning-budget 0`) at launch, `/no_think` in the system prompt for Qwen3 model ids, and `<think>` blocks stripped from replies. Without this, Qwen3 can spend the whole token budget thinking and produce no JSON, which would make every mode look broken.
- `LocalLlama.chat` accepts `temperature` (0–2) and `seed`, and only sends them when set.

## Still true

No real model has been run as part of this release (the build environment can't download weights). The first real numbers come from running experiment 6 on your own machine.

# Release notes: v0.5.1 (bug-fix patch)

Three fixes, each with regression tests in `tests/test_v051_fixes.py` (9 tests, all of which fail on v0.5.0). Full suite: 121 tests.

1. **Sentence-final numbers are now grounded** (`cpu_platform/grounding.py`). The number regex rejected any number followed by a period, so "pay with $50." never matched 50. This made the `purchase_change` semantic template impossible to pass and caused `adaptive`, `proof` and `semantic` modes to reject correct answers whenever a key number ended a sentence. Version strings and ordinals ("v1.2.3", "5th") are still rejected. A new test runs every supported smoke question through full `verify_semantic_plan`, not just `classify`.
2. **Job workers survive storage errors** (`cpu_platform/jobs.py`). Previously an exception while claiming or saving a job killed the worker thread and left the job stuck in `running` (uncancellable, counting against quota) until restart. Now claims back off and retry; result writes retry three times, then fall back to a bare `failed` / `result_storage_failed` record so the job leaves the active set. If storage is completely unavailable, the job stays `running` until restart recovery marks it `interrupted`, but the worker keeps running.
3. **One shared execution budget** (`cpu_platform/server.py`, `jobs.py`). Synchronous requests and queued jobs now share the `--max-inflight` semaphore, so together they never exceed it. Workers take a slot *before* claiming a job, so a job waiting for capacity stays `queued`, not `running`. With `--job-workers` greater than `--max-inflight`, the extra workers simply wait. While jobs hold every slot, synchronous requests get the existing 429 `capacity_exceeded`.

Recommended setup: `--max-inflight` equal to the number of parallel requests your llama.cpp server is configured for (often 1 on an 8 GB laptop).

# Release notes: v0.5.0 (cumulative)

**Focus:** transition from a one-shot local research prototype to an opt-in durable, CPU-native single-node job service.

## New in v0.5

- New `cpu_platform/jobs.py`: SQLite WAL queue with idempotent admission, atomic claims, single-owner lock, pending/running/terminal lifecycle, bounded worker pool, cancellation before execution, restart recovery, 24-hour default lazy retention and configurable quotas.
- HTTP `POST /v1/jobs`, `GET /v1/jobs/{id}`, `DELETE /v1/jobs/{id}` plus queue status counts in metrics. Optional tenant-specific keys with no cross-tenant job visibility.
- SDK `submit_job`, `get_job`, `cancel_job`, `wait_for_job`.
- Browser job console and one-command Windows/Linux startup scripts.
- `cpu_platform.experiment5` for reproducible local workload stress tests and queue timing.
- Offline tests for concurrency and backpressure, isolation, idempotency, cancellation, interrupted-job recovery, API validation and workloads.

## Preserved

- v0.1: Python SDK, local HTTP API, exact arithmetic, local llama.cpp model adapter, browser UI.
- v0.2: adaptive inference and bounded repair, comparison harness.
- v0.3: exact multi-step proof graphs and four-way evaluation.
- v0.4: restricted semantic-template checks, five-mode reproducible evaluation.

## Validation

- The offline tests exercise a fake model and exact arithmetic, not a real model.
- No inference accuracy, energy/cost advantage or frontier reasoning quality has been demonstrated.
- The API remains deliberately loopback-only. SQLite jobs store input/output content on disk, are not encrypted by this application, and are not intended for production internet exposure.
