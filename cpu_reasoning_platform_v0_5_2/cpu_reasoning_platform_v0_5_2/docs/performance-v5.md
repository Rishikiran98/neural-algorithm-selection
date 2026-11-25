# v0.5 workload experiment and scientific limitations

## Separate three kinds of measurement

1. **Job-service overhead:** `python -m cpu_platform.experiment5 --workload arithmetic --mode auto --n 16 --concurrency 4`. Exercises exact arithmetic without model calls, so it measures queue, storage and HTTP overhead only.
2. **CPU model-backed reasoning:** start the llama.cpp local CPU server with GPU layers disabled, then run `--workload wordproblems --mode semantic --n 6 --concurrency 1`. This is a repetitive template smoke test, NOT a reasoning-generalization result.
3. **Reasoning-quality comparison:** use `cpu_platform.experiment4`, held-out evaluation data, matched inference budgets, and independently checked CPU environment. See `docs/benchmark4.md`.

Experiment 5 reports job status, expected/observed answers, queue wait, execution latency, and wall-clock duration, with p50/p95 values and aggregate success/correctness. Wall time includes client polling overhead, so it is not a pure server latency metric. Attempted jobs per second (tasks divided by whole experiment elapsed time) should not be interpreted as peak CPU model throughput.

## What we still cannot claim

- No real-model benchmark was run during v0.5 offline release validation.
- No GPU-free pretraining pipeline is established.
- A correct arithmetic trace does not establish that the question was correctly interpreted.
- A repeated template test does not establish general-purpose intelligence.
- A queue throughput improvement does not establish energy efficiency, lower cost per correct answer, or increased reasoning accuracy.
