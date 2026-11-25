# CPU Reasoning Platform v0.10.0

## New: general local reasoning preview

The new `/v1/ask` endpoint supports explanation, coding, logic, science, math, and factual Q&A through the existing CPU-hosted LLM. The model-generated answers are **not independently verified**. Formal propositional entailment has a separate exact `/v1/verify-logic` endpoint. The original v0.9.1 arithmetic, benchmarks, frozen-dataset safeguards, API, durable jobs, and tests are retained.

Detailed instructions: [v0.10 guide](docs/general-v010.md).

# CPU Reasoning Platform v0.9.1

## v0.9.1: frozen-evaluation leakage guard fix

- Copied evaluation questions are now identified by SHA256 fingerprints of **normalized question text**, even when their row IDs have been changed, their casing/whitespace changes, or they are embedded in a mixed dataset. An exact match to **one** frozen evaluation question triggers evaluation handling.
- The benchmark refuses a copied/merged evaluation dataset, even with `--confirm-frozen-eval`; only the original frozen evaluation file is allowed with explicit confirmation.
- Each *new* split manifest records per-question fingerprints, and `validate` checks them. **Existing v0.9 split manifests remain supported**: fingerprints are computed from their original, SHA256-pinned split files. **Do not run `freeze --force`** to apply this fix. Keep the original frozen files and `eval_runs.jsonl`.
- A missing/altered frozen reference file now causes a safety error **before model calls**, rather than silently allowing a dataset to proceed. An accidental collision of row IDs alone no longer marks a different question as evaluation.
- The guard detects copied questions after case/whitespace/punctuation normalization, but **not paraphrases** or other semantic near-duplicates. It is an accidental-contamination safeguard, not a guarantee against data leakage or model pretraining contamination.

To verify in Windows PowerShell:

```powershell
py -m unittest discover -s tests -v
py -m cpu_platform.datasets validate
```

The second command requires an existing `data/splits/manifest.json`. For the first freeze only, use the original v0.9 instructions below.

## New in v0.9: evaluation you can defend

- **Frozen splits** (`py -m cpu_platform.datasets freeze`): your pilot questions become the *dev* set; 500 other GSM8K questions, de-duplicated and checked against the prompts, become a frozen *eval* set. Experiment 6 refuses eval runs without `--confirm-frozen-eval` and logs every one.
- **Provenance**: the SHA256 of the model file each server actually loaded, the llama.cpp build, peak memory, and an evidence table where anything unmeasured says NOT RUN.
- **Fairer cost**: comparisons are also matched per question, not only on average.
- **Checker v2**: flags numbers that come from nowhere, adds a `grounded` policy, and every checked sample says what was *not* checked. Adversarial fixtures keep it honest.
- **Reports**: every run writes `*.report.md` and `*.failures.jsonl` (wrong answers labelled by likely cause, with traces). `py -m cpu_platform.report <run>.jsonl` regenerates one for any earlier run.

See [docs/evaluation-v09.md](docs/evaluation-v09.md).

## New in v0.8: native thinking with a budget, plus a better checker

The first real runs showed the 0.6B model reasons far better when allowed to write its reasoning (5/5 vs 1/5 for grammar-constrained plans on the smoke set). v0.8 goes further:

- **Native thinking is back**, controlled per request: llama.cpp caps thinking at `--think-budget` tokens and then forces an answer. Arms: `small_think`, `small_think_vote`, `small_think_checked` (optional `large_think`).
- **The arithmetic checker reads real model writing**: `$` amounts, thousands commas, LaTeX, `x` for times, percentages, rounding. Three eligibility rules are reported for every sample.
- **No more toy questions by default**: experiment 6 runs on `data/gsm8k_test.jsonl` (100 questions) and refuses to start without it (`py scripts\fetch_gsm8k.py`).
- **Restart your model servers** with the v0.8 launcher (it no longer forces thinking off). A preflight check says so if needed.

See [docs/benchmark8.md](docs/benchmark8.md) for the pilot commands and how to read the results.

**CPU-only inference research platform with a durable, single-node job service.**

Version 0.5 retains all prior v0.1 through v0.4 functionality, including synchronous reasoning, exact arithmetic, adaptive model calls, multi-step arithmetic plans, restricted-template semantic checks, and five-way model evaluation. It adds an **opt-in SQLite job queue**, single-owner worker scheduling, client-scoped keys, idempotent submissions, per-key and global admission limits, terminal job status, job cancellation before execution, queue timings, SDK polling, a browser job console, and a workload experiment.

**Research status:** This is a functional *local systems preview*. A verified reasoning-quality improvement, frontier-level intelligence, CPU-only model pretraining, internet-facing security, and multi-node scaling **have not been demonstrated**.

## New in v0.7: let the model reason, then inspect explicit arithmetic

An initial five-question CPU run found that free-form reasoning performed better
than grammar-constrained plans, despite all constrained plans being syntactically
valid. This is preliminary evidence on very few easy problems, **not** a general
model-performance result. Version 0.7 introduces a paired experiment that asks:
**Can cheap symbolic checks improve voting over free-form reasoning chains?**

- `small_cot_vote`: N sampled free-form reasoning chains, voted without checking.
- `small_cot_checked`: evaluate **the same generated chains**, rejecting ones
  containing a detectable arithmetic error or lacking any checkable equation.
- Each extracted arithmetic equality is recomputed exactly, without executing
  model-generated text. Checker CPU overhead, coverage, errors, and abstentions
  are measured. This is **not semantic verification**.
- The existing v0.1 through v0.6 APIs, tests, and evaluation arms remain.

On Windows, with the Qwen3-0.6B llama.cpp server already running on `127.0.0.1:8080`:

```powershell
py -m unittest discover -s tests -v
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_windows_experiment7.ps1 --limit 5 --samples 4
```

Output: `results/experiment7.jsonl` and `results/experiment7.summary.json`.
**Model replies are saved in cleartext in the research JSONL**. Do not use sensitive
prompts. See [docs/benchmark7.md](docs/benchmark7.md) for evaluation limits and
100-question runs.

## New in v0.6: grammar-constrained plans (from the first real run)

The first real run (Qwen3-0.6B, CPU only, 6 threads, ~28 tokens/s on an i7-8850H) produced **0 valid step plans out of 8**: the model invented numbers, broke the format or abstained, at 13-35 CPU-seconds per plan. v0.6 moves that work to cheap symbolic code:

- A **per-question grammar** (llama.cpp GBNF) lets the model emit only well-formed plans whose numbers come from the question and whose references point to earlier steps.
- The model picks **operations and operands only**. Exact arithmetic computes every result, so there are no arithmetic slips and fewer tokens.
- Grounding now covers number words ("three", "a dozen") and unit conversions used only when both units appear (hours + minutes -> 60).
- Experiment 6 adds **reasoning references** (`small_cot`, `large_cot`: think step by step, then answer) and the `small_constrained` arm. A preflight check confirms your llama.cpp actually applies the grammar.

Whether the chosen operations are *right* is still unchecked, and is what experiment 6 measures.

## Since v0.5.2: hardware-aware, any machine

Nothing is hardcoded for one computer any more. The platform detects cores, RAM, GPUs and memory bandwidth and configures llama.cpp from them, the same way on Windows, Linux and macOS:

```bash
python -m cpu_platform.hardware               # what was detected, recommended settings
python -m cpu_platform.launch --list-models   # which models fit this machine, speed estimates
python -m cpu_platform.launch --model qwen3-0.6b              # CPU-only server on 127.0.0.1:8080
python -m cpu_platform.launch --model qwen3-4b --port 8081    # comparison model
```

Every GPU is hidden (`--device none` plus backend environment variables), the server log is scanned for GPU use, and a launch manifest records exactly what ran. **Experiment 6** (`cpu_platform.experiment6`) is the first direct test of the project thesis: small model + verifier-guided search vs a larger model **at matched CPU compute**. See [docs/hardware.md](docs/hardware.md) and [docs/benchmark6.md](docs/benchmark6.md).

## Windows quick start

1. Install Python 3.11+ and unzip the project. Open PowerShell in the extracted folder.
2. Run tests before starting the service:

```powershell
py -m unittest discover -s tests -v
```

3. Start the **CPU-only local service and persistent queue**:

```powershell
powershell -NoProfile -File .\scripts\start_windows_product.ps1
```

4. Open **http://127.0.0.1:8070**. Use either **Execute on CPU** for synchronous arithmetic or **Queue current problem** for durable background arithmetic. Try `calculate: (12+8)/5`.

5. In another PowerShell window, evaluate the local job service without an AI model:

```powershell
py -m examples.async_jobs
py -m cpu_platform.experiment5 --workload arithmetic --mode auto --n 16 --concurrency 4 --out results/experiment5.json
```

The job database is `results/jobs.sqlite3`. **Questions and results are stored as plaintext** in SQLite and may remain for 24 hours after completion by default. Only enable durable jobs for data you are comfortable storing locally; use restrictive file permissions and encrypted storage for sensitive experiments. Deleting the database while the server is running is unsafe. The test suite does not require any model download.

## Optional CPU model for real reasoning experiments

Install [llama.cpp](https://github.com/ggml-org/llama.cpp). In a separate PowerShell window:

```powershell
powershell -NoProfile -File .\scripts\serve_windows_cpu.ps1                 # qwen3-0.6b, settings from detected hardware
powershell -NoProfile -File .\scripts\serve_windows_cpu.ps1 --list-models   # other models that fit
```

The script runs `cpu_platform.launch`: a quantized Qwen model with zero GPU offload, all GPUs hidden, and threads and context size from the detected hardware. Model weights are **not included**, model latency and reasoning accuracy have **not been measured**, and the model may originally have been trained using GPUs. If the commands differ on your llama.cpp release, check its CLI help.

Run earlier CPU-model experiments with:

```powershell
powershell -NoProfile -File .\scripts\run_windows_experiment4.ps1
```

Or run a model-backed asynchronous workload, noting that the repeated templates are **not** a generalization benchmark:

```powershell
py -m cpu_platform.experiment5 --workload wordproblems --mode semantic --n 6 --concurrency 1 --out results/experiment5_model.json
```

## Linux and local server

```bash
python3 -m unittest discover -s tests -v
bash scripts/start_linux_product.sh
# Separate shell
python3 -m cpu_platform.experiment5 --n 16 --concurrency 4
```

For an existing v0.4 installation, **extract v0.5 into a new folder**, not over the old files. Copy only your experimental outputs as needed. The original v0.4 API remains available without a job database:

```bash
python3 -m cpu_platform.server --port 8070
```

## API v0.5 (durable jobs are opt-in)

With `--jobs-db results/jobs.sqlite3` enabled:

```http
POST /v1/jobs
Content-Type: application/json
Idempotency-Key: sample-001

{"question":"calculate: (12+8)/5","mode":"auto","max_calls":2}
```

Returns HTTP **202** with `id`, `status=queued`, timestamps, or **200** if the same tenant resubmits the identical idempotency key and request. Conflict returns HTTP 409. `GET /v1/jobs/<id>` returns only the owning key's job and its verified or unverified result. `DELETE /v1/jobs/<id>` cancels only a **queued** job; running jobs cannot be forcibly cancelled.

If there is no model server, exact arithmetic works without neural calls and word-problem model calls may fail with `local_model_unavailable`. A completed job can be `succeeded`, `rejected`, `failed`, `cancelled`, or `interrupted`; job success is not the same as a semantically verified answer.

The original routes are retained: `POST /v1/reason`, `POST /v1/verify-plan`, `POST /v1/verify-semantic-plan`, `GET /healthz`, `GET /readyz` and `GET /v1/metrics`.

### Python client

```python
from cpu_platform.client import CPUReasoningClient

api = CPUReasoningClient()
submitted = api.submit_job("calculate: 12*(7+3)", idempotency_key="first-001")
finished = api.wait_for_job(submitted["id"], timeout_seconds=60)
print(finished["status"], finished.get("result", {}).get("answer"))
```

### Isolation and limits

For single-user local use, leave auth unset. For multiple development clients, set **one** of the following in the environment before launching the server:

```powershell
$env:CPU_REASONING_API_KEY = "choose-a-long-random-key"
```

Or use a JSON mapping of tenant names to distinct long keys, such as `CPU_REASONING_API_KEYS_JSON` with the value `{"alice":"random-secret-one","bob":"random-secret-two"}`. This is local *development* key separation, not customer identity management. Don't commit credentials into the repository.

Configurable CLI settings: `--jobs-db`, `--job-workers` (default 1), `--max-queued` (32), `--max-per-tenant` (8), `--max-submissions-per-hour` (120), and `--job-ttl-hours` (24). Admission limits count queued and running jobs, and return 429 when reached. Idempotent retries do not consume additional slots. Worker interruptions after a restart are reported as `interrupted` and **not automatically retried**. Queued jobs are resumed after restart. The SQLite queue allows one owning server per database file. Since v0.5.1, queued jobs and synchronous `/v1/reason` calls share the single `--max-inflight` execution budget (default 2), so their combined concurrency never exceeds it.

**No direct internet exposure.** The HTTP server deliberately binds only to `127.0.0.1`. It lacks TLS, customer-grade authentication, durable authorization administration, distributed scheduling, audited infrastructure controls, secure secret management, and sandbox isolation for arbitrary plugins. Access a remote CPU server using SSH port forwarding, not an open public port.

## Documentation

- [v0.5 API and lifecycle](docs/api-v5.md)
- [Local threat model and persistence](docs/product-readiness.md)
- [Research and performance measurement limitations](docs/performance-v5.md)
- [v0.4 instructions and benchmark details](docs/legacy-v0.4.md)
- [v0.5 release notes](RELEASE_NOTES.md)

This software is MIT licensed. Model and benchmark dataset terms must be reviewed separately.
