# CPU Reasoning Platform v0.5.2

**CPU-only inference research platform with a durable, single-node job service.**

Version 0.5 retains all prior v0.1 through v0.4 functionality, including synchronous reasoning, exact arithmetic, adaptive model calls, multi-step arithmetic plans, restricted-template semantic checks, and five-way model evaluation. It adds an **opt-in SQLite job queue**, single-owner worker scheduling, client-scoped keys, idempotent submissions, per-key and global admission limits, terminal job status, job cancellation before execution, queue timings, SDK polling, a browser job console, and a workload experiment.

**Research status:** This is a functional *local systems preview*. A verified reasoning-quality improvement, frontier-level intelligence, CPU-only model pretraining, internet-facing security, and multi-node scaling **have not been demonstrated**.

## New in v0.5.2: hardware-aware, any machine

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
