# CPU Reasoning Platform architecture, v0.3

- HTTP entry point: `cpu_platform/server.py`. Loopback-only standard-library preview server with optional local bearer authentication, request size caps, concurrency semaphore, health and process-local metrics.
- Model adapter: `cpu_first/core.py` with local llama.cpp OpenAI-style chat API; `LazyLocalModel` uses loopback only. CPU flags are supplied to the separately launched llama.cpp process.
- Router: `cpu_platform/engine.py` supports `auto`, `baseline`, `hybrid`, `adaptive`, and the new `proof` mode. The new mode emits a structured plan and optionally repairs rejected plans under a hard two-model-call limit.
- Typed proof checker: `cpu_platform/proof.py` performs exact arithmetic replay of a bounded sequential DAG and rejects invalid step claims, references, excessive plan size, ungrounded numeric literals, or unauthorized expression structures. It never executes generated code.
- Lexical provenance checker: `cpu_platform/grounding.py`. Its numeric membership test does not validate word-problem semantics.
- Browser playground: `cpu_platform/static/index.html`. Includes local proof inspection with zero neural calls.
- Benchmark: `cpu_platform/experiment3.py`. Four-mode paired experiment comparing correctness against external numeric ground truth, with aggregate proof validity and use accounting.

## Trust boundary

The model's JSON is untrusted. The deterministic verifier accepts only a restricted schema, bounded arithmetic values and known step references, and independently recomputes claimed answers. The result is proof of arithmetic consistency, not proof that the question has been understood. Do not publish the loopback preview as an internet-facing multi-tenant service. There is no durable quota, TLS gateway, isolation, persistent identity or observability stack.

## v0.5 cumulative architecture extension

`cpu_platform/jobs.py` adds an optional local SQLite WAL queue with an exclusive OS lockfile for one owning server per database path. API admission validates requests, applies per-tenant key isolation, idempotency, global/per-tenant active limits and a rolling admission limit. The queue stores accepted prompts as local plaintext, and a small worker pool calls the existing `ReasoningEngine.reason` with bounded model calls. Results retain the v0.4 verification scope, including the difference between arithmetic/proof checks and restricted-template semantic checks.

The HTTP `App` continues to bind to loopback only. Synchronous v0.4 routes remain available even when jobs are disabled; asynchronous routes return `jobs_not_configured` without an opt-in database. The new job console in the browser uses the same key field. No new model training, multi-node orchestration, or generalized natural-language reasoner was introduced.
