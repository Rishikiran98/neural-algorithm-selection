# v0.5 product-readiness assessment

## Implemented and regression-tested

- Local HTTP routes, bearer key comparison, key-scoped job reads/cancellation, bounded request bodies, admission limits.
- Opt-in persistent SQLite queue with WAL, idempotency, atomic transaction-based claims and an exclusive single-owner OS lock.
- Worker pool bounded to 1 to 16 workers, queue execution timing, terminal statuses, queued cancellation, restart recovery semantics.
- Existing proof and restricted semantic verification, smoke benchmarks and local browser playground.

## Explicitly not implemented

- Internet-facing TLS/identity, SSO, customer account lifecycle, per-user RBAC, enterprise audit logging, secrets rotation.
- Distributed broker, multi-host worker scheduler, model placement, GPU/CPU energy accounting, scale-out automatic failover.
- Arbitrary tool execution sandbox, customer isolation guarantees, PII retention policy or SOC 2 controls.
- Training or running a frontier-level model without GPUs. The model runner remains an independently started local CPU inference process.

## Storage and recovery

Enabling `--jobs-db` stores questions, result traces and answers as plaintext for an adjustable TTL (1 to 720 hours, default 24). The TTL is enforced lazily during admission and startup, so disk erasure may occur later than the time boundary. SQLite deletes do not guarantee physical secure erasure. Secure disk encryption and deliberate database deletion are external operational requirements. If the process dies while executing a job, the next startup marks the job `interrupted`. A previously queued job resumes. Cancellation stops *only* queued work; it cannot terminate a currently running inference call.

## Deployment guidance

Bind only to loopback. Keep the model adapter and HTTP API on the same host or in a private SSH-forwarded configuration. Do not publish the Python `ThreadingHTTPServer` to the public internet, even behind an ad hoc port-forward. Treat `CPU_REASONING_API_KEY` and `CPU_REASONING_API_KEYS_JSON` as local development credentials, not production customer authentication. Monitor disk capacity and the queue. Model inference can be substantially slower than queue processing and can make 429 limits necessary.
