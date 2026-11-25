# Durable job lifecycle, v0.5

`POST /v1/jobs` requires `Content-Type: application/json`, enforces a 16 KiB body limit, accepts only `question`, `mode`, and `max_calls`, and validates values before persistence. `question` follows the engine's size cap. `mode` is one of `auto`, `baseline`, `hybrid`, `adaptive`, `proof`, or `semantic`. `max_calls` is optional and limited by engine settings.

## State machine

```text
                    completed with result
queued -> running -----------------------> succeeded / rejected
   |         |                                 
   |         +-- upstream/internal failure --> failed
   |         +-- process restart -----------> interrupted
   +-- DELETE /v1/jobs/{id} ---------------> cancelled
```

An interrupted job is not silently retried. A new request with a new idempotency key is required to run it again. Queued jobs remain queued across restart and are claimed by the next process. The lockfile at `<jobs-db>.lock` prevents two server processes from operating the same local SQLite queue simultaneously.

## Endpoints

| Method | Endpoint | Result |
| --- | --- | --- |
| POST | `/v1/jobs` | 202 new queue entry; 200 same tenant/key/request replay; 409 idempotency conflict; 429 limits |
| GET | `/v1/jobs/<32-lowercase-hex-id>` | 200 job for owner; 404 for other owner or nonexistent |
| DELETE | `/v1/jobs/<id>` | 200 cancelled if queued; 409 if running; 404 if not owned |
| GET | `/v1/metrics` | Existing process metrics plus aggregate job status counts and limits |
| GET | `/readyz` | Local model readiness plus `async_jobs_enabled` |

A job record contains: `id`, `status`, `created_at`, `updated_at`, optional `started_at`, `completed_at`, `queue_wait_ms`, `execution_ms`, `error`, and `result`. No prompt is returned in the job read API. The result may contain its own arithmetic proof trace and verification scope. There is no claim that all job results are semantically correct.

## Data and idempotency

The request body is stored as plaintext JSON in SQLite. `Idempotency-Key` accepts 1 to 96 ASCII letters, digits, dots, colons, underscores and hyphens. It is scoped to an authenticated tenant, or to `local` in unkeyed loopback development mode. Identical replays return the original job, including terminal jobs while retained. Same key with different request content is HTTP 409. The quota is enforced on distinct admissions. Defaults: max 32 queued+running globally, 8 per tenant, and 120 admissions per tenant per hour.

SQLite transaction claims prevent duplicate queue claims, and an OS lockfile prevents multiple owners from interpreting in-flight jobs as interrupted. This remains a **single-node local queue**. Disk encryption, backups, crash simulation, access review, and tenant privacy assurance are not yet production-grade.

## Errors

Errors are stable machine-readable codes: `jobs_not_configured` (503), `invalid_request_fields` (400), `invalid_question` (400), `invalid_mode` (400), `invalid_max_calls` (400), `invalid_idempotency_key` (400), `queue_full` (429), `hourly_submission_limit` (429), `job_not_found` (404), `job_already_running` (409), and `idempotency_conflict` (409). Authentication failures return 401. The process-local synchronous `/v1/reason` API remains available when jobs are disabled.
