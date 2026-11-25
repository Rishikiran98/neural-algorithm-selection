#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
chmod 700 results
printf '%s\n' 'Local API: http://127.0.0.1:8070' 'Durable jobs enabled: prompts/results stored on local disk in plaintext.'
exec python3 -m cpu_platform.server --port 8070 --model-url http://127.0.0.1:8080 --jobs-db results/jobs.sqlite3 --job-workers 1 --max-queued 32 --max-per-tenant 8 --job-ttl-hours 24 --max-inflight 1
