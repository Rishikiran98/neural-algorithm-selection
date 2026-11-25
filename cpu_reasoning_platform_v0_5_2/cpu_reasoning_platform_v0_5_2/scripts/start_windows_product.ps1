# CPU Reasoning Platform v0.5, local persistent job API.
# Optional: set CPU_REASONING_API_KEY in this PowerShell session before running.
$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
New-Item -ItemType Directory -Force -Path 'results' | Out-Null
Write-Host 'Local API: http://127.0.0.1:8070'
Write-Host 'Durable jobs enabled: prompts/results will be stored in results/jobs.sqlite3 in plaintext.'
Write-Host 'To use model inference, start scripts/serve_windows_cpu.ps1 in another PowerShell window.'
py -m cpu_platform.server --port 8070 --model-url http://127.0.0.1:8080 --jobs-db results/jobs.sqlite3 --job-workers 1 --max-queued 32 --max-per-tenant 8 --job-ttl-hours 24 --max-inflight 1
