# Launch API and local model first. Then execute in project root.
$ErrorActionPreference = 'Stop'
py -m cpu_platform.experiment3 --dataset data/smoke.jsonl --limit 5 --output results/experiment3_smoke.jsonl
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host 'Smoke output is exploratory only. Read results/experiment3_smoke.summary.json.'
