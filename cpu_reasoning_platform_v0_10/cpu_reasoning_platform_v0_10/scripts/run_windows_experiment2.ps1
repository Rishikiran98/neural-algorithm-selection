# Local 3-arm comparison using CPU-only llama.cpp and the running platform API.
# Start local llama server and platform server first in separate PowerShell windows.
$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
py -m cpu_platform.experiment2 --dataset data/smoke.jsonl --limit 5 --output results/experiment2_smoke.jsonl
if ($LASTEXITCODE -ne 0) {
  throw 'Benchmark failed. Confirm that the API is running and llama.cpp has loaded the model.'
}
Write-Host 'Research smoke test complete. Not enough data for model-quality conclusions.'
