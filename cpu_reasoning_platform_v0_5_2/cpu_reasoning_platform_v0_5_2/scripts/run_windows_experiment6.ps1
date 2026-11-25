# Experiment 006: small model + verified search vs a larger model at matched CPU compute.
# First, in two other PowerShell windows (from the project root):
#   powershell -NoProfile -File .\scripts\serve_windows_cpu.ps1 --model qwen3-0.6b --port 8080
#   powershell -NoProfile -File .\scripts\serve_windows_cpu.ps1 --model qwen3-4b --port 8081
# Optional for CPU-seconds instead of wall-clock: py -m pip install psutil
# Extra arguments are passed through, e.g. --dataset data/gsm8k_test.jsonl --limit 100
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
py -m cpu_platform.experiment6 --output results/experiment6.jsonl @args
exit $LASTEXITCODE
