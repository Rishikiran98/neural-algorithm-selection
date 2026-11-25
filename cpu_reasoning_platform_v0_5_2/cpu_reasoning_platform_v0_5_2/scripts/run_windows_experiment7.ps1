# Paired free-form reasoning experiment, small CPU model only (port 8080).
# Run from project root. The project inherits the llama.cpp CPU-only launcher.
# Explicitly use the same sampled reasoning chains for both arms.
# Example: powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_windows_experiment7.ps1 --limit 5 --samples 4
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
py -m cpu_platform.experiment6 --large-url none --arms small_cot_vote,small_cot_checked --output results/experiment7.jsonl @args
exit $LASTEXITCODE
