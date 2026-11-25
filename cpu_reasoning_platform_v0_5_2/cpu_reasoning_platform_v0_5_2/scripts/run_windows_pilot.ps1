# v0.8 pilot: 100 real GSM8K questions, small model reasoning + thinking + checking vs a large model.
# Two model servers must already be running, each in its OWN window that you leave alone:
#   py -m cpu_platform.launch --model qwen3-0.6b --port 8080
#   py -m cpu_platform.launch --model qwen3-8b  --port 8081
# One-time: py -m pip install datasets psutil ; py scripts\fetch_gsm8k.py --limit 200
# Extra arguments pass through, e.g. --limit 20 for a quicker first look.
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
py -m cpu_platform.experiment6 --dataset data/gsm8k_test.jsonl --limit 100 --output results/pilot_v08.jsonl @args
exit $LASTEXITCODE
