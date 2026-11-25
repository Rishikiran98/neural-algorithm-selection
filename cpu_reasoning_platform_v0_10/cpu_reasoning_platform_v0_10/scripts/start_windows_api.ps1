# Start in a second Windows PowerShell window while llama.cpp is running.
$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
py -m cpu_platform.server --port 8070 --model-url http://127.0.0.1:8080 --max-inflight 1
