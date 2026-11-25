# CPU-only llama.cpp server on Windows, configured from detected hardware.
# Prerequisite: winget install llama.cpp   (or set LLAMA_SERVER to a llama-server.exe)
# Usage (from the project root):
#   powershell -NoProfile -File .\scripts\serve_windows_cpu.ps1                       # qwen3-0.6b on 8080
#   powershell -NoProfile -File .\scripts\serve_windows_cpu.ps1 --model qwen3-4b --port 8081
#   powershell -NoProfile -File .\scripts\serve_windows_cpu.ps1 --list-models
# Threads, context size and GPU hiding are chosen by cpu_platform.launch; see docs/hardware.md.
$ErrorActionPreference = 'Stop'
py -m cpu_platform.launch @args
exit $LASTEXITCODE
