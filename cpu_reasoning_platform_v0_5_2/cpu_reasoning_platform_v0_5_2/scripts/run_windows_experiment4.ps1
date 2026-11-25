# Run from project root with API on 8070 and llama.cpp on 8080.
# Reproducible declaration. Confirm that the model server actually uses -ngl 0.
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Write-Host 'Running five-mode CPU-only research smoke test.'
Write-Host 'WARNING: five smoke questions do not establish research-level reasoning performance.'
py -m cpu_platform.experiment4 `
  --dataset data/smoke.jsonl --limit 5 `
  --output results/experiment4_smoke.jsonl `
  --model-id 'Qwen3-0.6B-GGUF' --quantization 'Q8_0' --launch-manifest results/launch_8080.json `
  --gpu-layers 0 `
  --seed 404 --repeats 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host 'Results: results/experiment4_smoke.summary.json'
