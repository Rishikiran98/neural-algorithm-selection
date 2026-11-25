# Does not require any model downloaded. It tests local exact arithmetic only.
$body = @{ question='calculate: 12*(7+3)'; mode='auto' } | ConvertTo-Json
$response = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8070/v1/reason -Body $body -ContentType 'application/json'
$response | ConvertTo-Json -Depth 8
if ($response.answer -ne '120' -or -not $response.arithmetic_verified -or $response.usage.model_calls -ne 0) {
    throw 'CPU-only math smoke test failed'
}
Write-Host 'PASS: exact arithmetic, no neural inference calls'
