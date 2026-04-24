$ErrorActionPreference = 'Stop'

$pythonCandidates = @(
  '.\.venv311\Scripts\python.exe',
  'python'
)

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
  try {
    & $candidate -c "print('ok')" *> $null
    if ($LASTEXITCODE -eq 0) {
      $pythonExe = $candidate
      break
    }
  } catch {}
}

if (-not $pythonExe) {
  throw 'Python not found. Please install Python or activate your project environment first.'
}

$jobs = @(
  @{ dataset = 'nyc'; weights = 'weights/myplan_nyc_best.h5'; output = 'weights/frontend_predictions_nyc.json' },
  @{ dataset = 'chicago'; weights = 'weights/myplan_chicago_best.h5'; output = 'weights/frontend_predictions_chicago.json' }
)

foreach ($job in $jobs) {
  if (-not (Test-Path $job.weights)) {
    Write-Warning "Skip $($job.dataset): missing $($job.weights)"
    continue
  }

  & $pythonExe infer_and_export_frontend.py --dataset $job.dataset --weights $job.weights --json_out $job.output
  if ($LASTEXITCODE -ne 0) {
    throw "Export failed for $($job.dataset)"
  }
}

Write-Host 'Done exporting frontend JSON files from weights.'
