param(
  [string]$Dataset = 'nyc',
  [string]$Output = ''
)

if (-not $Output) {
  $Output = "weights/frontend_metrics_$Dataset.json"
}

$payload = @{
  dataset = $Dataset
  models = @{
    Myplan = @{ aucpr = 0.0; aucroc = 0.0; f1 = 0.0; acc = 0.0; recall = 0.0 }
    SNIPER = @{ aucpr = 0.0; aucroc = 0.0; f1 = 0.0; acc = 0.0; recall = 0.0 }
    GSNet = @{ aucpr = 0.0; aucroc = 0.0; f1 = 0.0; acc = 0.0; recall = 0.0 }
  }
  ablation = @{
    noSmoothNoStream = @{ aucpr = 0.0; aucroc = 0.0; f1 = 0.0; acc = 0.0; recall = 0.0 }
    noSmooth = @{ aucpr = 0.0; aucroc = 0.0; f1 = 0.0; acc = 0.0; recall = 0.0 }
    noStream = @{ aucpr = 0.0; aucroc = 0.0; f1 = 0.0; acc = 0.0; recall = 0.0 }
    full = @{ aucpr = 0.0; aucroc = 0.0; f1 = 0.0; acc = 0.0; recall = 0.0 }
  }
}

$dir = Split-Path -Parent $Output
if ($dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
$payload | ConvertTo-Json -Depth 6 | Set-Content -Path $Output -Encoding UTF8
Write-Host "Saved metrics template to $Output"
