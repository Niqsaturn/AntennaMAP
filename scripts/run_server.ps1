$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
if (!(Test-Path $VenvPython)) {
  throw "Missing .venv. Run scripts/setup_env.ps1 first."
}

Write-Host "[AntennaMAP] Starting app at http://127.0.0.1:8000"
& $VenvPython -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
