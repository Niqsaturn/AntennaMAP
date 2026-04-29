$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Venv = Join-Path $Root '.venv'
$Python = 'python'

Write-Host "[AntennaMAP] Project root: $Root"

if (!(Test-Path $Venv)) {
  Write-Host "[AntennaMAP] Creating virtual environment..."
  & $Python -m venv $Venv
}

$VenvPython = Join-Path $Venv 'Scripts\python.exe'

Write-Host "[AntennaMAP] Installing dependencies..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

Write-Host "[AntennaMAP] Running tests..."
$env:PYTHONPATH='.'
& $VenvPython -m pytest -q

Write-Host "[AntennaMAP] Starting app at http://127.0.0.1:8000"
& $VenvPython -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
