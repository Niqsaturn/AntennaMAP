$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Venv = Join-Path $Root '.venv'
if (!(Test-Path $Venv)) {
  Write-Host "[AntennaMAP] Creating virtual environment..."
  python -m venv $Venv
}

$VenvPython = Join-Path $Venv 'Scripts\python.exe'
Write-Host "[AntennaMAP] Installing dependencies..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

Write-Host "[AntennaMAP] Running tests..."
$env:PYTHONPATH='.'
& $VenvPython -m pytest -q

Write-Host "[AntennaMAP] Import smoke check"
& $VenvPython -c "import backend.main as m; print('BaseModel:', bool(getattr(m, 'BaseModel', None)))"
