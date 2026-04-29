@echo off
setlocal

set ROOT=%~dp0\..
cd /d %ROOT%

echo [AntennaMAP] Project root: %CD%

if not exist .venv (
  echo [AntennaMAP] Creating virtual environment...
  python -m venv .venv
)

echo [AntennaMAP] Delegating to PowerShell launcher...
powershell -ExecutionPolicy Bypass -File scripts\install_and_run.ps1

endlocal
