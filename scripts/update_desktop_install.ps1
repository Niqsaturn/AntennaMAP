param(
  [string]$InstallDir = "$HOME\Desktop\AntennaMAP",
  [string]$Branch = "main"
)

$ErrorActionPreference = 'Stop'

if (!(Test-Path $InstallDir)) {
  throw "Install directory not found: $InstallDir"
}
if (!(Test-Path (Join-Path $InstallDir '.git'))) {
  throw "Directory exists but is not a git repo: $InstallDir"
}

Set-Location $InstallDir

Write-Host "[AntennaMAP] Updating repository in: $InstallDir"
git fetch origin $Branch
git checkout $Branch
git pull --ff-only origin $Branch

Write-Host "[AntennaMAP] Refreshing environment + tests"
& .\scripts\setup_env.ps1

Write-Host "[AntennaMAP] Launching server"
& .\scripts\run_server.ps1
