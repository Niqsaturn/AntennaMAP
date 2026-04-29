param(
  [Parameter(Mandatory=$true)]
  [string]$RepoUrl,
  [string]$TargetDir = "AntennaMAP"
)

$ErrorActionPreference = 'Stop'

if (Test-Path $TargetDir) {
  throw "Target directory '$TargetDir' already exists."
}

Write-Host "[AntennaMAP] Cloning repository..."
git clone $RepoUrl $TargetDir

Set-Location $TargetDir
Write-Host "[AntennaMAP] Running local installer/runner..."
& .\scripts\install_and_run.ps1
