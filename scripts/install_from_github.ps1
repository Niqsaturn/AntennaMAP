param(
  [Parameter(Mandatory=$true)]
  [string]$RepoUrl,
  [string]$TargetDir = "AntennaMAP",
  [string]$Branch = "main"
)

$ErrorActionPreference = 'Stop'

if (Test-Path "$TargetDir\.git") {
  Write-Host "[AntennaMAP] Existing clone found. Updating..."
  git -C $TargetDir fetch origin $Branch
  git -C $TargetDir checkout $Branch
  git -C $TargetDir pull --ff-only origin $Branch
} else {
  if (Test-Path $TargetDir) { throw "'$TargetDir' exists but is not a git clone." }
  Write-Host "[AntennaMAP] Cloning repository..."
  git clone --branch $Branch $RepoUrl $TargetDir
}

Set-Location $TargetDir
& .\scripts\setup_env.ps1
& .\scripts\run_server.ps1
