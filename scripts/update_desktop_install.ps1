param(
  [string]$InstallDir = "$HOME\AntennaMAP",
  [string]$Branch = "main",
  [string]$RepoUrl = ""
)

$ErrorActionPreference = 'Stop'

function Get-RepoUrl {
  param([string]$Provided)
  if ($Provided -and $Provided.Trim().Length -gt 0) { return $Provided }
  throw "RepoUrl is required when InstallDir does not contain a git repo. Use -RepoUrl https://github.com/<you>/AntennaMAP.git"
}

if (!(Test-Path $InstallDir)) {
  $repo = Get-RepoUrl -Provided $RepoUrl
  Write-Host "[AntennaMAP] Install directory missing. Cloning to: $InstallDir"
  git clone --branch $Branch $repo $InstallDir
}

$gitDir = Join-Path $InstallDir '.git'
if (!(Test-Path $gitDir)) {
  $repo = Get-RepoUrl -Provided $RepoUrl
  Write-Host "[AntennaMAP] Directory exists but is not a git repo. Re-cloning..."
  Remove-Item -Recurse -Force $InstallDir
  git clone --branch $Branch $repo $InstallDir
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
