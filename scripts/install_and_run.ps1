$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

& .\scripts\setup_env.ps1
& .\scripts\run_server.ps1
