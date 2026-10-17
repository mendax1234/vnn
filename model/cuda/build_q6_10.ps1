param([string]$Architecture = 'native', [string]$VisualStudioPath = '')
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'build.ps1') -Precision Q6_10 -Architecture $Architecture -VisualStudioPath $VisualStudioPath
