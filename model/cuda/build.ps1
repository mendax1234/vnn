param(
    [string]$Architecture = 'native',
    [string]$WeightsHeader = '',
    [string]$VisualStudioPath = '',
    [ValidateSet('FP16', 'Q6_10')][string]$Precision = 'FP16'
)
$ErrorActionPreference = 'Stop'
if ($Architecture -notmatch '^(native|sm_[0-9]+)$') { throw 'Architecture must be native or sm_XX.' }
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw 'nvcc is missing from PATH. Install the CUDA Toolkit and reopen PowerShell.'
}
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    if (-not $VisualStudioPath) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
        if (Test-Path -LiteralPath $vswhere) {
            $VisualStudioPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        }
    }
    if (-not $VisualStudioPath) { throw 'Install the Visual Studio Desktop development with C++ workload.' }
    $vcvars = Join-Path $VisualStudioPath 'VC/Auxiliary/Build/vcvars64.bat'
    if (-not (Test-Path -LiteralPath $vcvars)) { throw "Missing compiler setup: $vcvars" }
    # Import the compiler environment into this process; no persistent PATH change.
    $compilerEnvironment = & $env:ComSpec /d /c "call `"$vcvars`" >nul && set"
    if ($LASTEXITCODE -ne 0) { throw 'Visual Studio compiler environment setup failed.' }
    $seen = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($line in $compilerEnvironment) {
        if ($line -match '^([^=]+)=(.*)$' -and $seen.Add($Matches[1])) {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }
}
$buildDirectory = Join-Path $PSScriptRoot 'build'
New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null
if ($Precision -eq 'FP16') {
    if (-not $WeightsHeader) {
        $fullPrecision = Join-Path $PSScriptRoot 'weights.h'
        $WeightsHeader = if (Test-Path -LiteralPath $fullPrecision) { $fullPrecision } else { Join-Path $PSScriptRoot 'weights.h' }
    }
    $selected = (Resolve-Path -LiteralPath $WeightsHeader).Path.Replace('\', '/')
    if ($selected.Contains('"')) { throw 'Unsupported quote in header path.' }
    Set-Content -LiteralPath (Join-Path $buildDirectory 'weights_selected.h') -Encoding ascii -Value "#define CNN_WEIGHTS_HEADER `"$selected`""
    Write-Host "Weight source: $selected"
    if ($selected.EndsWith('/cuda/weights.h')) {
        Write-Warning 'The legacy header contains Q6.10-rounded weights. Export weights.h from the checkpoint for a full-precision source.'
    }
    $source = Join-Path $PSScriptRoot 'cnn_fp16.cu'
    $binary = Join-Path $buildDirectory 'cnn_fp16.exe'
    $libraries = @('-lcublas')
} else {
    if ($WeightsHeader) { throw 'Q6.10 reads a binary parameter blob at runtime; omit -WeightsHeader.' }
    $source = Join-Path $PSScriptRoot 'cnn_q6_10.cu'
    $binary = Join-Path $buildDirectory 'cnn_q6_10.exe'
    $libraries = @()
}
Push-Location $buildDirectory
try {
    & nvcc '-O3' '-std=c++17' "-arch=$Architecture" '-I' $buildDirectory '-Xcompiler=/O2,/utf-8' $source @libraries '-o' $binary
    if ($LASTEXITCODE -ne 0) { throw "CUDA compilation failed (exit $LASTEXITCODE)." }
} finally { Pop-Location }
Write-Host "Built $binary"
