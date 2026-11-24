# Verifies the staged installer payload before packaging.
param([string]$Root = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = "Stop"
$required = @(
    "src-tauri\resources\app\src\worker\standalone_runner.py",
    "src-tauri\resources\app\installer\__main__.py",
    "src-tauri\resources\app\requirements.txt",
    "src-tauri\resources\python\python.exe",
    "frontend\out\index.html"
)
foreach ($relative in $required) {
    $path = Join-Path $Root $relative
    if (-not (Test-Path $path)) { throw "Missing installer payload: $relative" }
}
Write-Host "Installer payload OK"
