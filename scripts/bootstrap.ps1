<#
.SYNOPSIS
  One-command development/portable setup for LLM Maker.
.DESCRIPTION
  Creates .venv, installs Python dependencies, builds the frontend, and compiles
  the desktop shell. Safe to re-run; nothing outside the repository is modified
  except the Cargo target directory configured in .cargo/config.toml.
.PARAMETER Check
  Only verify prerequisites; do not install or build anything.
.PARAMETER SkipFrontend
  Skip npm install/build.
.PARAMETER SkipRust
  Skip cargo build.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
#>
param(
    [switch]$Check,
    [switch]$SkipFrontend,
    [switch]$SkipRust,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Fail([string]$Message) {
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

function Info([string]$Message) { Write-Host "==> $Message" -ForegroundColor Cyan }

function Resolve-Python {
    if ($PythonExe) { return $PythonExe }
    foreach ($candidate in @("py -3.12", "py -3.11", "python")) {
        $parts = $candidate.Split(" ")
        $pythonArgs = @()
        if ($parts.Length -gt 1) { $pythonArgs = $parts[1..($parts.Length - 1)] }
        try {
            $version = & $parts[0] @pythonArgs -c "import sys; print(str(sys.version_info.major) + chr(46) + str(sys.version_info.minor))" 2>$null
            if ($LASTEXITCODE -eq 0 -and [version]$version -ge [version]"3.10") { return $candidate }
        } catch { }
    }
    return $null
}

$python = Resolve-Python
if (-not $python) { Fail "Python 3.10+ not found. Install Python 3.11 or 3.12 and re-run." }
Info "Python: $python"

$node = Get-Command node -ErrorAction SilentlyContinue
$npm = Get-Command npm -ErrorAction SilentlyContinue
$cargo = Get-Command cargo -ErrorAction SilentlyContinue
if (-not $node -or -not $npm) { Write-Host "WARN: Node.js/npm not found; frontend build will be skipped." -ForegroundColor Yellow }
if (-not $cargo) { Write-Host "WARN: cargo not found; desktop build will be skipped." -ForegroundColor Yellow }

if ($Check) {
    Info "Prerequisite check only (-Check)."
    & $python -c "import sys; print(sys.executable)"
    if ($node) { node --version }
    if ($cargo) { cargo --version }
    exit 0
}

$venv = Join-Path $root ".venv"
$venvPython = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Info "Creating virtual environment (.venv)"
    & $python -m venv $venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create .venv" }
}

Info "Installing Python dependencies (this can take several minutes on first run)"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "Python dependency installation failed" }

if (-not $SkipFrontend -and $node -and $npm) {
    Info "Installing frontend dependencies"
    Push-Location (Join-Path $root "frontend")
    try {
        npm ci
        if ($LASTEXITCODE -ne 0) { Fail "npm ci failed" }
        Info "Building frontend"
        npm run build
        if ($LASTEXITCODE -ne 0) { Fail "Frontend build failed" }
    } finally { Pop-Location }
}

if (-not $SkipRust -and $cargo) {
    Info "Building desktop shell (debug)"
    Push-Location (Join-Path $root "src-tauri")
    try {
        cargo build
        if ($LASTEXITCODE -ne 0) { Fail "cargo build failed" }
    } finally { Pop-Location }
}

Info "Setup complete."
Write-Host ""
Write-Host "Start the desktop app:"
Write-Host "  cargo run --manifest-path src-tauri/Cargo.toml   # native desktop shell"
Write-Host "  npm --prefix frontend run dev                    # browser UI on http://localhost:3000"
Write-Host ""
Write-Host "Data lives in .\data and .\studio.db; schema upgrades create backups in .\backups."