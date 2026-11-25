# Build a Windows NSIS installer with a private CPython runtime and app sources.
# Usage:
#   .\scripts\build_installer.ps1 -PythonArchive C:\downloads\cpython-3.11.zip -PythonSha256 <hex>
# The Python archive is never downloaded implicitly: pin the exact digest in CI.
param(
    [Parameter(Mandatory = $true)][string]$PythonArchive,
    [Parameter(Mandatory = $true)][string]$PythonSha256,
    [string]$RustToolchain = ""
)

$ErrorActionPreference = "Stop"
if ($RustToolchain) {
    $env:RUSTUP_TOOLCHAIN = $RustToolchain
}
if (-not $env:CARGO_TARGET_DIR) {
    # Git for Windows' windres mishandles spaces in the default target path.
    # Keep the target directory outside the project tree.
    $env:CARGO_TARGET_DIR = Join-Path $env:LOCALAPPDATA "LLMMaker\cargo-target"
}
$root = Split-Path -Parent $PSScriptRoot
$resources = Join-Path $root "src-tauri\resources"
$pythonDir = Join-Path $resources "python"
$appDir = Join-Path $resources "app"

if (-not (Test-Path $PythonArchive)) { throw "Python archive not found: $PythonArchive" }
$actual = (Get-FileHash -Algorithm SHA256 -Path $PythonArchive).Hash.ToLowerInvariant()
if ($actual -ne $PythonSha256.ToLowerInvariant()) {
    throw "Python archive SHA-256 mismatch: expected $PythonSha256, got $actual"
}

Write-Host "Staging CPython runtime..."
New-Item -ItemType Directory -Force -Path $pythonDir | Out-Null
Get-ChildItem -Force $pythonDir | Where-Object { $_.Name -ne ".gitkeep" } | Remove-Item -Recurse -Force
$extract = Join-Path $env:TEMP ("llmmaker-python-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $extract | Out-Null
try {
    if ($PythonArchive.EndsWith(".zip")) {
        Expand-Archive -Path $PythonArchive -DestinationPath $extract -Force
    } else {
        # Git for Windows ships GNU tar, which reads "C:\..." as a remote host.
        # Extract from the archive's directory so the path has no drive colon.
        $archivePath = (Resolve-Path $PythonArchive).Path
        Push-Location (Split-Path -Parent $archivePath)
        try {
            tar -xf (Split-Path -Leaf $archivePath) -C $extract
            if ($LASTEXITCODE -ne 0) { throw "Python archive extraction failed" }
        } finally {
            Pop-Location
        }
    }
    $entries = @(Get-ChildItem -Force $extract)
    $source = if ($entries.Count -eq 1 -and $entries[0].PSIsContainer) { $entries[0].FullName } else { $extract }
    Copy-Item -Path (Join-Path $source "*") -Destination $pythonDir -Recurse -Force
    $pythonExe = Get-ChildItem -Path $pythonDir -Recurse -Filter "python.exe" | Select-Object -First 1
    if (-not $pythonExe) { throw "No python.exe found in the staged archive" }
    if ($pythonExe.Directory.FullName -ne $pythonDir) {
        Copy-Item -Path (Join-Path $pythonExe.Directory.FullName "*") -Destination $pythonDir -Recurse -Force
    }
} finally {
    Remove-Item -Recurse -Force $extract -ErrorAction SilentlyContinue
}

Write-Host "Staging application sources..."
New-Item -ItemType Directory -Force -Path $appDir | Out-Null
Get-ChildItem -Force $appDir | Where-Object { $_.Name -ne ".gitkeep" } | Remove-Item -Recurse -Force
foreach ($item in @("src", "installer", "configs", "requirements.txt", "train.py")) {
    $source = Join-Path $root $item
    if (Test-Path $source) {
        Copy-Item -Path $source -Destination $appDir -Recurse -Force
    }
}
Get-ChildItem -Path $appDir -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path $appDir -Recurse -File -Filter "*.pyc" | Remove-Item -Force

Write-Host "Building static frontend..."
Push-Location (Join-Path $root "frontend")
try {
    $env:TAURI_BUILD = "true"
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "Frontend build failed" }
} finally {
    Pop-Location
}

Write-Host "Building NSIS installer..."
$tauriCli = Join-Path $root "frontend\node_modules\.bin\tauri.cmd"
if (-not (Test-Path $tauriCli)) { throw "Tauri CLI missing. Run npm ci in frontend first." }
Push-Location $root
try {
    & $tauriCli build --bundles nsis
    if ($LASTEXITCODE -ne 0) { throw "Tauri build failed" }
} finally {
    Pop-Location
}
$bundleDir = Join-Path $env:CARGO_TARGET_DIR "release\bundle\nsis"
Write-Host "Installer written under $bundleDir"
$installer = Get-ChildItem (Join-Path $bundleDir "*.exe") | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($installer) {
    $hash = (Get-FileHash -Algorithm SHA256 $installer.FullName).Hash.ToLowerInvariant()
    "$hash  $($installer.Name)" | Set-Content "$($installer.FullName).sha256" -Encoding ascii
    Write-Host "SHA-256: $hash"
}
