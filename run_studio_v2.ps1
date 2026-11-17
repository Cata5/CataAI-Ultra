# RoCanon-48.5M Sci-Fi Research Workstation Launcher (PowerShell)
$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "  LLM Maker — NEURAL RESEARCH WORKSTATION" -ForegroundColor Cyan
Write-Host "  Frontend: Next.js App Router (http://localhost:8080)" -ForegroundColor Green
Write-Host "  Backend : PyTorch CUDA Engine (http://localhost:8081)" -ForegroundColor Yellow
Write-Host "  Compute : Dynamic GPU / CUDA VRAM Protection" -ForegroundColor DarkCyan
Write-Host "=========================================================================" -ForegroundColor Cyan

$PythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "[0/3] Freeing any stale processes on ports 8080 and 8081..." -ForegroundColor DarkGray
$conns = Get-NetTCPConnection -LocalPort 8080,8081 -ErrorAction SilentlyContinue
foreach ($c in $conns) {
    Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
}

Write-Host "[1/3] Verifying PyTorch CUDA environment..." -ForegroundColor Gray
& $PythonExe -c "import torch; print(f'PyTorch CUDA Active: {torch.cuda.is_available()}')"

Write-Host "[2/3] Starting Python ML Engine on 127.0.0.1:8081..." -ForegroundColor Gray
$BackendProcess = Start-Process -FilePath $PythonExe -ArgumentList "dashboard.py --port 8081" -PassThru -NoNewWindow

Start-Sleep -Seconds 3

Write-Host "[3/3] Starting Next.js Production Server on port 8080..." -ForegroundColor Gray
Set-Location -Path (Join-Path $PSScriptRoot "frontend")

if (-not (Test-Path ".next")) {
    Write-Host "Building production bundle..." -ForegroundColor Yellow
    npm run build
}

Start-Process "http://localhost:8080"

try {
    npm run start -- -p 8080
} finally {
    Write-Host "Shutting down..." -ForegroundColor Yellow
    if ($BackendProcess -and -not $BackendProcess.HasExited) {
        $BackendProcess.Kill()
    }
}
