@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo =========================================================================
echo   LLM Maker — NEURAL RESEARCH WORKSTATION
echo   Active Model: RoCanon-48.5M Causal LM
echo   Frontend: Next.js App Router (http://localhost:8080)
echo   Backend : PyTorch CUDA Engine (http://localhost:8081)
echo   Compute : NVIDIA RTX 3060 Laptop (6GB VRAM)
echo =========================================================================

set PYTHON_EXE=.\.venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

echo [0/3] Freeing any stale processes on ports 8080 and 8081...
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 8080,8081 -ErrorAction SilentlyContinue; if ($c) { $c | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } }"

echo [1/3] Checking Python backend environment...
"%PYTHON_EXE%" -c "import torch; print(f'PyTorch CUDA Active: {torch.cuda.is_available()}')"

echo [2/3] Launching Python Decoupled ML Backend on 127.0.0.1:8081...
start "RoCanon Backend (Port 8081)" /B "%PYTHON_EXE%" dashboard.py --port 8081 %*

echo Waiting 3 seconds for backend control plane to initialize...
timeout /t 3 /nobreak >nul

echo [3/3] Starting Next.js Sci-Fi Workstation on http://localhost:8080...
cd frontend
if not exist ".next" (
    echo Building Next.js production bundle...
    call npm run build
)

echo Opening Workstation in default browser...
start http://localhost:8080

call npm run start -- -p 8080

pause
