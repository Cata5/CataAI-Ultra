@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ===========================================================================
echo       LLM MAKER — NATIVE TAURI 2 RUST DESKTOP WORKSTATION
echo ===========================================================================
echo  * Fast Cold Startup (No eager PyTorch loading)
echo  * Zero Idle Python/CUDA Memory Overhead (~13 MB Idle RAM)
echo  * Embedded Static Frontend (No continuous localhost server)
echo  * Isolated On-Demand ML Worker Process
echo ===========================================================================

set PATH=C:\Users\Cata\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT.LLVM_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin;C:\Users\Cata\.cargo\bin;%PATH%
set "CARGO_TARGET_DIR=C:\Users\Cata\.cargo\target_llm_maker"

echo [0/4] Freeing any previous running llm-maker instances...
powershell -NoProfile -Command "Stop-Process -Name 'llm-maker' -Force -ErrorAction SilentlyContinue"
timeout /t 1 /nobreak >nul

if "%1"=="quick" (
    echo [Quick mode] Skipping rebuild...
    goto launch
)

if "%1"=="dev" (
    echo Starting Tauri dev mode...
    cd frontend && npx tauri dev
    exit /b %ERRORLEVEL%
)

echo [1/4] Building Next.js static export for Tauri...
cd frontend
set TAURI_BUILD=true
call npm run build
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Next.js static export build failed!
    pause
    exit /b 1
)
cd ..

echo [2/4] Compiling Native Tauri Desktop binary...
powershell -NoProfile -Command "(Get-Item 'src-tauri/src/main.rs').LastWriteTime = Get-Date"
cd src-tauri
cargo build
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Cargo desktop build failed!
    pause
    exit /b 1
)
cd ..

echo [3/4] Updating root llm-maker.exe executable...
copy /Y "C:\Users\Cata\.cargo\target_llm_maker\x86_64-pc-windows-gnu\debug\llm-maker.exe" "%~dp0llm-maker.exe" >nul
copy /Y "C:\Users\Cata\.cargo\target_llm_maker\x86_64-pc-windows-gnu\debug\llm-maker.exe" "%~dp0src-tauri\target\release\llm-maker.exe" >nul

:launch
echo [4/4] Launching LLM Maker Native Desktop...
start "" "%~dp0llm-maker.exe"
echo LLM Maker launched successfully.
