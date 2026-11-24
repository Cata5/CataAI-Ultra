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

if exist "%USERPROFILE%\.cargo\bin" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
if not defined CARGO_TARGET_DIR set "CARGO_TARGET_DIR=%LOCALAPPDATA%\LLMMaker\cargo-target"

echo [0/4] Freeing any previous running llm-maker instances...
powershell -NoProfile -Command "Stop-Process -Name 'llm-maker' -Force -ErrorAction SilentlyContinue"
timeout /t 1 /nobreak >nul

if "%1"=="quick" (
    echo [Quick mode] Skipping rebuild...
    goto launch
)

if "%1"=="dev" (
    echo Starting Tauri dev mode...
    call "frontend\node_modules\.bin\tauri.cmd" dev
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
set "BUILT=%CARGO_TARGET_DIR%\debug\llm-maker.exe"
if not exist "%BUILT%" set "BUILT=%CARGO_TARGET_DIR%\x86_64-pc-windows-msvc\debug\llm-maker.exe"
if not exist "%BUILT%" set "BUILT=%CARGO_TARGET_DIR%\x86_64-pc-windows-gnu\debug\llm-maker.exe"
if not exist "%BUILT%" (
    echo [ERROR] Built llm-maker.exe not found under "%CARGO_TARGET_DIR%".
    pause
    exit /b 1
)
copy /Y "%BUILT%" "%~dp0llm-maker.exe" >nul
copy /Y "%BUILT%" "%~dp0src-tauri\target\release\llm-maker.exe" >nul

:launch
echo [4/4] Launching LLM Maker Native Desktop...
start "" "%~dp0llm-maker.exe"
echo LLM Maker launched successfully.
