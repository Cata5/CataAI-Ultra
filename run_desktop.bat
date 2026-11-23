@echo off
setlocal
echo ===========================================================================
echo       LLM MAKER — NATIVE TAURI 2 RUST DESKTOP WORKSTATION
echo ===========================================================================
echo  * Fast Cold Startup (No eager PyTorch loading)
echo  * Zero Idle Python/CUDA Memory Overhead (~13 MB Idle RAM)
echo  * Embedded Static Frontend (No continuous localhost server)
echo  * Isolated On-Demand ML Worker Process
echo ===========================================================================
set PATH=C:\Users\Cata\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT.LLVM_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin;C:\Users\Cata\.cargo\bin;%PATH%
set CARGO_TARGET_DIR=C:\Users\Cata\.cargo\target_llm_maker

if "%1"=="dev" (
    echo Starting Tauri dev mode...
    cd frontend && npx tauri dev
) else (
    echo Launching Native Desktop App...
    start "" "%~dp0llm-maker.exe"
)
