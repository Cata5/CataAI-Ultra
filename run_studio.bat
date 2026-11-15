@echo off
cd /d "%~dp0"
echo =========================================================================
echo   RoCanon-48.5M Studio & Training Platform v2 (NVIDIA RTX 3060 6GB)
echo =========================================================================
.\.venv\Scripts\python.exe dashboard.py %*
pause

