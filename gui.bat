@echo off
setlocal

echo ========================================
echo Musubi Tuner Web UI
echo ========================================
echo.

cd /d "%~dp0"

REM Stop any existing Musubi Tuner GUI instances on port 7860
echo Checking for existing instances on port 7860...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7860 ^| findstr LISTENING 2^>nul') do (
    echo Stopping existing instance (PID: %%a)...
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

REM Check if .venv exists, if not create it
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate the virtual environment
call .venv\Scripts\Activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Force Accelerate to use the repo config (single-GPU by default)
set "ACCELERATE_CONFIG_FILE=%~dp0accelerate_config.yaml"

REM Check if the package is installed by trying to import it
python -c "import musubi_tuner" 2>nul
if errorlevel 1 (
    echo Installing/updating dependencies...
    pip install -e ".[gui]"
    if errorlevel 1 (
        echo Error: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Verify torch is available (GUI/training tabs typically need it)
python -c "import torch" 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: PyTorch is not installed in this venv.
    echo Install with ONE of these (pick your CUDA):
    echo   pip install -e ".[gui,cu128]"
    echo   pip install -e ".[gui,cu124]"
    echo.
    pause
    exit /b 1
)

echo.
echo Starting Musubi Tuner Web UI...
echo.

python musubi_gui.py --inbrowser

pause



