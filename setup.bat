@echo off
REM ============================================================
REM  AI-Era SSD Emulator -- Windows Setup Script
REM  Run once from the project root:  .\setup.bat
REM ============================================================

setlocal EnableDelayedExpansion

set PROJECT_DIR=%~dp0
set VENV_DIR=%PROJECT_DIR%ai_ssd_env

echo.
echo ============================================================
echo   AI-Era SSD Emulator Setup (Windows)
echo ============================================================
echo Project root : %PROJECT_DIR%
echo Virtual env  : %VENV_DIR%
echo.

REM -- 1. Check Python -----------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.  Please install Python 3.10+ and add it to PATH.
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] Python %PY_VER% detected.
REM -- Validate Python version and architecture (require 64-bit Python 3.10/3.11 recommended)
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
for /f "delims=" %%a in ('python -c "import platform;print(platform.architecture()[0])"') do set PY_ARCH=%%a
echo [INFO] Python architecture: %PY_ARCH%
if "%PY_ARCH%" NEQ "64bit" (
    echo [WARN] 32-bit Python detected. This project requires 64-bit Python for prebuilt wheels.
    echo [WARN] Install 64-bit Python ^(3.10 or 3.11^) from https://www.python.org/downloads/windows/ and re-run setup.
    echo.
)
if %PY_MAJOR% GTR 3 (
    echo [WARN] Detected Python major version %PY_MAJOR%. Use Python 3.10 or 3.11 for best compatibility.
) else (
    if %PY_MAJOR% EQU 3 (
        if %PY_MINOR% GTR 11 (
            echo [WARN] Detected Python %PY_VER%. Some packages may not provide wheels for this version.
            echo [WARN] Consider using Python 3.11 ^(64-bit^) to avoid building from source.
        )
    )
)

REM -- 2. Create virtual environment ---------------------------------
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [SKIP] Virtual environment already exists at %VENV_DIR%
) else (
    echo [....] Creating virtual environment...
    REM Prefer Python 3.11 if the py launcher is available
    py -3.11 --version >nul 2>&1
    if %errorlevel% EQU 0 (
        py -3.11 -m venv "%VENV_DIR%"
    ) else (
        echo [INFO] Python 3.11 not found via py launcher; using default `python -m venv`.
        python -m venv "%VENV_DIR%"
    )
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

REM -- 3. Activate venv ----------------------------------------------
call "%VENV_DIR%\Scripts\activate.bat"
echo [OK] Virtual environment activated.

REM -- 4. Upgrade pip ------------------------------------------------
echo [....] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip upgraded.

REM -- 5. Install dependencies ---------------------------------------
echo [....] Installing dependencies (numpy, psutil, matplotlib, torch)...
echo        This may take a few minutes on first run.

REM Install CPU-only torch to keep download size manageable on a laptop.
REM Remove the --index-url line if you want CUDA support and have a GPU.
pip install --quiet numpy==1.26.4
pip install --quiet psutil==5.9.8
pip install --quiet matplotlib==3.9.0
REM Install a CPU-compatible PyTorch wheel; avoid a strict pin so pip can select a compatible build
pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch

if errorlevel 1 (
    echo [ERROR] Dependency installation failed.  Check your internet connection and try again.
    exit /b 1
)
echo [OK] All dependencies installed.

REM -- 6. Ensure project directories exist --------------------------
echo [....] Ensuring project directories...
if not exist "%PROJECT_DIR%data"    mkdir "%PROJECT_DIR%data"
if not exist "%PROJECT_DIR%results" mkdir "%PROJECT_DIR%results"
if not exist "%PROJECT_DIR%logs"    mkdir "%PROJECT_DIR%logs"
echo [OK] Directories ready: data\  results\  logs\

REM -- 7. Done -------------------------------------------------------
echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo To run the benchmark:
echo   1.  Activate the virtual environment:
echo       %VENV_DIR%\Scripts\activate.bat
echo   2.  Run the emulator:
echo       python emulator.py
echo.
echo Press any key to exit setup...
pause >nul
endlocal
