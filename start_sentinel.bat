@echo off
title Project Sentinel Launcher
color 0b

echo ==========================================================
echo           PROJECT SENTINEL PLATFORM LAUNCHER
echo ==========================================================
echo.

:: Check if backend exists
if not exist "backend\main.py" (
    echo [ERROR] Backend folder or main.py not found!
    echo Please run this file from the Sentinel root directory.
    pause
    exit /b
)

:: Start backend in a new command window
echo [INFO] Starting Sentinel FastAPI backend...
start "Sentinel Backend Server" cmd /k "cd backend && python -m uvicorn main:app --host 127.0.0.1 --port 8000"

:: Wait for a moment to let backend start up
timeout /t 3 /nobreak >nul

:: Open frontend in default browser
echo [INFO] Launching Sentinel frontend...
start "" "frontend\index.html"

echo.
echo ==========================================================
echo Sentinel is active!
echo - API is served at http://localhost:8000
echo - API Docs: http://localhost:8000/docs
echo.
echo You can close this window now. Keep the backend window open.
echo ==========================================================
timeout /t 5
