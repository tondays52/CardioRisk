@echo off
title CardioRisk AI - Multi-Modal Clinical Decision Platform
color 0A
echo ===============================================================================
echo               CARDIORISK AI v3.0.0 - SYSTEM LAUNCHER
echo          Multi-Modal Physiological Deep Learning Decision Platform
echo ===============================================================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .\venv
    echo Please make sure the venv directory exists.
    pause
    exit /b 1
)

echo [1/2] Launching FastAPI Backend with HTTPS on port 8000 (0.0.0.0)...
start "CardioRisk AI - Backend Server" cmd /k "title Backend API (Port 8000 HTTPS) && color 0B && .\venv\Scripts\python.exe -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem"

echo [WAIT] Giving backend 3 seconds to initialize...
timeout /t 3 /nobreak >nul

echo [2/2] Launching Streamlit Frontend Dashboard on port 8501...
start "CardioRisk AI - Frontend Dashboard" cmd /k "title Streamlit Dashboard (Port 8501) && color 0E && .\venv\Scripts\streamlit.exe run frontend/app.py"

echo.
echo ===============================================================================
echo  [SUCCESS] CardioRisk AI is now running!
echo  
echo  * Web Landing Page:   http://localhost:8000/landing/
echo  * Frontend Dashboard: http://localhost:8501
echo  * Backend API Docs:   http://localhost:8000/docs
echo  * Interactive Tools:  http://localhost:8000/tools/ppg_capture.html
echo  
echo  Default Doctor Login:  doctor / doctor123
echo  Default Patient Login: patient / patient123
echo ===============================================================================

echo.
echo Leave this launcher window or close it when done.
pause
