# CardioRisk AI - PowerShell Application Launcher
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "              CARDIORISK AI v3.0.0 - SYSTEM LAUNCHER" -ForegroundColor Green
Write-Host "         Multi-Modal Physiological Deep Learning Decision Platform" -ForegroundColor Cyan
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonExe = Join-Path $ScriptDir "venv\Scripts\python.exe"
$StreamlitExe = Join-Path $ScriptDir "venv\Scripts\streamlit.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Host "[ERROR] Virtual environment python.exe not found at $PythonExe" -ForegroundColor Red
    exit 1
}

Write-Host "[1/2] Starting FastAPI Backend with HTTPS on https://0.0.0.0:8000..." -ForegroundColor Yellow
Start-Process -FilePath $PythonExe -ArgumentList "-m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem" -WindowStyle Normal

Start-Sleep -Seconds 3

Write-Host "[2/2] Starting Streamlit Frontend on http://localhost:8501..." -ForegroundColor Yellow
Start-Process -FilePath $StreamlitExe -ArgumentList "run frontend/app.py" -WindowStyle Normal

Write-Host ""
Write-Host "===============================================================================" -ForegroundColor Green
Write-Host " [SUCCESS] CardioRisk AI is now running!" -ForegroundColor Green
Write-Host ""
Write-Host " * Web Landing Page:     http://localhost:8000/landing/" -ForegroundColor White
Write-Host " * Streamlit UI:          http://localhost:8501" -ForegroundColor White
Write-Host " * Swagger Documentation: http://localhost:8000/docs" -ForegroundColor White
Write-Host " * Interactive Tools:     http://localhost:8000/tools/ppg_capture.html" -ForegroundColor White

Write-Host ""
Write-Host " Default Credentials:" -ForegroundColor Cyan
Write-Host "  - Doctor:  doctor  / doctor123" -ForegroundColor White
Write-Host "  - Patient: patient / patient123" -ForegroundColor White
Write-Host "===============================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press Enter to exit this launcher window..."
Read-Host
