@echo off
title TrainIQ Backend
set PYTHONUTF8=1
cd /d C:\Users\arman\trainiq\backend

:restart
echo.
echo [%time%] Iniciando TrainIQ backend...
.venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port 8000
echo.
echo [%time%] Backend parou. Reiniciando em 3 segundos...
timeout /t 3 /nobreak >nul
goto restart
