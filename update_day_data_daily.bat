@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    py -3 update_day_data.py %*
    exit /b %errorlevel%
)
"C:\Users\Administrator\AppData\Local\Programs\Python\Python314\python.exe" update_day_data.py %*
