@echo off
REM 使用虚拟环境运行train.py
call venv\Scripts\activate.bat
python train.py %*
if errorlevel 1 (
    echo.
    echo Error: Training failed. Make sure you're using the virtual environment.
    pause
)

