@echo off
REM Agri-Mind Dashboard - Quick Setup Script for Windows
REM This script sets up and runs the dashboard with one command

echo 🌾 Agri-Mind Dashboard - Quick Setup
echo ====================================
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python found
python --version
echo.

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install requirements
echo 📥 Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo ✅ Setup complete!
echo.
echo 🚀 Starting Agri-Mind Dashboard...
echo.
echo 📍 The dashboard will open at: http://localhost:8501
echo 🎯 Demo Mode is enabled by default for easy testing
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run Streamlit
streamlit run app.py

pause
