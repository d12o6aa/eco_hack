#!/bin/bash

# Agri-Mind Dashboard - Quick Setup Script
# This script sets up and runs the dashboard with one command

echo "🌾 Agri-Mind Dashboard - Quick Setup"
echo "===================================="
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Mac/Linux
    source venv/bin/activate
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Starting Agri-Mind Dashboard..."
echo ""
echo "📍 The dashboard will open at: http://localhost:8501"
echo "🎯 Demo Mode is enabled by default for easy testing"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run Streamlit
streamlit run app.py
