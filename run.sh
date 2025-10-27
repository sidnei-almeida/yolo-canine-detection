#!/bin/bash

# Script to run the Canine AI application

echo "🐕 Starting Canine AI..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if necessary
if [ ! -f "venv/.deps_installed" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.deps_installed
fi

# Run application
echo "🚀 Starting Streamlit application..."
echo ""
streamlit run app.py

# Deactivate virtual environment on exit
deactivate

