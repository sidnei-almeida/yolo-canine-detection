#!/bin/bash

# Script to run the Canine AI application

echo "🐕 Starting DogBreed Vision API..."
echo ""

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

# Run API
echo "🚀 Starting FastAPI server..."
echo ""
uvicorn app:app --host 0.0.0.0 --port 7860

# Deactivate virtual environment on exit
deactivate

