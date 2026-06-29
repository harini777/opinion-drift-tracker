#!/bin/bash
# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "🚀 Starting Opinion Drift Tracker in virtual environment..."
    ./venv/bin/streamlit run app.py
else
    echo "⚠️ venv directory not found! Running with system streamlit..."
    streamlit run app.py
fi
