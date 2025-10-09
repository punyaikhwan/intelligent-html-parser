#!/bin/bash

# Intelligent HTML Parser Startup Script

echo "Starting Intelligent HTML Parser..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export HOST=0.0.0.0
export PORT=5000

echo "Starting Flask application..."
echo "API will be available at: http://localhost:5000"
echo "Health check: http://localhost:5000/"
echo "Parser status: http://localhost:5000/status"
echo "Parse endpoint: POST http://localhost:5000/parse"

# Start the application
python app.py