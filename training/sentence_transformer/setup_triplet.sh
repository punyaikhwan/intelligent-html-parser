#!/bin/bash

# Setup script untuk fine-tuning Sentence Transformer

echo "Setting up environment for Sentence Transformer fine-tuning..."

# Create virtual environment
if [ ! -d "venv_triplet" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_triplet
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv_triplet/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements_triplet.txt

# Create directories for models and results
echo "Creating directories..."
mkdir -p models
mkdir -p results
mkdir -p logs

echo "Setup completed!"
echo ""
echo "To use the environment:"
echo "1. Activate: source venv_triplet/bin/activate"
echo "2. Edit triplet_data.json with your data"
echo "3. Run training: python triplet_training.py"
echo "4. Evaluate model: python evaluate_model.py"