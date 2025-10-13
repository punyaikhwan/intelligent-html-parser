#!/bin/bash

# Create virtual environment
python3 -m venv venv_llm
source venv_llm/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

echo "LLM training environment setup complete!"
echo "To activate the environment, run: source venv_llm/bin/activate"