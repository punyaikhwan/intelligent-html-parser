#!/bin/bash

# Create virtual environment
python3 -m venv venv_llm
source venv_llm/bin/activate

python run_training.py