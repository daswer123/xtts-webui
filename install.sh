#!/bin/bash

# Create a Python virtual environment
python -m venv venv
# Activate the virtual environment
source venv/bin/activate

# Install other dependencies from requirements.txt
pip install -r requirements.txt
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118

echo "Install deepspeed for Linux for python 3.10.x and CUDA 11.8"
python scripts/modeldownloader.py

echo "Install complete."