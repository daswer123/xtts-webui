#!/bin/bash

source venv/bin/activate

python scripts/modeldownloader.py
python app.py --deepspeed --rvc

echo "Launch"