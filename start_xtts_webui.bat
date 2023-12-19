@echo off

call venv/scripts/activate
python app.py --deepspeed

pause