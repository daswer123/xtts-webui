@echo off

SET TEMP=temp

call venv/scripts/activate

python app.py --deepspeed --rvc

pause