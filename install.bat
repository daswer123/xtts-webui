@echo off

python -m venv venv
call venv/scripts/activate

pip install torch==2.1.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r .\requirements.txt

echo Install complete.
pause