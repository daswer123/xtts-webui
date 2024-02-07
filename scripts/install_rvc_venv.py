import os
import subprocess
import sys
import platform

# Create a virtual environment rvc_venv inside the venv folder
subprocess.run([sys.executable, "-m", "venv", "venv/rvc_venv"])

if sys.platform == "win32":  # Если операционная система - Windows.
    activate_script = "venv\\rvc_venv\\Scripts\\activate"
    commands_after_activation = [
        f"{activate_script} && pip install rvc-python",
        f"{activate_script} && pip install openvoice-cli",
        f"{activate_script} && pip install torch==1.13.1+cu117 torchaudio==0.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117"
    ]
else:  # For Unix-like systems
    activate_script = "venv\\rvc_venv\\bin\\activate"
    commands_after_activation = [
        f"source {activate_script} && pip install rvc-python",
        f"source {activate_script} && pip install openvoice-cli",
        f"source {activate_script} && pip install torch==1.13.1+cu117 torchaudio==0.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117"
    ]

for command in commands_after_activation:
    subprocess.run(command, shell=True)