import os
import subprocess
import sys

# Create a virtual environment rvc_venv inside the venv folder
subprocess.run([sys.executable, "-m", "venv", f"venv{os.sep}rvc_venv"])

if sys.platform == "win32":  # If operating system is Windows.
    activate_script_path = f"venv{os.sep}rvc_venv{os.sep}Scripts{os.sep}activate"
else:  # For Unix-like systems.
    activate_script_path = f"venv{os.sep}rvc_venv{os.sep}bin{os.sep}activate"

commands_after_activation = [
    f"{sys.executable} -m pip install rvc-python",
    f"{sys.executable} -m pip install openvoice-cli",
    (f"{sys.executable} -m pip install torch==1.13.1+cu117 "
     f"torchaudio==0.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117")
]

for command in commands_after_activation:
    subprocess.run(f"{activate_command()} && {command}", shell=True)

def activate_command():
    if sys.platform == "win32":
        return f"{activate_script_path}"
    else:
        return f". {activate_script_path}"
