# XTTS-WebUI

## About the Project
XTTS-WebUI offers two interfaces designed to enhance your experience with XTTS. One interface serves as a user-friendly approach to interact with XTTS, while the other is an enhanced version of the web interface intended for fine-tuning. This project simplifies interaction with XTTS through two files and provides a feature-rich web interface for optimal utilization.

## 1 - webui 
![image](https://github.com/daswer123/xtts-webui/assets/22278673/9849e558-a7a6-471f-89f2-32acc74ebb38)

## 2 - finetune webui
![image](https://github.com/daswer123/xtts-webui/assets/22278673/f0d7768e-77e4-4087-acb2-175eeddc9d2c)

![image](https://github.com/daswer123/xtts-webui/assets/22278673/0b5b1b99-2678-4cd0-ae80-bebf66b06e3d)

## Key Features
- **xtts_finetune_webui.py**: An improved version of the official web interface tailored for ease of use on personal computers.
- **xtts_webui.py**: Original work that will be further developed; currently operates flawlessly, featuring numerous settings and quick image addition capabilities.

If you don't know where to start for xtts-finetune-webui, watch this [video](https://www.youtube.com/watch?v=8tpDiiouGxc)

## Install

1. Make sure you have `CUDA` installed
2. `https://github.com/daswer123/xtts-webui`
3. `cd xtts-webui`
4. `python -m venv venv`
5. `venv\scripts\activate` or `source venv\scripts\activate` if you use linux
6. `pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118`
7. `pip install -r requirements.txt`

### If you're using Windows

1. First start `install.bat`
2. To start the webui, start `start_xtts_webui.bat` for webui
3. If you want start finetune webui, start `start_xtts_finetune_webui.bat` 
4. Go to the local address, you can see it in console

## Differences between xtts-finetune webui and the [official webui](https://github.com/coqui-ai/TTS/pull/3296)

### Data processing

1. Updated faster-whisper to 0.10.0 with the ability to select a larger-v3 model.
2. Changed output folder to output folder inside the main folder.
3. If there is already a dataset in the output folder and you want to add new data, you can do so by simply adding new audio, what was there will not be processed again and the new data will be automatically added
4. Turn on VAD filter
5. After the dataset is created, a file is created that specifies the language of the dataset. This file is read before training so that the language always matches. It is convenient when you restart the interface

### Fine-tuning XTTS Encoder

1. Added the ability to select the base model for XTTS, as well as when you re-training does not need to download the model again.
2. Added ability to select custom model as base model during training, which will allow finetune already finetune model.
3. Added possibility to get optimized version of the model for 1 click ( step 2.5, put optimized version in output folder).
4. You can choose whether to delete training folders after you have optimized the model
5. When you optimize the model, the example reference audio is moved to the output folder
6. Checking for correctness of the specified language and dataset language

### Inference

1. Added possibility to customize infer settings during model checking.

### Other

1. If you accidentally restart the interface during one of the steps, you can load data to additional buttons
2. Removed the display of logs as it was causing problems when restarted
3. The finished result is copied to the ready folder, these are fully finished files, you can move them anywhere and use them as a standard model




