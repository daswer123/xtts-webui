import subprocess
import os
import json
import glob
from tqdm import tqdm
from pathlib import Path
import requests

def download_rvc_models():
    folder = './rvc/base_model'
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    files = {
        "hubert_base.pt": "https://huggingface.co/Daswer123/RVC_Base/resolve/main/hubert_base.pt",
        "rmvpe.pt": "https://huggingface.co/Daswer123/RVC_Base/resolve/main/rmvpe.pt"
    }
    
    for filename, url in files.items():
        file_path = os.path.join(folder, filename)
    
        if not os.path.exists(file_path):
            print(f'File {filename} not found, start loading...')
    
            response = requests.get(url)
    
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f'File {filename} successfully loaded.')
            else:
                print(f'f {filename}.')
        else:
            print(f'File {filename} already exists.')

def get_rvc_models(this_dir):
    rvc_models_base = this_dir / "rvc"
    # exclude the base_models folder from scanning
    exclude_dir = rvc_models_base / "base_models"

    models = []

    # Go through all folders inside rvc except base_models
    for model_dir in rvc_models_base.iterdir():
        if model_dir.is_dir() and model_dir != exclude_dir:
            pth_files = list(model_dir.glob('*.pth'))
            index_files = list(model_dir.glob('*.index'))
            model_name = model_dir.name

            # If there are .pth files, add model information to the list
            if pth_files:
                model_info = {'model_name': model_name}
                model_info['model_path'] = str(pth_files[0].absolute())
                if index_files:
                    model_info['index_path'] = str(index_files[0].absolute())
                models.append(model_info)

    return models

def find_rvc_model_by_name(this_dir,model_name):
    print(model_name)
    models = get_rvc_models(this_dir)
    for model in models:
        if model['model_name'] == model_name:
            # Check model_index key if it's exists 
            model_index = model.get('index_path', None)
            return model["model_path"] , model_index
    return None


def infer_rvc(pitch,index_rate,protect_voiceless,method,index_path,model_path,input_path,opt_path):
    f0method = method
    device = "cuda:0"
    is_half = True
    filter_radius = 3
    resample_sr = 0
    rms_mix_rate = 1
    protect = protect_voiceless
    crepe_hop_length = 128
    f0_minimum = 50
    f0_maximum = 1100
    autotune_enable = False

    index_path = str(index_path)
    model_path = str(model_path)
    try:
        cmd = [
            'venv/rvc_venv/scripts/python', 'scripts/rvc/test_infer.py',
            str(pitch), input_path,model_path,
            f0method, opt_path,
            index_path,
            str(index_rate),
            device,
            str(is_half).lower(),  # Convert boolean to lowercase string ('true'/'false')
            str(filter_radius),
            str(resample_sr),
            str(rms_mix_rate),
            str(protect).lower(),  # Convert boolean to lowercase string ('true'/'false')
            str(crepe_hop_length),
            str(f0_minimum),
            str(f0_maximum),
            str(autotune_enable).lower()  # Convert boolean to lowercase string ('true'/'false')
        ]
        subprocess.run(cmd)
    except Exception as e:
        print(f"Error: {e}")
        return False
    

from tqdm import tqdm

def infer_files(input_dir,config_path):
    # Загрузка конфигурации
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Получение списка всех файлов в input_dir
    files = glob.glob(os.path.join(input_dir, '*'))

    # Проверка существования директории output и создание ее, если необходимо
    output_dir = os.path.join(input_dir, '../output')
    os.makedirs(output_dir, exist_ok=True)

    # Обработка каждого файла
    for file in tqdm(files, desc="Processing files"):
        file_name = os.path.splitext(os.path.basename(file))[0]
        for character, character_config in config.items():
            if character in file_name:
                model_path = character_config['model_path']
                model_index = character_config['model_index']
                f0up_key = character_config['pitch']
                opt_path = os.path.join(output_dir, f'{file_name}.mp3')

                infer_rvc(f0up_key, file, model_index, model_path, opt_path)
                break  # Если мы нашли соответствующего персонажа, прерываем цикл


# def infer_rvc(f0up_key: int, input_path: str, index_path: str, f0method: str, opt_path: str, model_path: str, index_rate: float, 
#                device: str, is_half: bool, filter_radius: int, resample_sr: int, rms_mix_rate: float, protect: float, 
#                crepe_hop_length: int, f0_minimum: int, f0_maximum: int, autotune_enable: bool):
#     cmd = ['venv/scripts/python', 'libs/rvc/test_infer.py', 
#            str(f0up_key), input_path, index_path, f0method, opt_path, model_path, str(index_rate), device, 
#            str(is_half), str(filter_radius), str(resample_sr), str(rms_mix_rate), str(protect), str(crepe_hop_length), 
#            str(f0_minimum), str(f0_maximum), str(autotune_enable)]
#     subprocess.run(cmd)

# Example main(0, input.wav, model.index, model.pth, output.wav)