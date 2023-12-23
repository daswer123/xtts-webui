import subprocess
import os
import json
import glob
from tqdm import tqdm
from pathlib import Path

def get_rvc_models(this_dir):
    rvc_models_base = this_dir / "rvc"
    # Исключаем папку base_models из сканирования
    exclude_dir = rvc_models_base / "base_models"

    models = []

    # Проходим по всем папкам внутри rvc, кроме base_models
    for model_dir in rvc_models_base.iterdir():
        if model_dir.is_dir() and model_dir != exclude_dir:
            pth_files = list(model_dir.glob('*.pth'))
            index_files = list(model_dir.glob('*.index'))
            model_name = model_dir.name

            # Если есть .pth файлы, добавляем информацию о моделях в список
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
            return model["model_path"] , model["index_path"]
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

    print("INPUT")
    print(pitch,index_rate,protect_voiceless,method,index_path,model_path,input_path,opt_path)
    print("OUTPUT")
    print(str(pitch), input_path, index_path, f0method, opt_path, model_path, str(index_rate), device, 
           str(is_half), str(filter_radius), str(resample_sr), str(rms_mix_rate), str(protect), str(crepe_hop_length), 
           str(f0_minimum), str(f0_maximum), str(autotune_enable))

    print("DONE")
    cmd = [
        'venv/scripts/python', 'scripts/rvc/test_infer.py',
        str(pitch), input_path, index_path,
        f0method, opt_path,
        model_path,
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