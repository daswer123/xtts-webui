import subprocess
import os
import json
import glob
from tqdm import tqdm

# def infer_rvc(f0up_key: int, input_path: str, index_path: str, f0method: str, opt_path: str, model_path: str, index_rate: float, 
#                device: str, is_half: bool, filter_radius: int, resample_sr: int, rms_mix_rate: float, protect: float, 
#                crepe_hop_length: int, f0_minimum: int, f0_maximum: int, autotune_enable: bool):
#     cmd = ['venv/scripts/python', 'libs/rvc/test_infer.py', 
#            str(f0up_key), input_path, index_path, f0method, opt_path, model_path, str(index_rate), device, 
#            str(is_half), str(filter_radius), str(resample_sr), str(rms_mix_rate), str(protect), str(crepe_hop_length), 
#            str(f0_minimum), str(f0_maximum), str(autotune_enable)]
#     subprocess.run(cmd)

# Example main(0, input.wav, model.index, model.pth, output.wav)
def infer_rvc(f0up_key,input_path,index_path,model_path,opt_path):
    f0method = "rmvpe"
    index_rate = 0.5
    device = "cuda:0"
    is_half = True
    filter_radius = 3
    resample_sr = 0
    rms_mix_rate = 1
    protect = 0.33
    crepe_hop_length = 128
    f0_minimum = 50
    f0_maximum = 1100
    autotune_enable = False

    cmd = ['venv/scripts/python', 'libs/rvc/test_infer.py', 
           str(f0up_key), input_path, index_path, f0method, opt_path, model_path, str(index_rate), device, 
           str(is_half), str(filter_radius), str(resample_sr), str(rms_mix_rate), str(protect), str(crepe_hop_length), 
           str(f0_minimum), str(f0_maximum), str(autotune_enable)]
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

