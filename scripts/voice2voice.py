import subprocess
import os
import json
import glob
from tqdm import tqdm
from pathlib import Path
import requests
import shutil


def get_rvc_models(this_dir):
    rvc_models_base = this_dir / "voice2voice" / "rvc"
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


def find_rvc_model_by_name(this_dir, model_name):
    models = get_rvc_models(this_dir)
    for model in models:
        if model['model_name'] == model_name:
            # Check model_index key if it's exists
            model_index = model.get('index_path', None)
            return model["model_path"], model_index
    return None


def get_openvoice_refs(this_dir):
    openvoice_models_base = this_dir / "voice2voice" / "openvoice"

    # Define a list of audio file extensions you are interested in
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".aac"}

    audio_files = []

    # Iterate through the openvoice_models_base directory recursively
    for audio_file in openvoice_models_base.rglob('*'):
        # Check if the file has one of the audio extensions
        if audio_file.suffix in audio_extensions:
            # Add the name of the file to the list
            audio_files.append(audio_file.name)

    return audio_files


def find_openvoice_ref_by_name(this_dir, filename):
    openvoice_models_base = this_dir / "voice2voice" / "openvoice"
    models = get_openvoice_refs(this_dir)
    for model in models:
        if model == filename:
            return openvoice_models_base / model


def infer_rvc(pitch, index_rate, protect_voiceless, method, index_path, model_path, input_path, opt_path, filter_radius = 3, resemple_rate = 0, envelope_mix = 0.25):
    f0method = method
    device = "cuda:0"
    protect = protect_voiceless
    autotune_enable = False

    index_path = str(index_path)
    model_path = str(model_path)
    try:
        cmd = [
            'venv/rvc_venv/scripts/python', '-m', 'rvc_python',
            '--input', input_path,
            '--model', index_path,
            '--pitch', str(pitch),
            '--method', f0method,
            '--output', opt_path,
            '--index', model_path,
            '--index_rate', str(index_rate),
            '--device', device,
            '--protect', str(protect).lower(),
            '--filter_radius', str(filter_radius),
            '--resample_sr', str(resemple_rate),
            '--rms_mix_rate', str(envelope_mix)
        ]

        subprocess.run(cmd)
    except Exception as e:
        print(f"Error: {e}")
        return False

def infer_rvc_batch(model_name,pitch, index_rate, protect_voiceless, method, index_path, model_path, paths, opt_path,filter_radius=3,resemple_rate=0,envelope_mix=0.25):
    f0method = method
    device = "cuda:0"
    protect = protect_voiceless

    index_path = str(index_path)
    model_path = str(model_path)

    temp_dir = Path(os.getcwd()) / "temp" / "rvc"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    temp_dir.mkdir(exist_ok=True)
    new_paths = []
    for file in paths:
        # Copy in temp dir with same name and 
        new_path = temp_dir / Path(file).name
        shutil.copy(file, new_path)
        new_paths.append(new_path)

    try:
        cmd = [
            'venv/rvc_venv/scripts/python', '-m', 'rvc_python',
            '--dir', temp_dir,
            '--model', index_path,
            '--pitch', str(pitch),
            '--method', f0method,
            '--output', opt_path,
            '--index', model_path,
            '--index_rate', str(index_rate),
            '--device', device,
            '--protect', str(protect).lower(),
            '--filter_radius', str(filter_radius),
            '--resample_sr', str(resemple_rate),
            '--rms_mix_rate', str(envelope_mix)
        ]

        subprocess.run(cmd)

        # Add suffix to all inference files, can be in any formate
        # add _rvc and model_name var
        opt_path = Path(opt_path).parent
        for file in glob.glob(f"{opt_path}/*"):
            new_name = Path(file).stem + f"_rvc_{model_name}" + Path(file).suffix
            os.rename(file, opt_path / new_name)
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def infer_openvoice(input_path, ref_path, output_path):
    try:
        cmd = [
            'venv/rvc_venv/scripts/python', '-m', 'openvoice_cli', "single",
            '-i', input_path,
            '-r', ref_path,
            '-o', output_path,
            "-d", "cuda:0"
        ]

        subprocess.run(cmd)
    except Exception as e:
        print(f"Error: {e}")
        return False


# def translate_audio_file(input_file,translate_whisper_model,translate_audio_mode,translate_source_lang,translate_target_lang,translate_speaker_lang):
