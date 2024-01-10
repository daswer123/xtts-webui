import os
import sys
import tempfile
from pathlib import Path

import os
import shutil
import glob

import gradio as gr
import librosa.display
import numpy as np

import torch
import traceback
from scripts.utils.formatter import format_audio_list, find_latest_best_model
from scripts.utils.gpt_train import train_gpt

from xtts_webui import *


def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def preprocess_dataset(audio_path, language, whisper_model, out_path, train_status_bar):
    clear_gpu_cache()

    train_csv = ""
    eval_csv = ""

    out_path = os.path.join(out_path, "dataset")
    os.makedirs(out_path, exist_ok=True)
    if audio_path is None:
        return "You should provide one or multiple audio files! If you provided it, probably the upload of the files is not finished yet!", "", ""
    else:
        try:
            progress = gr.Progress(track_tqdm=True)
            train_meta, eval_meta, audio_total_size = format_audio_list(
                audio_path, whisper_model=whisper_model, target_language=language, out_path=out_path, gradio_progress=progress)
        except:
            traceback.print_exc()
            error = traceback.format_exc()
            return f"The data processing was interrupted due an error !! Please check the console to verify the full error message! \n Error summary: {error}", "", ""

    # clear_gpu_cache()

    # if audio total len is less than 2 minutes raise an error
    if audio_total_size < 120:
        message = "The sum of the duration of the audios that you provided should be at least 2 minutes!"
        print(message)
        return message, "", ""

    print("Dataset Processed!")
    return "Dataset Processed!", train_meta, eval_meta


def train_model(custom_model, version, language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
    clear_gpu_cache()

    print(output_path)
    run_dir = output_path / "run"

    # # Remove train dir
    # if run_dir.exists():
    #     os.remove(run_dir)

    # Check if the dataset language matches the language you specified
    lang_file_path = output_path / "dataset" / "lang.txt"

    # Check if lang.txt already exists and contains a different language
    current_language = None
    if lang_file_path.exists():
        with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
            current_language = existing_lang_file.read().strip()
            if current_language != language:
                print("The language that was prepared for the dataset does not match the specified language. Change the language to the one specified in the dataset")
                language = current_language

    if not train_csv or not eval_csv:
        return "You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields !", "", "", "", ""
    try:
        # convert seconds to waveform frames
        max_audio_length = int(max_audio_length * 22050)
        speaker_xtts_path, config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(
            custom_model, version, language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path=output_path, max_audio_length=max_audio_length)
    except:
        traceback.print_exc()
        error = traceback.format_exc()
        return f"The training was interrupted due an error !! Please check the console to check the full error message! \n Error summary: {error}", "", "", "", ""

    # copy original files to avoid parameters changes issues
    # os.system(f"cp {config_path} {exp_path}")
    # os.system(f"cp {vocab_file} {exp_path}")

    ready_dir = Path(output_path) / "ready"

    ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")

    shutil.copy(ft_xtts_checkpoint, ready_dir / "unoptimize_model.pth")
    # os.remove(ft_xtts_checkpoint)

    ft_xtts_checkpoint = os.path.join(ready_dir, "unoptimize_model.pth")

    # Reference
    # Move reference audio to output folder and rename it
    speaker_reference_path = Path(speaker_wav)
    speaker_reference_new_path = ready_dir / "reference.wav"
    shutil.copy(speaker_reference_path, speaker_reference_new_path)

    print("Model training done!")
    # clear_gpu_cache()
    return "Model training done!", config_path, vocab_file, ft_xtts_checkpoint, speaker_xtts_path, speaker_reference_new_path


def optimize_model(out_path, clear_train_data):
    # print(out_path)
    out_path = Path(out_path)  # Ensure that out_path is a Path object.

    ready_dir = out_path / "ready"
    run_dir = out_path / "run"
    dataset_dir = out_path / "dataset"

    # Clear specified training data directories.
    if clear_train_data in {"run", "all"} and run_dir.exists():
        try:
            shutil.rmtree(run_dir)
        except PermissionError as e:
            print(f"An error occurred while deleting {run_dir}: {e}")

    if clear_train_data in {"dataset", "all"} and dataset_dir.exists():
        try:
            shutil.rmtree(dataset_dir)
        except PermissionError as e:
            print(f"An error occurred while deleting {dataset_dir}: {e}")

    # Get full path to model
    model_path = ready_dir / "unoptimize_model.pth"

    if not model_path.is_file():
        return "Unoptimized model not found in ready folder", ""

    # Load the checkpoint and remove unnecessary parts.
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    del checkpoint["optimizer"]

    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]

    # Make sure out_path is a Path object or convert it to Path
    os.remove(model_path)

    # Save the optimized model.
    optimized_model_file_name = "model.pth"
    optimized_model = ready_dir/optimized_model_file_name

    torch.save(checkpoint, optimized_model)
    ft_xtts_checkpoint = str(optimized_model)

    clear_gpu_cache()

    return f"Model optimized and saved at {ft_xtts_checkpoint}!", ft_xtts_checkpoint


def load_params(model_name):
    path_output = Path("./finetuned_models") / model_name

    dataset_path = path_output / "dataset"

    if not dataset_path.exists():
        return "The output folder does not exist!", "", "", "en"

    eval_train = dataset_path / "metadata_train.csv"
    eval_csv = dataset_path / "metadata_eval.csv"

    # Write the target language to lang.txt in the output directory
    lang_file_path = dataset_path / "lang.txt"

    # Check if lang.txt already exists and contains a different language
    current_language = None
    if os.path.exists(lang_file_path):
        with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
            current_language = existing_lang_file.read().strip()

    clear_gpu_cache()

    print(current_language)
    return "The data has been updated", eval_train, eval_csv, current_language


def copy_model_directory(src, dst):
    shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
    print("Model training and optimization done! Model moved into models folder.")


def train_xtts_model(
    custom_model_name,
    upload_train_file,
    train_whisper_model,
    train_lang,
    train_version,
    train_csv,
    eval_csv,
    train_custom_model,
    num_epochs,
    batch_size,
    grad_acumm,
    max_audio_length,
    clear_train_data,
    train_status_bar
):
    XTTS.unload_model()

    output_train_folder = Path("./finetuned_models") / custom_model_name
    output_train_folder.mkdir(parents=True, exist_ok=True)
    train_msg = ""

    if (train_csv and eval_csv):
        print("Train csv and eval csv already exists")
    else:
        train_msg, train_csv, eval_csv = preprocess_dataset(
            upload_train_file, train_lang, train_whisper_model, output_train_folder, train_status_bar)

    # print(train_csv,eval_csv)

    # Train Models
    train_model(train_custom_model, train_version, train_lang, train_csv, eval_csv,
                num_epochs, batch_size, grad_acumm, output_train_folder, max_audio_length)

    # Optimize and shrink models
    optimize_model(output_train_folder, "none")

    # Move model to models folder
    ready_folder = output_train_folder / "ready"
    model_destination = f"models/{custom_model_name}"
    shutil.copytree(str(ready_folder), str(
        model_destination), dirs_exist_ok=True)

    # Move reference to speaker list
    speaker_reference_path = ready_folder / "reference.wav"
    reference_destination = f"speakers/{custom_model_name}.wav"
    shutil.copy(speaker_reference_path, reference_destination)

    train_msg = "Model training done! Model moved into models folder"

    return train_msg


load_params_btn.click(fn=load_params, inputs=[custom_model_name], outputs=[
                      train_status_bar, train_csv, eval_csv, train_lang])

train_btn.click(fn=train_xtts_model, inputs=[
    custom_model_name,
    upload_train_file,
    train_whisper_model,
    train_lang,
    train_version,
    train_csv,
    eval_csv,
    train_custom_model,
    num_epochs,
    batch_size,
    grad_acumm,
    max_audio_length,
    clear_train_data,
    train_status_bar
],
    outputs=[train_status_bar]
)
