import argparse
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
import torchaudio
import traceback
from scripts.utils.formatter import format_audio_list,find_latest_best_model
from scripts.utils.gpt_train import train_gpt

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Clear logs
def remove_log_file(file_path):
     log_file = Path(file_path)

     if log_file.exists() and log_file.is_file():
         log_file.unlink()

# remove_log_file(str(Path.cwd() / "log.out"))

def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

XTTS_MODEL = None
def load_model(xtts_checkpoint, xtts_config, xtts_vocab,xtts_speaker):
    global XTTS_MODEL
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "You need to run the previous steps or manually set the `XTTS checkpoint path`, `XTTS config path`, and `XTTS vocab path` fields !!"
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("Loading XTTS model! ")
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab,speaker_file_path=xtts_speaker, use_deepspeed=False)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("Model Loaded!")
    return "Model Loaded!"

def run_tts(lang, tts_text, speaker_audio_file, temperature, length_penalty,repetition_penalty,top_k,top_p,sentence_split,use_config):
    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to run the previous step to load the model !!", None, None



    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(audio_path=speaker_audio_file, gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, max_ref_length=XTTS_MODEL.config.max_ref_len, sound_norm_refs=XTTS_MODEL.config.sound_norm_refs)
    
    if use_config:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=XTTS_MODEL.config.temperature, # Add custom parameters here
            length_penalty=XTTS_MODEL.config.length_penalty,
            repetition_penalty=XTTS_MODEL.config.repetition_penalty,
            top_k=XTTS_MODEL.config.top_k,
            top_p=XTTS_MODEL.config.top_p,
            enable_text_splitting = True
        )
    else:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature, # Add custom parameters here
            length_penalty=length_penalty,
            repetition_penalty=float(repetition_penalty),
            top_k=top_k,
            top_p=top_p,
            enable_text_splitting = sentence_split
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    return "Speech generated !", out_path, speaker_audio_file


def load_params_tts(out_path,version):
    
    out_path = Path(out_path)

    # base_model_path = Path.cwd() / "models" / version 

    # if not base_model_path.exists():
    #     return "Base model not found !","","",""

    ready_model_path = out_path / "ready" 

    vocab_path =  ready_model_path / "vocab.json"
    config_path = ready_model_path / "config.json"
    speaker_path =  ready_model_path / "speakers_xtts.pth"
    reference_path  = ready_model_path / "reference.wav"

    model_path = ready_model_path / "model.pth"

    if not model_path.exists():
        model_path = ready_model_path / "unoptimize_model.pth"
        if not model_path.exists():
          return "Params for TTS not found", "", "", ""         

    return "Params for TTS loaded", model_path, config_path, vocab_path,speaker_path, reference_path
     

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""XTTS fine-tuning demo\n\n"""
        """
        Example runs:
        python3 TTS/demos/xtts_ft_demo/xtts_demo.py --port 
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to run the gradio demo. Default: 8011",
        default=8011,
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="Output path (where data and checkpoints will be saved) Default: output/",
        default=str(Path.cwd() / "finetune_models"),
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs to train. Default: 6",
        default=6,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size. Default: 2",
        default=2,
    )
    parser.add_argument(
        "--grad_acumm",
        type=int,
        help="Grad accumulation steps. Default: 1",
        default=1,
    )
    parser.add_argument(
        "--max_audio_length",
        type=int,
        help="Max permitted audio size in seconds. Default: 11",
        default=11,
    )

    args = parser.parse_args()

    with gr.Blocks() as demo:
        with gr.Tab("1 - Data processing"):
            out_path = gr.Textbox(
                label="Output path (where data and checkpoints will be saved):",
                value=args.out_path,
            )
            # upload_file = gr.Audio(
            #     sources="upload",
            #     label="Select here the audio files that you want to use for XTTS trainining !",
            #     type="filepath",
            # )
            upload_file = gr.File(
                file_count="multiple",
                label="Select here the audio files that you want to use for XTTS trainining (Supported formats: wav, mp3, and flac)",
            )

            whisper_model = gr.Dropdown(
                label="Whisper Model",
                value="large-v3",
                choices=[
                    "large-v3",
                    "large-v2",
                    "large",
                    "medium",
                    "small"
                ],
            )

            lang = gr.Dropdown(
                label="Dataset Language",
                value="en",
                choices=[
                    "en",
                    "es",
                    "fr",
                    "de",
                    "it",
                    "pt",
                    "pl",
                    "tr",
                    "ru",
                    "nl",
                    "cs",
                    "ar",
                    "zh",
                    "hu",
                    "ko",
                    "ja"
                ],
            )
            progress_data = gr.Label(
                label="Progress:"
            )
            # demo.load(read_logs, None, logs, every=1)

            prompt_compute_btn = gr.Button(value="Step 1 - Create dataset")
        
            def preprocess_dataset(audio_path, language, whisper_model, out_path,progress=gr.Progress(track_tqdm=True)):
                clear_gpu_cache()

                train_csv = ""
                eval_csv = ""

                out_path = os.path.join(out_path, "dataset")
                os.makedirs(out_path, exist_ok=True)
                if audio_path is None:
                    return "You should provide one or multiple audio files! If you provided it, probably the upload of the files is not finished yet!", "", ""
                else:
                    try:
                        train_meta, eval_meta, audio_total_size = format_audio_list(audio_path, whisper_model = whisper_model, target_language=language, out_path=out_path, gradio_progress=progress)
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

        with gr.Tab("2 - Fine-tuning XTTS Encoder"):
            load_params_btn = gr.Button(value="Load Params from output folder")
            version = gr.Dropdown(
                label="XTTS base version",
                value="v2.0.2",
                choices=[
                    "v2.0.3",
                    "v2.0.2",
                    "v2.0.1",
                    "v2.0.0",
                    "main"
                ],
            )
            train_csv = gr.Textbox(
                label="Train CSV:",
            )
            eval_csv = gr.Textbox(
                label="Eval CSV:",
            )
            custom_model = gr.Textbox(
                label="(Optional) Custom model.pth file , leave blank if you want to use the base file.",
                value="",
            )
            num_epochs =  gr.Slider(
                label="Number of epochs:",
                minimum=1,
                maximum=100,
                step=1,
                value=args.num_epochs,
            )
            batch_size = gr.Slider(
                label="Batch size:",
                minimum=1,
                maximum=16,
                step=1,
                value=args.batch_size,
            )
            grad_acumm = gr.Slider(
                label="Grad accumulation steps:",
                minimum=2,
                maximum=128,
                step=1,
                value=args.grad_acumm,
            )
            max_audio_length = gr.Slider(
                label="Max permitted audio size in seconds:",
                minimum=2,
                maximum=20,
                step=1,
                value=args.max_audio_length,
            )
            clear_train_data = gr.Dropdown(
                label="Clear train data, you will delete selected folder, after optimizing",
                value="run",
                choices=[
                    "none",
                    "run",
                    "dataset",
                    "all"
                ])
            
            progress_train = gr.Label(
                label="Progress:"
            )

            # demo.load(read_logs, None, logs_tts_train, every=1)
            train_btn = gr.Button(value="Step 2 - Run the training")
            optimize_model_btn = gr.Button(value="Step 2.5 - Optimize the model")
            
            def train_model(custom_model,version,language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
                clear_gpu_cache()

                run_dir = Path(output_path) / "run"

                # # Remove train dir
                if run_dir.exists():
                    os.remove(run_dir)
                
                # Check if the dataset language matches the language you specified 
                lang_file_path = Path(output_path) / "dataset" / "lang.txt"

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
                    speaker_xtts_path,config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(custom_model,version,language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path=output_path, max_audio_length=max_audio_length)
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
                return "Model training done!", config_path, vocab_file, ft_xtts_checkpoint,speaker_xtts_path, speaker_reference_new_path

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
                optimized_model_file_name="model.pth"
                optimized_model=ready_dir/optimized_model_file_name
            
                torch.save(checkpoint, optimized_model)
                ft_xtts_checkpoint=str(optimized_model)

                clear_gpu_cache()
        
                return f"Model optimized and saved at {ft_xtts_checkpoint}!", ft_xtts_checkpoint

            def load_params(out_path):
                path_output = Path(out_path)
                
                dataset_path = path_output / "dataset"

                if not dataset_path.exists():
                    return "The output folder does not exist!", "", ""

                eval_train = dataset_path / "metadata_train.csv"
                eval_csv = dataset_path / "metadata_eval.csv"

                # Write the target language to lang.txt in the output directory
                lang_file_path =  dataset_path / "lang.txt"

                # Check if lang.txt already exists and contains a different language
                current_language = None
                if os.path.exists(lang_file_path):
                    with open(lang_file_path, 'r', encoding='utf-8') as existing_lang_file:
                        current_language = existing_lang_file.read().strip()

                clear_gpu_cache()

                print(current_language)
                return "The data has been updated", eval_train, eval_csv, current_language

        with gr.Tab("3 - Inference"):
            with gr.Row():
                with gr.Column() as col1:
                    load_params_tts_btn = gr.Button(value="Load params for TTS from output folder")
                    xtts_checkpoint = gr.Textbox(
                        label="XTTS checkpoint path:",
                        value="",
                    )
                    xtts_config = gr.Textbox(
                        label="XTTS config path:",
                        value="",
                    )

                    xtts_vocab = gr.Textbox(
                        label="XTTS vocab path:",
                        value="",
                    )
                    xtts_speaker = gr.Textbox(
                        label="XTTS speaker path:",
                        value="",
                    )
                    progress_load = gr.Label(
                        label="Progress:"
                    )
                    load_btn = gr.Button(value="Step 3 - Load Fine-tuned XTTS model")

                with gr.Column() as col2:
                    speaker_reference_audio = gr.Textbox(
                        label="Speaker reference audio:",
                        value="",
                    )
                    tts_language = gr.Dropdown(
                        label="Language",
                        value="en",
                        choices=[
                            "en",
                            "es",
                            "fr",
                            "de",
                            "it",
                            "pt",
                            "pl",
                            "tr",
                            "ru",
                            "nl",
                            "cs",
                            "ar",
                            "zh",
                            "hu",
                            "ko",
                            "ja",
                        ]
                    )
                    tts_text = gr.Textbox(
                        label="Input Text.",
                        value="This model sounds really good and above all, it's reasonably fast.",
                    )
                    with gr.Accordion("Advanced settings", open=False) as acr:
                        temperature = gr.Slider(
                            label="temperature",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=0.75,
                        )
                        length_penalty  = gr.Slider(
                            label="length_penalty",
                            minimum=-10.0,
                            maximum=10.0,
                            step=0.5,
                            value=1,
                        )
                        repetition_penalty = gr.Slider(
                            label="repetition penalty",
                            minimum=1,
                            maximum=10,
                            step=0.5,
                            value=5,
                        )
                        top_k = gr.Slider(
                            label="top_k",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=50,
                        )
                        top_p = gr.Slider(
                            label="top_p",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=0.85,
                        )
                        sentence_split = gr.Checkbox(
                            label="Enable text splitting",
                            value=True,
                        )
                        use_config = gr.Checkbox(
                            label="Use Inference settings from config, if disabled use the settings above",
                            value=False,
                        )
                    tts_btn = gr.Button(value="Step 4 - Inference")

                with gr.Column() as col3:
                    progress_gen = gr.Label(
                        label="Progress:"
                    )
                    tts_output_audio = gr.Audio(label="Generated Audio.")
                    reference_audio = gr.Audio(label="Reference audio used.")

            prompt_compute_btn.click(
                fn=preprocess_dataset,
                inputs=[
                    upload_file,
                    lang,
                    whisper_model,
                    out_path,
                    train_csv,
                    eval_csv
                ],
                outputs=[
                    progress_data,
                    train_csv,
                    eval_csv,
                ],
            )

            load_params_btn.click(
                fn=load_params,
                inputs=[out_path],
                outputs=[
                    progress_train,
                    train_csv,
                    eval_csv,
                    lang
                ]
            )


            train_btn.click(
                fn=train_model,
                inputs=[
                    custom_model,
                    version,
                    lang,
                    train_csv,
                    eval_csv,
                    num_epochs,
                    batch_size,
                    grad_acumm,
                    out_path,
                    max_audio_length,
                ],
                outputs=[progress_train, xtts_config, xtts_vocab, xtts_checkpoint,xtts_speaker, speaker_reference_audio],
            )

            optimize_model_btn.click(
                fn=optimize_model,
                inputs=[
                    out_path,
                    clear_train_data
                ],
                outputs=[progress_train,xtts_checkpoint],
            )
            
            load_btn.click(
                fn=load_model,
                inputs=[
                    xtts_checkpoint,
                    xtts_config,
                    xtts_vocab,
                    xtts_speaker
                ],
                outputs=[progress_load],
            )

            tts_btn.click(
                fn=run_tts,
                inputs=[
                    tts_language,
                    tts_text,
                    speaker_reference_audio,
                    temperature,
                    length_penalty,
                    repetition_penalty,
                    top_k,
                    top_p,
                    sentence_split,
                    use_config
                ],
                outputs=[progress_gen, tts_output_audio,reference_audio],
            )

            load_params_tts_btn.click(
                fn=load_params_tts,
                inputs=[
                    out_path,
                    version
                    ],
                outputs=[progress_load,xtts_checkpoint,xtts_config,xtts_vocab,xtts_speaker,speaker_reference_audio],
            )

    demo.launch(
        share=False,
        debug=False,
        server_port=args.port,
        server_name="localhost",
        inbrowser=True
    )
