from scripts.modeldownloader import get_folder_names_advanced

from datetime import datetime
from scripts.funcs import resemble_enhance_audio,save_audio_to_wav

from scripts.tts_funcs import TTSWrapper

import os
import gradio as gr
from pathlib import Path
from loguru import logger
import glob

# Read css
css = os.path.join(os.path.dirname(__file__), "style.css")
with open(css) as f:
    css = f.read()

# Default Folders , you can change them via API
DEVICE = os.getenv('DEVICE',"cuda")
OUTPUT_FOLDER = os.getenv('OUTPUT', 'output')
SPEAKER_FOLDER = os.getenv('SPEAKER', 'speakers')
BASE_URL = os.getenv('BASE_URL', '127.0.0.1:8020')
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")
LOWVRAM_MODE = os.getenv("LOWVRAM_MODE") == 'true'
USE_DEEPSPEED = os.getenv("DEEPSPEED","true") == 'true'
MODEL_VERSION = os.getenv("MODEL_VERSION","v2.0.2")
WHISPER_VERSION = os.getenv("WHISPER_VERSION","none")

RVC_ENABLE = os.getenv("RVC_ENABLED") == 'true'

supported_languages = {
    "ar":"Arabic",
    "pt":"Brazilian Portuguese",
    "zh-cn":"Chinese",
    "cs":"Czech",
    "nl":"Dutch",
    "en":"English",
    "fr":"French",
    "de":"German",
    "it":"Italian",
    "pl":"Polish",
    "ru":"Russian",
    "es":"Spanish",
    "tr":"Turkish",
    "ja":"Japanese",
    "ko":"Korean",
    "hu":"Hungarian",
    "hi":"Hindi"
}

reversed_supported_languages = {name: code for code, name in supported_languages.items()}
reversed_supported_languages_list = list(reversed_supported_languages.keys())

# INIT MODEL
XTTS = TTSWrapper(OUTPUT_FOLDER,SPEAKER_FOLDER,LOWVRAM_MODE,MODEL_SOURCE,MODEL_VERSION,DEVICE)

# LOAD MODEL
logger.info(f"Start loading model {MODEL_VERSION}")
this_dir = Path(__file__).parent.resolve()
# XTTS.load_model(this_dir) 


def instrument_enchane_audio(
  resemble_audio_single,
  resemble_audio_batch,
  resemble_audio_batch_path,
  resemble_choose_action,
  resemble_chunk_seconds,
  resemble_chunk_overlap,
  resemble_solver,
  resemble_num_funcs,
  resemble_temperature,
  resemble_denoise,
  resemble_output_type,
  resemble_status_label
  ):

  print(
    "resemble_audio_single",resemble_audio_single,
    "\n\nresemble_audio_batch",resemble_audio_batch,
    "\n\nresemble_choose_action",resemble_choose_action,
    "\n\nresemble_audio_batch_path",resemble_audio_batch_path,
    "\n\nresemble_chunk_seconds",resemble_chunk_seconds,
    "\n\nresemble_chunk_overlap",resemble_chunk_overlap,
    "\n\nresemble_solver",resemble_solver,
    "\n\nresemble_num_funcs",resemble_num_funcs,
    "\n\nresemble_temperature",resemble_temperature,
    "\n\nresemble_denoise",resemble_denoise,
    "\n\nresemble_output_type",resemble_output_type,
    "\n\nresemble_status_label",resemble_status_label
  )

  resemble_status_label = gr.Progress(track_tqdm=True)

  output_folder = this_dir / OUTPUT_FOLDER
  folder_name = ""

  # Define folder name
  if resemble_choose_action == "both":
    # Current date 
    folder_name = "resemble_enhance_both"
  else:
    folder_name = f"resemble_enhance_{resemble_choose_action}"

  folder_name += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

  # Create output dir
  output_dir = output_folder / folder_name
  os.makedirs(output_dir, exist_ok=True)

  # Save Audio
  if resemble_audio_single is not None:
    rate, y = resemble_audio_single
    input_file = save_audio_to_wav(rate, y,this_dir)

  use_enhance = resemble_choose_action == "both" or resemble_choose_action == "only_enchance"
  use_denoise = resemble_choose_action == "both" or resemble_choose_action == "only_denoise"

  audio_files = resemble_audio_batch

  if resemble_audio_batch_path:
    audio_files = glob.glob(resemble_audio_batch_path + "/*.wav")
    audio_files += glob.glob(resemble_audio_batch_path + "/*.mp3")
    audio_files += glob.glob(resemble_audio_batch_path + "/*.flac")

  if resemble_audio_batch or resemble_audio_batch_path:
    if resemble_status_label is not None:
        tqdm_object = resemble_status_label.tqdm(audio_files, desc="Enchance files...")
    else:
        tqdm_object = tqdm(audio_files)


    for file in tqdm_object:
      output_audio = resemble_enhance_audio(
        audio_path = file,
        use_enhance = use_enhance ,
        use_denoise = use_denoise,
        solver = resemble_solver,
        nfe = resemble_num_funcs,
        tau = resemble_temperature,
        chunk_seconds = resemble_chunk_seconds,
        chunks_overlap = resemble_chunk_overlap,
        denoising = resemble_denoise,
        output_type = "wav",
        output_folder=folder_name
      )
  else:
    output_audio = resemble_enhance_audio(
      audio_path = input_file,
      use_enhance = use_enhance ,
      use_denoise = use_denoise,
      solver = resemble_solver,
      nfe = resemble_num_funcs,
      tau = resemble_temperature,
      chunk_seconds = resemble_chunk_seconds,
      chunks_overlap = resemble_chunk_overlap,
      denoising = resemble_denoise,
      output_type = "wav",
      output_folder=folder_name
    )

      # With glob collect audio files and create list with path to each file ( wav, mp3, flac)
      
  return gr.Audio(label="Denoise",visible=use_denoise,value=output_audio[0]),gr.Audio(label="Enhance",visible=use_enhance,value=output_audio[1]),"Done"
  # return result_denoise,result_enhance,resemble_status_label

with gr.Blocks(css=css) as demo:
    gr.Markdown(value="# XTTS-webui by [daswer123](https://github.com/daswer123)")
    with gr.Row(elem_classes="model-choose"):
        models_list = get_folder_names_advanced(this_dir / "models")
        model = gr.Dropdown(
                    label="Select XTTS model version",
                    value=MODEL_VERSION,
                    choices=models_list,
                    elem_classes="model-choose__checkbox"
                )
        refresh_model_btn = gr.Button(value="Update",elem_classes="model-choose__btn")
        
    with gr.Tab("Text2Voice"):
        from parts.text2voice import *
                    
    with gr.Tab("Voice2Voice"):
      with gr.Tab("RVC"):
        gr.Markdown("WIP")
      with gr.Tab("OpenVoice"):
        gr.Markdown("WIP")
      with gr.Tab("Translate"):
        gr.Markdown("WIP")

    with gr.Tab("Train"):
      gr.Markdown("WIP")
    
    with gr.Tab("Instuments"):
      with gr.Tab("Resemble Enhance"):
        with gr.Row():
          with gr.Column():
            with gr.Tab("Single"):
              resemble_audio_single= gr.Audio(label="Single file",value=None)
            with gr.Tab("Batch"):
              resemble_audio_batch = gr.File(file_count="multiple",label="Batch files",file_types=["audio"])
              resemble_audio_batch_path = gr.Textbox(label="Path to folder with audio files (High priority)",value=None)
            resemble_choose_action = gr.Radio(label="Choose action",choices=["only_enchance","only_denoise", "both"],value="both")
            resemble_chunk_seconds = gr.Slider(minimum=2, maximum=40, value=8, step=1, label="Chunk seconds (more secods more VRAM usage and faster inference speed)")
            resemble_chunk_overlap = gr.Slider(minimum=0.1, maximum=2, value=1, step=0.2, label="Overlap seconds")
            resemble_solver = gr.Dropdown(label="CFM ODE Solver (Midpoint is recommended)",choices=["Midpoint", "RK4", "Euler"], value="Midpoint")
            resemble_num_funcs = gr.Slider(minimum=1, maximum=128, value=64, step=1, label="CFM Number of Function Evaluations (higher values in general yield better quality but may be slower)")
            resemble_temperature = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="CFM Prior Temperature (higher values can improve quality but can reduce stability)")
            resemble_denoise = gr.Checkbox(value=True, label="Denoise Before Enhancement (tick if your audio contains heavy background noise)")
            resemble_output_type = gr.Dropdown(label="Output type",choices=["wav","mp3"],value="wav")
          with gr.Column():
            resemble_status_label = gr.Label(value="Upload file or files and click on Generate button")
            result_denoise = gr.Audio(label="Denoise",interactive=False,visible=False,value=None)
            result_enhance = gr.Audio(label="Result",interactive=False,visible=True,value=None)
            resemble_generate_btn = gr.Button(value="Enhance")

    
    resemble_generate_btn.click(fn=instrument_enchane_audio,
    inputs=[
      resemble_audio_single,
      resemble_audio_batch,
      resemble_audio_batch_path,
      resemble_choose_action,
      resemble_chunk_seconds,
      resemble_chunk_overlap,
      resemble_solver,
      resemble_num_funcs,
      resemble_temperature,
      resemble_denoise,
      resemble_output_type,
      resemble_status_label
      ],
    outputs=[result_denoise,result_enhance,resemble_status_label])  
        

    
    # LOAD FUNCTIONS AND HANDLERS
    import modules
if __name__ == "__main__":
    demo.queue()
    demo.launch(inbrowser=True,share=True)   
