

import numpy as np
import soundfile as sf

from scripts.modeldownloader import get_folder_names,get_folder_names_advanced,install_deepspeed_based_on_python_version
from scripts.tts_funcs import TTSWrapper
from scripts.funcs import save_audio_to_wav,resample_audio,move_and_rename_file

import os
import gradio as gr
from pathlib import Path
from loguru import logger

import uuid

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
USE_DEEPSPEED = os.getenv("DEEPSPEED") == 'true'
MODEL_VERSION = os.getenv("MODEL_VERSION","v2.0.2")
WHISPER_VERSION = os.getenv("WHISPER_VERSION","none")

if USE_DEEPSPEED:
  install_deepspeed_based_on_python_version()

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
# XTTS = None

# LOAD MODEL
logger.info(f"Start loading model {MODEL_VERSION}")
this_dir = Path(__file__).parent.resolve()
XTTS.load_model(this_dir) 

def reload_model(model):
    XTTS.unload_model()

    XTTS.model_version = model
    XTTS.load_model(this_dir)

    return model

def change_infer_type(infer_type):
    XTTS.unload_model()

    XTTS.model_source = infer_type
    XTTS.load_model(this_dir)

    return infer_type

def change_language(languages):
    lang_code = reversed_supported_languages[languages]
    XTTS.language = lang_code
    return languages


def change_current_speaker(ref_speaker_list,speaker_value_text):
    XTTS.speaker_wav = ref_speaker_list

    speaker_value_text = ref_speaker_list

    return ref_speaker_list,speaker_value_text

def clear_current_speaker_audio(ref_speaker_audio,speaker_path_text,ref_speaker_list,speaker_value_text):
    speakers_list = XTTS.get_speakers()
    speaker_value = ""
    if not speakers_list:
        speakers_list = ["None"]
        speaker_value = "None"
    else:
        speaker_value = speakers_list[0]
        speaker_value_text = speaker_value

    return "",speaker_value,gr.Dropdown(label="Reference Speaker from folder 'speakers'",value=speaker_value,choices=speakers_list)

def change_current_speaker_audio(auto_cut,ref_speaker_audio,speaker_path_text,ref_speaker_list,speaker_value_text,use_resample):
#  Get audio, save it to tempo and resample it.
   output_path = ""
   rate, y = ref_speaker_audio
   output_path = save_audio_to_wav(rate, y,this_dir,auto_cut)
   if use_resample:
     output_path = resample_audio(output_path,this_dir)  

#  Assign the invisible variables the values we will use for generation
   XTTS.speaker_wav = output_path
   speaker_path_text = output_path

 # Get the list of speakers to assign a reference and select it in the list
 #  Add a reference item that will operate on the example given by the user
   speakers_list = XTTS.get_speakers()
   speaker_value_text = "reference"

   speakers_list.append(speaker_value_text)

   return output_path,speaker_path_text, "reference", gr.Dropdown(label="Reference Speaker from folder 'speakers'",value="reference",choices=speakers_list)

def reload_list(model):
    models_list = get_folder_names_advanced(this_dir / "models")
    return gr.Dropdown(
                    label="XTTS model",
                    value=model,
                    choices=models_list,
                    elem_classes="model-choose__checkbox"
                ) 

def update_speakers_list(speakers_list,speaker_value_text,speaker_path_text):
    speakers_list = XTTS.get_speakers()
    speaker_value = ""
    if not speakers_list:
        speakers_list = ["None"]
        speaker_value = "None"
    else:
        speaker_value = speakers_list[0]
        speaker_value_text = speaker_value

    if speaker_path_text:
        speakers_list.append("reference")

    return gr.Dropdown(label="Reference Speaker from folder 'speakers'",value=speaker_value,choices=speakers_list),speaker_value_text


def save_speaker(speaker_wav_save_name,speaker_path_text,ref_speaker_list):
    speakers_list = XTTS.get_speakers()
    speaker_value = speaker_wav_save_name
    speaker_value_text = speaker_wav_save_name
    speakers_list.append(speaker_wav_save_name)
    move_and_rename_file(speaker_path_text, XTTS.speaker_folder,speaker_wav_save_name)
    # INPUT ref_speaker,speaker_wav_save_name,speaker_path_text,ref_speaker_list
    # OUTPUT ref_speaker,speaker_wav_save_name,,speaker_path_text,speaker_value_text,ref_speaker_list
    return None,"","",speaker_wav_save_name,gr.Dropdown(label="Reference Speaker from folder 'speakers'",value=speaker_value,choices=speakers_list)


def generate_audio(text,languages,speaker_value_text,speaker_path_text,additional_text,temperature,length_penalty,repetition_penalty,top_k,top_p,speed,sentence_split):

    ref_speaker_wav = ""

    if speaker_path_text and speaker_value_text == "reference":
        ref_speaker_wav = speaker_path_text 
    else:
        ref_speaker_wav = speaker_value_text 

    lang_code = reversed_supported_languages[languages]
    options = {
        "temperature": temperature,
        "length_penalty": length_penalty,
        "repetition_penalty": float(repetition_penalty),
        "top_k": top_k,
        "top_p": top_p,
        "speed": speed,
        "sentence_split": sentence_split,
    }

    # Check if the file already exists, if yes, add a number to the filename
    count = 1
    output_file_path = f"{additional_text}_({count})_{speaker_value_text}.wav"
    while os.path.exists(os.path.join('output', output_file_path)):
        count += 1
        output_file_path = f"{additional_text}_({count})_{speaker_value_text}.wav"
    
    # Perform TTS and save to the generated filename
    output_file = XTTS.process_tts_to_file(text, lang_code, ref_speaker_wav, options, output_file_path)
    return gr.make_waveform(audio=output_file),output_file
    
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

        model.change(fn=reload_model, inputs=[model], outputs=[model])
        refresh_model_btn.click(fn=reload_list, inputs=[model],outputs=[model])

    with gr.Tab("Text2Voice"):
        # with gr.Column():
        with gr.Row():
            with gr.Column():
                text = gr.TextArea(label="Input Textt",placeholder="Input Text Here...")
                languages = gr.Dropdown(label="Language",choices=reversed_supported_languages_list,value="English")

                infer_type = gr.Radio(["api", "local"],value="local", label="Infer type", info="Affects how the voice generation will be played back")
                with gr.Accordion("Advanced settings (Works only in local infer type)", open=False) as acr:
                        speed = gr.Slider(
                                label="speed",
                                minimum=0.1,
                                maximum=2,
                                step=0.05,
                                value=1,
                            )
                        temperature = gr.Slider(
                            label="temperature",
                            minimum=0.01,
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
                            minimum=0.01,
                            maximum=1,
                            step=0.05,
                            value=0.85,
                        )
                        sentence_split = gr.Checkbox(
                            label="Enable text splitting",
                            value=True,
                        )
                
                speakers_list = XTTS.get_speakers()
                # speakers_list = ["Popa"]
                speaker_value = ""
                if not speakers_list:
                    speakers_list = ["None"]
                    speaker_value = "None"
                else:
                    speaker_value = speakers_list[0]
                    XTTS.speaker_wav = speaker_value

                # Variables
                speaker_value_text = gr.Textbox(label="Reference Speaker Name",value=speaker_value,visible=False)
                speaker_path_text = gr.Textbox(label="Reference Speaker Path",value="",visible=False)
                speaker_wav_modifyed = gr.Checkbox("Reference Audio",visible=False, value = False )

                with gr.Row():
                  ref_speaker_list = gr.Dropdown(label="Reference Speaker from folder 'speakers'",value=speaker_value,choices=speakers_list)
                  update_ref_speaker_list_btn = gr.Button(value="Update",elem_classes="speaker-update__btn")

                ref_speaker = gr.Audio(label="Reference Speaker (mp3, wav, flac)")

                with gr.Accordion(label="Reference Speaker settings",open=True):
                  with gr.Row():
                     use_resample = gr.Checkbox(label="Resample reference audio to 24000Hz")
                  auto_cut = gr.Slider(
                            label="Automatically trim audio up to x seconds, 0 without trimming ",
                            minimum=0,
                            maximum=30,
                            step=1,
                            value=0,
                        )
                  gr.Markdown(value="You can save the downloaded recording or microphone recording to a shared list, you need to set a name and click save")
                  speaker_wav_save_name = gr.Textbox(label="Speaker save name",value="new_speaker_name")
                  save_speaker_btn = gr.Button(value="Save Speaker")

                infer_type.input(fn=change_infer_type, inputs=[infer_type],outputs=[infer_type])

                ref_speaker_list.change(fn=change_current_speaker, inputs=[ref_speaker_list,speaker_value_text],outputs=[ref_speaker_list,speaker_value_text])
                update_ref_speaker_list_btn.click(fn=update_speakers_list, inputs=[ref_speaker_list,speaker_value_text,speaker_path_text],outputs=[ref_speaker_list,speaker_value_text])

                ref_speaker.stop_recording(fn=change_current_speaker_audio, inputs=[auto_cut,ref_speaker,speaker_path_text,speaker_value_text,ref_speaker_list,use_resample],outputs=[ref_speaker,speaker_path_text,speaker_value_text,ref_speaker_list])
                ref_speaker.upload(fn=change_current_speaker_audio, inputs=[auto_cut,ref_speaker,speaker_path_text,speaker_value_text,ref_speaker_list,use_resample],outputs=[ref_speaker,speaker_path_text,speaker_value_text,ref_speaker_list])
                ref_speaker.clear(fn=clear_current_speaker_audio,inputs=[ref_speaker,speaker_path_text,speaker_value_text,ref_speaker_list],outputs=[speaker_path_text,speaker_value_text,ref_speaker_list])
                # ref_speaker.change(fn=change_current_speaker_audio, inputs=[auto_cut,ref_speaker,speaker_path_text,speaker_value_text,ref_speaker_list,use_resample],outputs=[speaker_wav_modifyed,speaker_path_text,speaker_value_text,ref_speaker_list])

                languages.change(fn=change_language, inputs=[languages], outputs=[languages])
                save_speaker_btn.click(fn=save_speaker,inputs=[speaker_wav_save_name,speaker_path_text,ref_speaker_list],outputs=[ref_speaker,speaker_wav_save_name,speaker_path_text,speaker_value_text,ref_speaker_list])

            with gr.Column():
                video_gr = gr.Video(label="Waveform Visual",interactive=False)
                audio_gr = gr.Audio(label="Synthesised Audio",interactive=False, autoplay=False)
                generate_btn = gr.Button(value="Generate",size="lg",elem_classes="generate-btn")

                with gr.Accordion(label="Output settings",open=True):
                  additional_text_input = gr.Textbox(label="File Name Value", value="output")
                #   WIP
                #   output_format = gr.Radio(["mp3","wav"],value="wav", label="Output Format")
                #   improve_output_quality = gr.Checkbox(label="Improve output quality",value=False)

                generate_btn.click(
                    fn=generate_audio,
                    inputs=[text, languages, speaker_value_text, speaker_path_text, additional_text_input, temperature, length_penalty, repetition_penalty, top_k, top_p, speed, sentence_split],
                    outputs=[video_gr, audio_gr]
                )


# if __name__ == "__main__":
    # demo.launch()   
