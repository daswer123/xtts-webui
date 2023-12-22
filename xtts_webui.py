

import numpy as np
import soundfile as sf

from scripts.modeldownloader import get_folder_names,get_folder_names_advanced,install_deepspeed_based_on_python_version
from scripts.tts_funcs import TTSWrapper
from scripts.funcs import save_audio_to_wav,resample_audio,move_and_rename_file,improve_and_convert_audio,improve_ref_audio,resemble_enchance_audio,cut_audio,str_to_list

import os
import gradio as gr
import langid
from pathlib import Path
from loguru import logger

import shutil

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

def predict_lang(text,selected_lang):
    language_predicted = langid.classify(text)[0].strip()  # strip need as there is space at end!

    # tts expects chinese as zh-cn
    if language_predicted == "zh":
            # we use zh-cn
            language_predicted = "zh-cn"
    
    # Check if language in supported langs
    if language_predicted not in supported_languages:
        language_predicted = selected_lang
        logger.warning(f"Language {language_predicted} not supported, using {supported_languages[selected_lang]}")

    return language_predicted

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


def change_current_speaker(ref_speaker_list,speaker_value_text,show_ref_speaker_from_list):
    XTTS.speaker_wav = ref_speaker_list
    speaker_value_text = ref_speaker_list

    if show_ref_speaker_from_list:
        speaker_path = XTTS.get_speaker_sample(ref_speaker_list)
        print(speaker_path)
        if speaker_path == None:
            return ref_speaker_list,speaker_value_text,gr.Audio(visible=False,value=None)
        return ref_speaker_list,speaker_value_text,gr.Audio(label="Speaker Example", value=speaker_path, visible=True)

    return ref_speaker_list,speaker_value_text,None

def clear_current_speaker_audio(ref_speaker_audio,speaker_path_text,ref_speaker_list,speaker_value_text,speaker_ref_wavs):
    speakers_list = XTTS.get_speakers()
    speaker_value = ""
    if not speakers_list:
        speakers_list = ["None"]
        speaker_value = "None"
    else:
        speaker_value = speakers_list[0]
        speaker_value_text = speaker_value

    if speaker_ref_wavs:
        speakers_list.append("multi_reference")
        speaker_value = "multi_reference"

    return "",speaker_value,gr.Dropdown(label="Reference Speaker from folder 'speakers'",value=speaker_value,choices=speakers_list), gr.Button(visible=False)

def change_current_speaker_audio(improve_reference_resemble,improve_reference_audio,auto_cut,ref_speaker_audio,speaker_path_text,ref_speaker_list,speaker_value_text,use_resample,speaker_ref_wavs):
#  Get audio, save it to tempo and resample it.
   output_path = ""
   rate, y = ref_speaker_audio
   output_path = save_audio_to_wav(rate, y,this_dir,auto_cut)

   if improve_reference_resemble:
     output_path = resemble_enchance_audio(output_path,True)
   if use_resample:
     output_path = resample_audio(output_path,this_dir)

   if improve_reference_audio:
     output_path =  improve_ref_audio(output_path,this_dir)

#  Assign the invisible variables the values we will use for generation
   XTTS.speaker_wav = output_path
   speaker_path_text = output_path

 # Get the list of speakers to assign a reference and select it in the list
 #  Add a reference item that will operate on the example given by the user
   speakers_list = XTTS.get_speakers()
   speaker_value_text = "reference"

   speakers_list.append(speaker_value_text)

   if speaker_ref_wavs:
        speakers_list.append("multi_reference")

   return output_path,speaker_path_text, "reference", gr.Dropdown(label="Reference Speaker from folder 'speakers'",value="reference",choices=speakers_list),gr.Button(value="Save a single sample for the speaker",visible=True)

def reload_list(model):
    models_list = get_folder_names_advanced(this_dir / "models")
    return gr.Dropdown(
                    label="XTTS model",
                    value=model,
                    choices=models_list,
                    elem_classes="model-choose__checkbox"
                ) 

def update_speakers_list(speakers_list,speaker_value_text,speaker_path_text,speaker_ref_wavs):
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
    
    if speaker_ref_wavs:
        speakers_list.append("multi_reference")

    return gr.Dropdown(label="Reference Speaker from folder 'speakers'",value=speaker_value,choices=speakers_list),speaker_value_text

def save_speaker(speaker_wav_save_name,speaker_path_text,ref_speaker_list,speaker_ref_wavs):
    move_and_rename_file(speaker_path_text, XTTS.speaker_folder,speaker_wav_save_name)

    speakers_list = XTTS.get_speakers()
    speaker_value = speaker_wav_save_name
    speaker_value_text = speaker_wav_save_name

    if speaker_ref_wavs:
        speakers_list.append("multi_reference")

    return None,"","",speaker_wav_save_name,gr.Dropdown(label="Reference Speaker from folder 'speakers'",value=speaker_value,choices=speakers_list,allow_custom_value=True),gr.Button(visible=False)

def save_speaker_multiple(speaker_wav_save_name,ref_speaker_list,ref_speakers,speaker_ref_wavs,speaker_path_text):
    new_ref_speaker_list = str_to_list(speaker_ref_wavs)
    speaker_dir = Path(XTTS.speaker_folder) / speaker_wav_save_name

    for file in new_ref_speaker_list:
        print(file)
        file = Path(file)
        move_and_rename_file(file, speaker_dir,os.path.basename(file))

    speakers_list = XTTS.get_speakers()
    speaker_value = speaker_wav_save_name
    speaker_value_text = speaker_wav_save_name

    if speaker_path_text:
        speakers_list.append("reference")

    return "",speaker_value,None,"",gr.Dropdown(label="Reference Speaker from folder 'speakers'",value=speaker_value,choices=speakers_list,allow_custom_value=True), gr.Button(visible=False)

def switch_waveform(enable_waveform,video_gr):
    if enable_waveform:
        return gr.Video(label="Waveform Visual",visible=True,interactive=False)
    else:
        return  gr.Video(label="Waveform Visual",interactive=False,visible=False)

def switch_speaker_example_visibility(ref_speaker_example,show_ref_speaker_from_list,ref_speaker_list):
    if show_ref_speaker_from_list:
        speaker_path = XTTS.get_speaker_sample(ref_speaker_list)
        if speaker_path == None:
            return gr.Audio(visible=False,value=None)
        return gr.Audio(label="Speaker Example", value=speaker_path, visible=True)
    else:
        return gr.Audio(label="Speaker Example", visible=False,value=None)

def create_multiple_reference(ref_speakers, use_resample=False, improve_reference_audio=False, auto_cut=0,improve_reference_resemble=False,speaker_value_text=""):
    ref_dir = Path(f'temp/reference_speaker_{uuid.uuid4()}')
    ref_dir.mkdir(parents=True, exist_ok=True)

    # Processing and moving new files.
    processed_files = []
    for index, speaker_file in enumerate(ref_speakers):
        current_file = Path(speaker_file)

        original_filename = current_file.stem  # Save the original file name without extension

        if auto_cut > 0:
            current_file = cut_audio(current_file, duration=auto_cut)

        if improve_reference_resemble:
            current_file = Path(resemble_enchance_audio(current_file,True))

        # Apply resampling if the use_resample flag is set.
        if use_resample:
            current_file = Path(resample_audio(current_file, this_dir=ref_dir))

        # Improve audio if the improve_reference_audio flag is set.
        if improve_reference_audio:
            current_file = Path(improve_ref_audio(current_file, this_dir=ref_dir))

        # The first file will be called preview.wav
        # new_location_name = "preview.wav" if index == 0 else f"{original_filename}.wav"
        new_location_name = f"{original_filename}.wav"
        new_location = ref_dir / new_location_name
        processed_files.append(new_location)
        try:
             # Move the prepared file to the target directory.
             shutil.move(str(current_file), str(new_location))
             print(f'Processed and moved {current_file} to {new_location}')
        except Exception as e:
             print(f'Failed to process or move {current_file}. Reason: {e}')
    print(f'Processed {len(processed_files)} files.')
    processed_files = [str(path) for path in processed_files]

    # Add ref to list and select it 
    speakers_list = XTTS.get_speakers()
    speaker_value = "multi_reference"
    speakers_list.append("multi_reference")

    if speaker_value_text:
        speakers_list.append("reference")

    return processed_files, processed_files,speaker_value, gr.Dropdown(label="Reference Speaker from folder 'speakers'",value=speaker_value,choices=speakers_list),gr.Button(value="Save multiple samples for the speaker",visible=True)
    
def clear_multiple_reference(ref_speaker):
    speakers_list = XTTS.get_speakers()
    speaker_value = ""
    if not speakers_list:
        speakers_list = ["None"]
        speaker_value = "None"
    else:
        speaker_value = speakers_list[0]
        speaker_value_text = speaker_value

    if ref_speaker:
        speakers_list.append("reference")
        speaker_value = "reference"

    return None, "",speaker_value, gr.Dropdown(label="Reference Speaker from folder 'speakers'",value=speaker_value,choices=speakers_list),gr.Button(visible=False)

def generate_audio(
    # Resemble enhance Settings
    enhance_resemble_chunk_seconds,
    enhance_resemble_chunk_overlap,
    enhance_resemble_solver,
    enhance_resemble_num_funcs,
    enhance_resemble_temperature,
    enhance_resemble_denoise,
    # Batch
    batch_generation,
    batch_generation_path,
    ref_speakers,
    # Features
    language_auto_detect,
    enable_waveform,
    improve_output_audio,
    improve_output_resemble,
    #  Default settings
    output_type,
    text,
    languages,
    # Help variables
    speaker_value_text,
    speaker_path_text,
    additional_text,
    # TTS settings
    temperature,length_penalty,
    repetition_penalty,
    top_k,
    top_p,
    speed,
    sentence_split):

    resemble_enhance_settings = {
        "chunk_seconds": enhance_resemble_chunk_seconds,
        "chunks_overlap": enhance_resemble_chunk_overlap,
        "solver": enhance_resemble_solver,
        "nfe": enhance_resemble_num_funcs,
        "tau": enhance_resemble_temperature,
        "denoising": enhance_resemble_denoise,
        "use_enhance": improve_output_resemble
    }

    ref_speaker_wav = ""

    print("Using ready reference")
    ref_speaker_wav = speaker_value_text

    if speaker_path_text and speaker_value_text == "reference":
        print("Using folder reference")
        ref_speaker_wav = speaker_path_text

    if ref_speakers and speaker_value_text == "multi_reference":
        print("Using multiple reference")
        ref_speakers_list = str_to_list(ref_speakers)
        ref_speaker_wav = Path(ref_speakers_list[0]).parent.absolute()
        ref_speaker_wav = str(ref_speaker_wav)
      
    lang_code = reversed_supported_languages[languages]

    if language_auto_detect:
        lang_code = predict_lang(text,lang_code)

    options = {
        "temperature": temperature,
        "length_penalty": length_penalty,
        "repetition_penalty": float(repetition_penalty),
        "top_k": top_k,
        "top_p": top_p,
        "speed": speed,
        "sentence_split": sentence_split,
    }
    # Find all .txt files in the filder and write this to batch_generation
    if batch_generation_path and Path(batch_generation_path).exists():
        batch_generation = [f for f in Path(batch_generation_path).glob('*.txt')]
        
    if batch_generation:
        for file_path in batch_generation:
            with open(file_path,encoding="utf-8",mode="r") as f:
                text = f.read()

                if language_auto_detect:
                    lang_code = predict_lang(text,lang_code)

                filename = os.path.basename(file_path)
                filename = filename.split(".")[0]

                output_file_path = f"{filename}_{additional_text}_{speaker_value_text}.{output_type}"
                output_file = XTTS.process_tts_to_file(text, lang_code, ref_speaker_wav, options, output_file_path)

                if improve_output_audio:
                   output_file = improve_and_convert_audio(output_file,output_type)

                if improve_output_resemble:
                    output_file = resemble_enchance_audio(**resemble_enhance_settings,audio_path=output_file,output_type=output_type)

        if enable_waveform:
            return gr.make_waveform(audio=output_file),output_file
        else:
            return None,output_file

    # Check if the file already exists, if yes, add a number to the filename
    count = 1
    output_file_path = f"{additional_text}_({count})_{speaker_value_text}.{output_type}"
    while os.path.exists(os.path.join('output', output_file_path)):
        count += 1
        output_file_path = f"{additional_text}_({count})_{speaker_value_text}.{output_type}"
    
    # Perform TTS and save to the generated filename
    output_file = XTTS.process_tts_to_file(text, lang_code, ref_speaker_wav, options, output_file_path)

    if improve_output_audio:
        output_file = improve_and_convert_audio(output_file,output_type)

    if improve_output_resemble:
        output_file = resemble_enchance_audio(**resemble_enhance_settings,audio_path=output_file,output_type=output_type)

    if enable_waveform:
        return gr.make_waveform(audio=output_file),output_file
    else:
        return None,output_file
    
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
                with gr.Tab("Text"):
                  text = gr.TextArea(label="Input Textt",placeholder="Input Text Here...")
                with gr.Tab("Batch"):
                  batch_generation = gr.Files(label="Upload .txt files",file_types=["text"]) 
                  batch_generation_path = gr.Textbox(label="Path to folder with .txt files, Has priority over all ",value="") 

                language_auto_detect = gr.Checkbox(label="Enable language auto detect",info="If your language is not supported or the text is less than 20 characters, this function will not work")
                languages = gr.Dropdown(label="Language",choices=reversed_supported_languages_list,value="English")
                speed = gr.Slider(
                                label="speed",
                                minimum=0.1,
                                maximum=2,
                                step=0.05,
                                value=1,
                            )
                with gr.Accordion("Advanced settings", open=False) as acr:
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

                        infer_type = gr.Radio(["api", "local"],value="local", label="Type of Processing",
                        info="Defines how the text will be processed,local gives you more options. Api does not allow you to use advanced settings")
                
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
                speaker_ref_wavs = gr.Text(visible=False)

                with gr.Row():
                  ref_speaker_list = gr.Dropdown(label="Reference Speaker from folder 'speakers'",value=speaker_value,choices=speakers_list)
                  show_ref_speaker_from_list = gr.Checkbox(value=False,label="Show example",info="This option will allow you to listen to your reference sample")
                  update_ref_speaker_list_btn = gr.Button(value="Update",elem_classes="speaker-update__btn")
                ref_speaker_example = gr.Audio(label="speaker example", sources="upload",visible=False,interactive=False)

                with gr.Tab(label="Single"):
                  ref_speaker = gr.Audio(label="Reference Speaker (mp3, wav, flac)",editable=False)
                with gr.Tab(label="Multiple"):
                  ref_speakers = gr.Files(label="Reference Speakers (mp3, wav, flac)",file_types=["audio"])

                with gr.Accordion(label="Reference Speaker settings.",open=True):
                  gr.Markdown(value="**Note: the settings only work when you enable them and upload files when they are enabled**")
                  gr.Markdown(value="Take a look at how to create good samples [here](https://github.com/daswer123/xtts-api-server?tab=readme-ov-file#note-on-creating-samples-for-quality-voice-cloning)")
                  with gr.Row():
                     use_resample = gr.Checkbox(label="Resample reference audio to 22050Hz",info="This is for better processing",value=True)
                     improve_reference_audio = gr.Checkbox(label="Clean up reference audio",info="Trim silence, use lowpass and highpass filters", value=False)
                     improve_reference_resemble = gr.Checkbox(label="Resemble enhancement (Uses extra 4GB VRAM)",info="You can find the settings next to the settings for the result",value=False)
                  auto_cut = gr.Slider(
                            label="Automatically trim audio up to x seconds, 0 without trimming ",
                            minimum=0,
                            maximum=30,
                            step=1,
                            value=0,
                        )
                  gr.Markdown(value="You can save the downloaded recording or microphone recording to a shared list, you need to set a name and click save")
                  speaker_wav_save_name = gr.Textbox(label="Speaker save name",value="new_speaker_name")
                  save_speaker_btn = gr.Button(value="Save a single sample for the speaker",visible=False)
                  save_multiple_speaker_btn = gr.Button(value="Save multiple samples for the speaker",visible=False)

                infer_type.input(fn=change_infer_type, inputs=[infer_type],outputs=[infer_type])

                ref_speaker_list.change(fn=change_current_speaker, inputs=[ref_speaker_list,speaker_value_text,show_ref_speaker_from_list],outputs=[ref_speaker_list,speaker_value_text,ref_speaker_example])
                update_ref_speaker_list_btn.click(fn=update_speakers_list, inputs=[ref_speaker_list,speaker_value_text,speaker_path_text,speaker_ref_wavs],outputs=[ref_speaker_list,speaker_value_text])
                show_ref_speaker_from_list.change(fn=switch_speaker_example_visibility,inputs=[ref_speaker_example,show_ref_speaker_from_list,ref_speaker_list],outputs=[ref_speaker_example])

                ref_speaker.stop_recording(fn=change_current_speaker_audio, inputs=[improve_reference_resemble,improve_reference_audio,auto_cut,ref_speaker,speaker_path_text,speaker_value_text,ref_speaker_list,use_resample,speaker_ref_wavs],outputs=[ref_speaker,speaker_path_text,speaker_value_text,ref_speaker_list,save_speaker_btn])
                ref_speaker.upload(fn=change_current_speaker_audio, inputs=[improve_reference_resemble,improve_reference_audio,auto_cut,ref_speaker,speaker_path_text,speaker_value_text,ref_speaker_list,use_resample,speaker_ref_wavs],outputs=[ref_speaker,speaker_path_text,speaker_value_text,ref_speaker_list,save_speaker_btn])
                ref_speaker.clear(fn=clear_current_speaker_audio,inputs=[ref_speaker,speaker_path_text,speaker_value_text,ref_speaker_list,speaker_ref_wavs],outputs=[speaker_path_text,speaker_value_text,ref_speaker_list,save_speaker_btn])
                # ref_speaker.change(fn=change_current_speaker_audio, inputs=[auto_cut,ref_speaker,speaker_path_text,speaker_value_text,ref_speaker_list,use_resample],outputs=[speaker_wav_modifyed,speaker_path_text,speaker_value_text,ref_speaker_list])

                ref_speakers.upload(fn=create_multiple_reference,inputs=[ref_speakers,use_resample,improve_reference_audio,auto_cut,improve_reference_resemble,speaker_value_text],outputs=[ref_speakers,speaker_ref_wavs,speaker_value_text,ref_speaker_list,save_multiple_speaker_btn])
                ref_speakers.clear(fn=clear_multiple_reference,inputs=[ref_speaker],outputs=[ref_speakers,speaker_ref_wavs,speaker_value_text,ref_speaker_list,save_multiple_speaker_btn])

                languages.change(fn=change_language, inputs=[languages], outputs=[languages])
                
                save_speaker_btn.click(fn=save_speaker,inputs=[speaker_wav_save_name,speaker_path_text,ref_speaker_list,speaker_ref_wavs],outputs=[ref_speaker,speaker_wav_save_name,speaker_path_text,speaker_value_text,ref_speaker_list,save_speaker_btn])
                save_multiple_speaker_btn.click(fn=save_speaker_multiple,inputs=[speaker_wav_save_name,ref_speaker_list,ref_speakers,speaker_ref_wavs,speaker_path_text],outputs=[speaker_wav_save_name,speaker_value_text,ref_speakers,speaker_ref_wavs,ref_speaker_list])
            
            with gr.Column():
                video_gr = gr.Video(label="Waveform Visual",visible=False,interactive=False)
                audio_gr = gr.Audio(label="Synthesised Audio",interactive=False, autoplay=False)
                generate_btn = gr.Button(value="Generate",size="lg",elem_classes="generate-btn")

                with gr.Accordion(label="Output settings",open=True):
                  with gr.Column():
                    with gr.Row():
                      enable_waveform = gr.Checkbox(label="Enable Waveform",value=False)
                      improve_output_audio = gr.Checkbox(label="Improve output quality (Reduces noise and makes audio slightly better)",value=False)
                      improve_output_resemble = gr.Checkbox(label="Resemble enhancement (Uses extra 4GB VRAM)",value=False)
                    with gr.Accordion(label="Resemble enhancement Settings",open=False):
                        enhance_resemble_chunk_seconds = gr.Slider(minimum=2, maximum=40, value=8, step=1, label="Chunk seconds (more secods more VRAM usage and faster inference speed)")
                        enhance_resemble_chunk_overlap = gr.Slider(minimum=0.1, maximum=2, value=1, step=0.2, label="Overlap seconds")
                        enhance_resemble_solver = gr.Dropdown(label="CFM ODE Solver (Midpoint is recommended)",choices=["Midpoint", "RK4", "Euler"], value="Midpoint")
                        enhance_resemble_num_funcs = gr.Slider(minimum=1, maximum=128, value=64, step=1, label="CFM Number of Function Evaluations (higher values in general yield better quality but may be slower)")
                        enhance_resemble_temperature = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="CFM Prior Temperature (higher values can improve quality but can reduce stability)")
                        enhance_resemble_denoise = gr.Checkbox(value=True, label="Denoise Before Enhancement (tick if your audio contains heavy background noise)")
                    with gr.Row():
                      output_type = gr.Radio(["mp3","wav"],value="wav", label="Output Type")
                  additional_text_input = gr.Textbox(label="File Name Value", value="output")

                generate_btn.click(
                    fn=generate_audio,
                    inputs=[
                        # Resemble enhance Settings
                        enhance_resemble_chunk_seconds,
                        enhance_resemble_chunk_overlap,
                        enhance_resemble_solver,
                        enhance_resemble_num_funcs,
                        enhance_resemble_temperature,
                        enhance_resemble_denoise,
                        # Batch
                        batch_generation,
                        batch_generation_path,
                        speaker_ref_wavs,
                        # Features
                        language_auto_detect,
                        enable_waveform,
                        improve_output_audio,
                        improve_output_resemble,
                        #  Default settings
                        output_type,
                        text,
                        languages,
                        # Help variables
                        speaker_value_text,
                        speaker_path_text,
                        additional_text_input,
                        # TTS settings
                        temperature,
                        length_penalty,
                        repetition_penalty,
                        top_k,
                        top_p,
                        speed,
                        sentence_split
                        ],
                    outputs=[video_gr, audio_gr]
                )


                enable_waveform.change(
                    fn=switch_waveform,
                    inputs=[enable_waveform,video_gr],
                    outputs=[video_gr]
                )


if __name__ == "__main__":
    demo.launch(inbrowser=True,share=True)   
