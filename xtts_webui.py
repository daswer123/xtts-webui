from scripts.voice2voice import infer_rvc,get_openvoice_refs, infer_rvc_batch, infer_openvoice,find_openvoice_ref_by_name
from scripts.funcs import save_audio_to_wav
import shutil
from datetime import datetime
import glob
from scripts.modeldownloader import get_folder_names_advanced
from scripts.tts_funcs import TTSWrapper

import os
import gradio as gr
from pathlib import Path
from loguru import logger

# Read css
css = os.path.join(os.path.dirname(__file__), "style.css")
with open(css) as f:
    css = f.read()

# Default Folders , you can change them via API
DEVICE = os.getenv('DEVICE', "cuda")
OUTPUT_FOLDER = os.getenv('OUTPUT', 'output')
SPEAKER_FOLDER = os.getenv('SPEAKER', 'speakers')
BASE_URL = os.getenv('BASE_URL', '127.0.0.1:8020')
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")
LOWVRAM_MODE = os.getenv("LOWVRAM_MODE") == 'true'
USE_DEEPSPEED = os.getenv("DEEPSPEED", "true") == 'true'
MODEL_VERSION = os.getenv("MODEL_VERSION", "v2.0.2")
WHISPER_VERSION = os.getenv("WHISPER_VERSION", "none")

RVC_ENABLE = os.getenv("RVC_ENABLED") == 'true'

supported_languages = {
    "ar": "Arabic",
    "pt": "Brazilian Portuguese",
    "zh-cn": "Chinese",
    "cs": "Czech",
    "nl": "Dutch",
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pl": "Polish",
    "ru": "Russian",
    "es": "Spanish",
    "tr": "Turkish",
    "ja": "Japanese",
    "ko": "Korean",
    "hu": "Hungarian",
    "hi": "Hindi"
}

reversed_supported_languages = {
    name: code for code, name in supported_languages.items()}
reversed_supported_languages_list = list(reversed_supported_languages.keys())

# INIT MODEL
XTTS = TTSWrapper(OUTPUT_FOLDER, SPEAKER_FOLDER, LOWVRAM_MODE,
                  MODEL_SOURCE, MODEL_VERSION, DEVICE)

# LOAD MODEL
logger.info(f"Start loading model {MODEL_VERSION}")
this_dir = Path(__file__).parent.resolve()
# XTTS.load_model(this_dir)


def infer_openvoice_audio(openvoice_audio_single,openvoice_audio_batch,openvoice_audio_batch_path,openvoice_voice_ref_list,openvoice_status_bar,speaker_path_text):
    print("hello world")

    if openvoice_voice_ref_list == "None":
        return None, None, "Please select Refernce audio"

    if not openvoice_audio_single and not openvoice_audio_batch and not openvoice_audio_batch_path:
        return None, None, "Please load audio"

    openvoice_status_bar = gr.Progress(track_tqdm=True)

    output_folder = this_dir / OUTPUT_FOLDER
    folder_name = ""

    done_message = ""

    folder_name = "openvoice"
    folder_name += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save Audio
    if openvoice_audio_single is not None:
        rate, y = openvoice_audio_single
        input_file = save_audio_to_wav(rate, y, this_dir)

    audio_files = openvoice_audio_batch

    if openvoice_audio_batch_path:
        audio_files = glob.glob(openvoice_audio_batch_path + "/*.wav")
        audio_files += glob.glob(openvoice_audio_batch_path + "/*.mp3")
        audio_files += glob.glob(openvoice_audio_batch_path + "/*.flac")

    if openvoice_audio_batch or openvoice_audio_batch_path:
        output_folder = output_folder / folder_name
        os.makedirs(output_folder, exist_ok=True)

        output_audio = "OPENVOICE FUNC"
        tqdm_object = openvoice_status_bar.tqdm(audio_files, desc="Tuning Files...")

        for audio_file in tqdm_object:
            allow_infer = True

            # Initialize ref_path to None to ensure it has a value in all branches
            ref_voice_opvoice_path = None
    
            # Check if starts with "speaker/"
            if openvoice_voice_ref_list.startswith("speaker/"):
                speaker_wav = openvoice_voice_ref_list.split("/")[-1]
    
                if speaker_wav == "reference" and speaker_path_text:
                    ref_voice_opvoice_path = speaker_path_text
                else:
                    ref_voice_opvoice_path = XTTS.get_speaker_path(speaker_wav)
                    if type(ref_voice_opvoice_path) == list:
                        ref_voice_opvoice_path = ref_voice_opvoice_path[0]
    
                if speaker_wav == "reference" and not speaker_path_text:
                    allow_infer = False
                    print("Reference not found")
            else:   
                    
                    ref_voice_opvoice_path = find_openvoice_ref_by_name(
                        this_dir, openvoice_voice_ref_list)

            file_basename = Path(audio_file).stem
            output_filename = output_folder / f"openvoice_{file_basename}.wav"
            if allow_infer:
                infer_openvoice(input_path=audio_file,
                                ref_path=ref_voice_opvoice_path, output_path=output_filename)
        return None, None, f"Files saved in {output_folder} folder"
    else:
        # Func for single file rvc
        temp_dir = this_dir / "output"
        # result = temp_dir / f"{speaker_value_text}_tuned_{count}.{output_type}"
        allow_infer = True

        # Initialize ref_path to None to ensure it has a value in all branches
        ref_voice_opvoice_path = None

        # Check if starts with "speaker/"
        if openvoice_voice_ref_list.startswith("speaker/"):
            speaker_wav = openvoice_voice_ref_list.split("/")[-1]
            file_basename = speaker_wav

            if speaker_wav == "reference" and speaker_path_text:
                ref_voice_opvoice_path = speaker_path_text
            else:
                ref_voice_opvoice_path = XTTS.get_speaker_path(speaker_wav)
                if type(ref_voice_opvoice_path) == list:
                    ref_voice_opvoice_path = ref_voice_opvoice_path[0]

            if speaker_wav == "reference" and not speaker_path_text:
                allow_infer = False
                print("Reference not found")
        else:   
                file_basename = Path(openvoice_voice_ref_list).stem
                ref_voice_opvoice_path = find_openvoice_ref_by_name(
                    this_dir, openvoice_voice_ref_list)
        
        output_filename = output_folder / f"openvoice_{file_basename}_f{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        if allow_infer:
            infer_openvoice(input_path=input_file,
                            ref_path=ref_voice_opvoice_path, output_path=output_filename)

            # Update the output_file with the absolute path to the result
            output_file = "test.wav"

        output_audio = "OPENVOICE FUNC"
        done_message = f"Done"
        output_audio = "test.wav"

    return None, gr.Audio(label="Result", value=output_audio), done_message


def infer_rvc_audio(
        # INPUT
        rvc_audio_single,
        rvc_audio_batch,
        rvc_audio_batch_path,
        # PATH
        rvc_voice_settings_model_name,
        rvc_voice_settings_model_path,
        rvc_voice_settings_index_path,
        # SETTINGS
        rvc_voice_settings_pitch,
        rvc_voice_settings_index_rate,
        rvc_voice_settings_protect_voiceless,
        rvc_voice_settings_method,
        rvc_voice_filter_radius,
        rvc_voice_resemple_rate,
        rvc_voice_envelope_mix,
        # STATUS
        rvc_voice_status_bar

):
    if not rvc_voice_settings_model_name:
        return None, None, "Please select RVC model"

    if not rvc_audio_single and not rvc_audio_batch and not rvc_audio_batch_path:
        return None, None, "Please load audio"

    rvc_voice_status_bar = gr.Progress(track_tqdm=True)

    output_folder = this_dir / OUTPUT_FOLDER
    folder_name = ""

    done_message = ""

    folder_name = "rvc"
    folder_name += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save Audio
    if rvc_audio_single is not None:
        rate, y = rvc_audio_single
        input_file = save_audio_to_wav(rate, y, this_dir)

    audio_files = rvc_audio_batch

    if rvc_audio_batch_path:
        audio_files = glob.glob(rvc_audio_batch_path + "/*.wav")
        audio_files += glob.glob(rvc_audio_batch_path + "/*.mp3")
        audio_files += glob.glob(rvc_audio_batch_path + "/*.flac")

    if rvc_audio_batch or rvc_audio_batch_path:
        output_folder = output_folder / folder_name / "temp"
        os.makedirs(output_folder, exist_ok=True)

        output_audio = infer_rvc_batch(
            model_name=rvc_voice_settings_model_name,
            pitch=rvc_voice_settings_pitch,
            index_rate=rvc_voice_settings_index_rate,
            protect_voiceless=rvc_voice_settings_protect_voiceless,
            method=rvc_voice_settings_method,
            index_path=rvc_voice_settings_model_path,
            model_path=rvc_voice_settings_index_path,
            paths=audio_files,
            opt_path=output_folder,
            filter_radius=rvc_voice_filter_radius,
            resemple_rate=rvc_voice_resemple_rate,
            envelope_mix=rvc_voice_envelope_mix,
        )
        print(output_folder)
        done_message = f"Done, file saved in {folder_name} folder"
        return None, None, done_message
    else:
        # Func for single file rvc
        output_file_name = output_folder / \
            f"rvc_{rvc_voice_settings_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        infer_rvc(
            pitch=rvc_voice_settings_pitch,
            index_rate=rvc_voice_settings_index_rate,
            protect_voiceless=rvc_voice_settings_protect_voiceless,
            method=rvc_voice_settings_method,
            index_path=rvc_voice_settings_model_path,
            model_path=rvc_voice_settings_index_path,
            input_path=input_file,
            opt_path=output_file_name,
            filter_radius=rvc_voice_filter_radius,
            resemple_rate=rvc_voice_resemple_rate,
            envelope_mix=rvc_voice_envelope_mix)
        done_message = f"Done"
        output_audio = output_file_name

    return None, gr.Audio(label="Result", value=output_audio), done_message


with gr.Blocks(css=css) as demo:
    gr.Markdown(
        value="# XTTS-webui by [daswer123](https://github.com/daswer123)")
    with gr.Row(elem_classes="model-choose"):
        models_list = get_folder_names_advanced(this_dir / "models")
        model = gr.Dropdown(
            label="Select XTTS model version",
            value=MODEL_VERSION,
            choices=models_list,
            elem_classes="model-choose__checkbox"
        )
        refresh_model_btn = gr.Button(
            value="Update", elem_classes="model-choose__btn")

    with gr.Tab("Text2Voice"):
        from parts.text2voice import *

    with gr.Tab("Voice2Voice"):
        with gr.Tab("RVC"):
            with gr.Row():
                with gr.Column():
                    with gr.Tab("Single"):
                        rvc_audio_single = gr.Audio(
                            label="Single file", value=None)
                    with gr.Tab("Batch"):
                        rvc_audio_batch = gr.File(
                            file_count="multiple", label="Batch files", file_types=["audio"])
                        rvc_audio_batch_path = gr.Textbox(
                            label="Path to folder with audio files (High priority)", value=None)
                    with gr.Column():
                        with gr.Row():
                            rvc_voice_settings_model_name = gr.Dropdown(
                                label="RVC Model name", info="Create a folder with your model name in the rvc folder and put .pth and .index there , .index optional", choices=rvc_models)
                            rvc_voice_settings_update_btn = gr.Button(
                                value="Update", elem_classes="rvc_update-btn", visible=True)
                        rvc_voice_settings_model_path = gr.Textbox(
                            label="RVC Model", value="", visible=True, interactive=False)
                        rvc_voice_settings_index_path = gr.Textbox(
                            label="Index file", value="", visible=True, interactive=False)
                        rvc_voice_settings_pitch = gr.Slider(
                            minimum=-24, maximum=24, value=0, step=1, label="Pitch")
                        rvc_voice_settings_index_rate = gr.Slider(
                            minimum=0, maximum=1, value=0.75, step=0.01, label="Index rate")
                        rvc_voice_settings_protect_voiceless = gr.Slider(
                            minimum=0, maximum=0.5, value=0.33, step=0.01, label="Protect voiceless")
                        rvc_voice_settings_method = gr.Radio(
                            ["crepe", "pm", "rmvpe", "harvest"], value="rmvpe", label="RVC Method")
                        rvc_voice_filter_radius = gr.Slider(
                            minimum=0, maximum=7, value=3, step=1, label="If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.")
                        rvc_voice_resemple_rate = gr.Slider(
                            minimum=0, maximum=48000, value=0, step=1, label="Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling")
                        rvc_voice_envelope_mix = gr.Slider(
                            minimum=0, maximum=1, value=0.25, step=0.01, label="Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is used")
                with gr.Column():
                    rvc_voice_status_bar = gr.Label(
                        value="Choose model and click Infer button")
                    rvc_video_output = gr.Video(
                        label="Waveform RVC", value=None, interactive=False, visible=False)
                    rvc_voice_output = gr.Audio(
                        label="Result", value=None, interactive=False)
                    rvc_voice_infer_btn = gr.Button(value="Infer")

        with gr.Tab("OpenVoice"):
            with gr.Row():
                with gr.Column():
                    with gr.Tab("Single"):
                        openvoice_audio_single = gr.Audio(
                            label="Single file", value=None)
                    with gr.Tab("Batch"):
                        openvoice_audio_batch = gr.File(
                            file_count="multiple", label="Batch files", file_types=["audio"])
                        openvoice_audio_batch_path = gr.Textbox(
                            label="Path to folder with audio files (High priority)", value=None)
                    with gr.Column():
                        openvoice_voice_ref_list = get_openvoice_refs(this_dir)

                        if len(openvoice_voice_ref_list) == 0:
                            openvoice_voice_ref_list = ["None"]

                        gr.Markdown(
                            "**Add samples to the voice2voice/openvoice audio files folder or select from the reference speaker list**")
                        with gr.Row():
                            opvoice_voice_ref_list = gr.Dropdown(
                                label="Reference sample", value=openvoice_voice_ref_list[0], choices=openvoice_voice_ref_list)
                            opvoice_voice_show_speakers = gr.Checkbox(
                                value=False, label="Show choises from the speakers folder")
                with gr.Column():
                    openvoice_status_bar = gr.Label(
                        value="Choose reference and click Infer button")
                    openvoice_video_output = gr.Video(
                        label="Waveform RVC", value=None, interactive=False, visible=False)
                    openvoice_voice_output = gr.Audio(
                        label="Result", value=None, interactive=False)
                    openvoice_voice_infer_btn = gr.Button(value="Infer")
        with gr.Tab("Translate"):
            gr.Markdown("WIP")

    with gr.Tab("Train"):
        from parts.train import *

    with gr.Tab("Instuments"):
        from parts.instuments import *

    from modules.text2voice.voice2voice import select_rvc_model, update_rvc_model, update_openvoice_ref_list
    rvc_voice_settings_model_name.change(fn=select_rvc_model, inputs=[rvc_voice_settings_model_name], outputs=[
        rvc_voice_settings_model_path, rvc_voice_settings_index_path])
    rvc_voice_settings_update_btn.click(fn=update_rvc_model, inputs=[rvc_voice_settings_model_name], outputs=[
        rvc_voice_settings_model_name, rvc_voice_settings_model_path, rvc_voice_settings_index_path])

    rvc_voice_infer_btn.click(fn=infer_rvc_audio, inputs=[
        # INPUT
        rvc_audio_single,
        rvc_audio_batch,
        rvc_audio_batch_path,
        # PATH
        rvc_voice_settings_model_name,
        rvc_voice_settings_model_path,
        rvc_voice_settings_index_path,
        # SETTINGS
        rvc_voice_settings_pitch,
        rvc_voice_settings_index_rate,
        rvc_voice_settings_protect_voiceless,
        rvc_voice_settings_method,
        rvc_voice_filter_radius,
        rvc_voice_resemple_rate,
        rvc_voice_envelope_mix,
        # STATUS
        rvc_voice_status_bar
    ], outputs=[
        rvc_video_output,
        rvc_voice_output,
        rvc_voice_status_bar
    ])

    opvoice_voice_show_speakers.change(fn=update_openvoice_ref_list, inputs=[
        opvoice_voice_ref_list, opvoice_voice_show_speakers], outputs=[opvoice_voice_ref_list])

    openvoice_voice_infer_btn.click(fn=infer_openvoice_audio,inputs=[openvoice_audio_single,openvoice_audio_batch,openvoice_audio_batch_path,opvoice_voice_ref_list,openvoice_status_bar,speaker_path_text],outputs=[openvoice_video_output,openvoice_voice_output,openvoice_status_bar])
    # LOAD FUNCTIONS AND HANDLERS
    import modules
if __name__ == "__main__":
    demo.queue()
    demo.launch(inbrowser=True, share=True)
