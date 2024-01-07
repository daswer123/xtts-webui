from scripts.modeldownloader import get_folder_names_advanced
from scripts.tts_funcs import TTSWrapper
from scripts.voice2voice import get_rvc_models,find_rvc_model_by_name

import os
import gradio as gr
from pathlib import Path
from loguru import logger

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
            
            with gr.Column():
                status_bar = gr.Label(label="Status bar",value="Enter text, select language and reference speaker, and click Generate")
                video_gr = gr.Video(label="Waveform Visual",visible=False,interactive=False)
                audio_gr = gr.Audio(label="Synthesised Audio",interactive=False, autoplay=False)
                generate_btn = gr.Button(value="Generate",size="lg",elem_classes="generate-btn")

                rvc_models = []
                current_rvc_model = ""
                if RVC_ENABLE:
                    # Get RVC models
                    rvc_models = []
                    current_rvc_model = ""
                    rvc_models_full = get_rvc_models(this_dir)
                    if len(rvc_models_full) > 1:
                      current_rvc_model = rvc_models_full[0]["model_name"]
                      for rvc_model in rvc_models_full:
                        rvc_models.append(rvc_model["model_name"])
                    # print(rvc_models)

                with gr.Accordion(label="Output settings",open=True):
                  with gr.Column():
                    with gr.Row():
                      enable_waveform = gr.Checkbox(label="Enable Waveform",info="Create video based on audio in the form of a waveform",value=False)
                      improve_output_audio = gr.Checkbox(label="Improve output quality",info="Reduces noise and makes audio slightly better",value=False)
                      improve_output_resemble = gr.Checkbox(label="Resemble enhancement",info="Uses Resemble enhance to improve sound quality through neural networking. Uses extra 4GB VRAM",value=False)
                    with gr.Row():
                      improve_output_rvc = gr.Radio(label="Choose RVC or OpenVoice to improve result",visible=RVC_ENABLE,info="Uses RVC to convert the output to the RVC model voice, make sure you have a model folder with the pth file inside the rvc folder",choices=["RVC","OpenVoice","None"],value="None")
                    with gr.Accordion(label="Resemble enhancement Settings",open=False):
                        enhance_resemble_chunk_seconds = gr.Slider(minimum=2, maximum=40, value=8, step=1, label="Chunk seconds (more secods more VRAM usage and faster inference speed)")
                        enhance_resemble_chunk_overlap = gr.Slider(minimum=0.1, maximum=2, value=1, step=0.2, label="Overlap seconds")
                        enhance_resemble_solver = gr.Dropdown(label="CFM ODE Solver (Midpoint is recommended)",choices=["Midpoint", "RK4", "Euler"], value="Midpoint")
                        enhance_resemble_num_funcs = gr.Slider(minimum=1, maximum=128, value=64, step=1, label="CFM Number of Function Evaluations (higher values in general yield better quality but may be slower)")
                        enhance_resemble_temperature = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="CFM Prior Temperature (higher values can improve quality but can reduce stability)")
                        enhance_resemble_denoise = gr.Checkbox(value=True, label="Denoise Before Enhancement (tick if your audio contains heavy background noise)")
                    
                    with gr.Accordion(label="OpenVoice settings",visible=RVC_ENABLE, open=False):
                      gr.Markdown("**Download directly or use from the speaker's library**")
                      opvoice_ref = gr.Audio(label="OpenVoice Reference",interactive=True)
                      opvoice_ref_list = gr.Dropdown(label="Reference Speaker",value="None",choices=["None"])
                      opvoice_show_speakers = gr.Checkbox(value=False,label="Show results from the speakers folder")

                    with gr.Accordion(label="RVC settings",visible=RVC_ENABLE, open=False):
                      # RVC variables 
                      rvc_settings_model_path = gr.Textbox(label="RVC Model",value="",visible=True,interactive=False)
                      rvc_settings_index_path = gr.Textbox(label="Index file",value="",visible=True,interactive=False)
                      with gr.Row():
                        rvc_settings_model_name = gr.Dropdown(label="RVC Model name",info="Create a folder with your model name in the rvc folder and put .pth and .index there , .index optional",choices=rvc_models)
                        rvc_settings_update_btn = gr.Button(value="Update",elem_classes="rvc_update-btn",visible=True)
                      rvc_settings_pitch = gr.Slider(minimum=-24, maximum=24, value=0, step=1, label="Pitch")
                      rvc_settings_index_rate = gr.Slider(minimum=0, maximum=1, value=0.8, step=0.01, label="Index rate")
                      rvc_settings_protect_voiceless = gr.Slider(minimum=0, maximum=0.5, value=0.33, step=0.01, label="Protect voiceless")
                      rvc_settings_method = gr.Radio(["crepe", "mangio-crepe","rmvpe","harvest"],value="rmvpe", label="RVC Method")
                    with gr.Row():
                      output_type = gr.Radio(["mp3","wav"],value="wav", label="Output Type")
                  additional_text_input = gr.Textbox(label="File Name Value", value="output")

                 # Variables
                speaker_value_text = gr.Textbox(label="Reference Speaker Name",value=speaker_value,visible=False)
                speaker_path_text = gr.Textbox(label="Reference Speaker Path",value="",visible=False)
                speaker_wav_modifyed = gr.Checkbox("Reference Audio",visible=False, value = False )
                speaker_ref_wavs = gr.Text(visible=False)                
    
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
        gr.Markdown("WIP")
    
    # LOAD FUNCTIONS AND HANDLERS
    import modules
if __name__ == "__main__":
    demo.queue()
    demo.launch(inbrowser=True,share=True)   
