
from scripts.modeldownloader import get_folder_names_advanced
from scripts.tts_funcs import TTSWrapper
from silero_tts.silero_tts import SileroTTS

import os
import gradio as gr
from pathlib import Path
from loguru import logger

from i18n.i18n import I18nAuto
i18n = I18nAuto()

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

deepl_api_key = ""

reversed_supported_languages = {
    name: code for code, name in supported_languages.items()}
reversed_supported_languages_list = list(reversed_supported_languages.keys())
supported_languages_list = list(supported_languages.keys())

# INIT SILERO TTS 
SILERO = SileroTTS(language="ru", model_id="v4_ru")

# INIT MODEL
XTTS = TTSWrapper(OUTPUT_FOLDER, SPEAKER_FOLDER, LOWVRAM_MODE,
                  MODEL_SOURCE, MODEL_VERSION, DEVICE)

# LOAD MODEL
logger.info(f"{i18n('Start loading model')} {MODEL_VERSION}")
this_dir = Path(__file__).parent.resolve()

logger.info(f"this dir: {this_dir}")
XTTS.load_model(this_dir)


with gr.Blocks(css=css) as demo:
    gr.Markdown(
        value=f"# XTTS-webui by [daswer123](https://github.com/daswer123)\n{i18n(' ')}")
    with gr.Row(elem_classes="model-choose"):
        models_list = get_folder_names_advanced(this_dir / "models")
        model = gr.Dropdown(
            label=i18n("Select XTTS model version"),
            value=MODEL_VERSION,
            choices=models_list,
            elem_classes="model-choose__checkbox"
        )
        refresh_model_btn = gr.Button(
            value=i18n("Update"), elem_classes="model-choose__btn")

    with gr.Tab(i18n("Text2Voice")):
        from parts.text2voice import *

    with gr.Tab(i18n("Voice2Voice")):
        from parts.voice2voice import *

    # TODO 
    # Fix train Tab
    with gr.Tab(i18n("Train"),render=False):
        from parts.train import *

    with gr.Tab(i18n("Instuments")):
        from parts.instuments import *

    
    # LOAD FUNCTIONS AND HANDLERS
    import modules
if __name__ == "__main__":
    demo.queue()
    demo.launch(inbrowser=True, share=True)
