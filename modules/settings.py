import langid

import gradio as gr
from loguru import logger

from xtts_webui import *

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

def change_infer_type(infer_type):
    XTTS.unload_model()

    XTTS.model_source = infer_type
    XTTS.load_model(this_dir)

    return infer_type

def change_language(languages):
    lang_code = reversed_supported_languages[languages]
    XTTS.language = lang_code
    return languages

languages.change(fn=change_language, inputs=[languages], outputs=[languages])
infer_type.input(fn=change_infer_type, inputs=[infer_type],outputs=[infer_type])