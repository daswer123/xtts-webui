import gradio as gr
from loguru import logger

from xtts_webui import *


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
# infer_type.input(fn=change_infer_type, inputs=[infer_type], outputs=[infer_type])
