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


def change_silero_language(silero_language):
    SILERO.change_language(silero_language)
    
    avalible_models = SILERO.get_available_models()[silero_language]
    # avalible_speakers = SILERO.get_available_speakers()
    # avalible_sample_rates = SILERO.get_available_sample_rates()
    
    return gr.update(choices=avalible_models, value=avalible_models[0])

def change_silero_model(silero_model):
    SILERO.change_model(silero_model)
    
    silero_avalible_speakers = list(SILERO.get_available_speakers())
    silero_avalible_sample_rate = list(SILERO.get_available_sample_rates())
    
    return gr.update(choices=silero_avalible_speakers, value=silero_avalible_speakers[0]) , gr.update(choices=silero_avalible_sample_rate, value=silero_avalible_sample_rate[-1])

def change_silero_speaker(silero_speaker):
    SILERO.change_speaker(silero_speaker)
    
def change_silero_sample_rate(silero_sample_rate):
    SILERO.change_sample_rate(silero_sample_rate)

silero_language.change(fn=change_silero_language,inputs=[silero_language],outputs=[silero_models])
silero_models.change(fn=change_silero_model,inputs=[silero_models],outputs=[silero_speaker,silero_sample_rate])
silero_speaker.change(fn=change_silero_speaker,inputs=[silero_speaker],outputs=[])
silero_sample_rate.change(fn=change_silero_sample_rate,inputs=[silero_sample_rate],outputs=[])