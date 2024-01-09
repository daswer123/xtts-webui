import gradio as gr
from scripts.modeldownloader import get_folder_names_advanced

from xtts_webui import *


def reload_model(model):
    XTTS.unload_model()

    XTTS.model_version = model
    XTTS.load_model(this_dir)

    return model


def reload_list(model):
    models_list = get_folder_names_advanced(this_dir / "models")
    return gr.Dropdown(
        label="XTTS model",
        value=model,
        choices=models_list,
        elem_classes="model-choose__checkbox"
    )


model.change(fn=reload_model, inputs=[model], outputs=[model])
refresh_model_btn.click(fn=reload_list, inputs=[model], outputs=[model])
