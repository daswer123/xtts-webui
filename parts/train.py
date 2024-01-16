import gradio as gr
from xtts_webui import *

from i18n.i18n import I18nAuto
i18n = I18nAuto()

with gr.Column():
    gr.Markdown(f"""
{i18n('This train tab allows you to fine-tune the XTTS, resulting in a model that matches the models that can be used for voiceovers.')}

{i18n("You need to download a set of dates, select a language and click 'Train'. Once training is complete, your model will automatically rotate into the model rest position.")}

{i18n("You can watch a video from the developers that shows how to do")} [finetune](
https://www.youtube.com/watch?v=8tpDiouGxc)
    """)
    gr.Markdown(i18n("# 1 Stage - Prepare Dataset"))
    load_params_btn = gr.Button(value=i18n("Load Params from output folder"))
    custom_model_name = gr.Textbox(
        label=i18n("Finetune Model Name"),
        value="my_finetune_model",
        interactive=True,
    )

    upload_train_file = gr.File(
        file_count=i18n("multiple"),
        label=i18n("Select here the audio files that you want to use for XTTS trainining (Supported formats: wav, mp3, and flac)"),
    )

    train_whisper_model = gr.Dropdown(
        label=i18n("Whisper Model"),
        value="large-v3",
        choices=["large-v3", "large-v2", "large", "medium", "small"],
    )

    train_lang = gr.Dropdown(
        label=i18n("Dataset Language"),
        value="en",
        choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr",
                 "ru", "nl", "cs", "ar", "zh", "hu", "ko", "ja"],
    )

    gr.Markdown(i18n("# 2 Stage - Finetune XTTS"))

    train_version = gr.Dropdown(
        label=i18n("XTTS base version"),
        value="v2.0.2",
        choices=["v2.0.3", "v2.0.2", "v2.0.1", "v2.0.0", "main"],
    )

    train_csv = gr.Textbox(
        label=i18n("Train CSV:"),
        visible=False
    )

    eval_csv = gr.Textbox(
        label=i18n("Eval CSV:"),
        visible=False
    )

    train_custom_model = gr.Textbox(
        label=i18n("(Optional) Custom model.pth file , leave blank if you want to use the base file."),
        value="",
    )

    num_epochs = gr.Slider(
        label=i18n("Number of epochs:"),
        minimum=1,
        maximum=100,
        step=1,
        value=6,
    )

    batch_size = gr.Slider(
        label=i18n("Batch size:"),
        minimum=1,
        maximum=16,
        step=1,
        value=2,
    )

    grad_acumm = gr.Slider(
        label=i18n("Grad accumulation steps:"),
        minimum=2,
        maximum=128,
        step=1,
        value=2,
    )

    max_audio_length = gr.Slider(
        label=i18n("Max permitted audio size in seconds:"),
        minimum=2,
        maximum=20,
        step=1,
        value=11,
    )

    clear_train_data = gr.Dropdown(
        label=i18n("Clear train data, you will delete selected folder, after optimizing"),
        value="none",
        choices=["none", "run", "dataset", "all"]
    )

    train_status_bar = gr.Label(
        label=i18n("Train Status Bar"), value=i18n("Load data, choose options and click Train"))
    train_btn = gr.Button(value=i18n("Train"))
