import gradio as gr
from xtts_webui import *

with gr.Tab("Translate"):
    with gr.Row():
        with gr.Column():
            with gr.Tab("Single"):
                translate_audio_single = gr.Audio(
                    label="Single file", value=None)
            # LATER
            with gr.Tab("Batch"):
                gr.Markdown("Work in progress...")
                with gr.Column(visible=False):
                    translate_audio_batch = gr.File(
                        file_count="multiple", label="Batch files", file_types=["audio"])
                    translate_audio_batch_path = gr.Textbox(
                        label="Path to folder with audio files (High priority)", value=None)
            with gr.Column():
                translate_whisper_model = gr.Dropdown(label="Whisper Model", choices=[
                                                      "small", "medium", "large-v2", "large-v3"], value="medium")
                translate_audio_mode = gr.Radio(label="Mode", choices=[
                                                1, 2], value=2, info="1 - Takes each sentence as a sample and voices the text using this sample\n2 - Intended for 1 speaker, takes the sample that is longer and closest to the current sentence.")
                translate_translator = gr.Radio(label="Translator", choices=[
                                                "google", "bing", "baidu"], value="google")
                with gr.Row():
                    translate_source_lang = gr.Text(
                        value="auto", label="Enter lang code of source lang, if you want define automaticly type 'auto'")
                    translate_target_lang = gr.Dropdown(
                        label="Target lang", choices=supported_languages_list, value="ru")
                    translate_speaker_lang = gr.Dropdown(
                        label="Speaker Accent", choices=supported_languages_list, value="ru")
                with gr.Column():
                  with gr.Accordion("XTTS settings", open=False,visible=True):
                    translate_speed = gr.Slider(
                        label="Speed",
                        minimum=0.5,
                        maximum=2,
                        step=0.01,
                        value=1,
                    )
                    translate_temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.01,
                        maximum=1,
                        step=0.05,
                        value=0.75,
                    )
                    translate_length_penalty = gr.Slider(
                        label="Length Penalty",
                        minimum=-10.0,
                        maximum=10.0,
                        step=0.5,
                        value=1,
                    )
                    translate_repetition_penalty = gr.Slider(
                        label="Repetition Penalty",
                        minimum=1,
                        maximum=10,
                        step=0.5,
                        value=5,
                    )
                    translate_top_k = gr.Slider(
                        label="Top K",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=50,
                    )
                    translate_top_p = gr.Slider(
                        label="Top P",
                        minimum=0.01,
                        maximum=1,
                        step=0.05,
                        value=0.85,
                    )
                    translate_sentence_split = gr.Checkbox(
                        label="Enable text splitting",
                        value=False,
                    )
                    
        with gr.Column():
            translate_status_bar = gr.Label(
                value="Select target language, mode and upload audio then press translate button.")
            translate_video_output = gr.Video(
                label="Waveform Translate", value=None, interactive=False, visible=False)
            translate_voice_output = gr.Audio(
                label="Result", value=None, interactive=False)
            translate_btn = gr.Button(value="Translate")

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
                    minimum=0, maximum=1, value=1, step=0.01, label="Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is used")
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
                label="Waveform OpenVoice", value=None, interactive=False, visible=False)
            openvoice_voice_output = gr.Audio(
                label="Result", value=None, interactive=False)
            openvoice_voice_infer_btn = gr.Button(value="Infer")
