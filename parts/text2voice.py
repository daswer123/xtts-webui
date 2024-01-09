
import gradio as gr
from scripts.voice2voice import get_rvc_models, find_rvc_model_by_name, get_openvoice_refs

from xtts_webui import *

with gr.Row():
    with gr.Column():
        with gr.Tab("Text"):
            text = gr.TextArea(label="Input Text",
                               placeholder="Input Text Here...")
        with gr.Tab("Batch"):
            batch_generation = gr.Files(
                label="Upload .txt files", file_types=["text"])
            batch_generation_path = gr.Textbox(
                label="Path to folder with .txt files, Has priority over all ", value="")

        language_auto_detect = gr.Checkbox(
            label="Enable language auto detect", info="If your language is not supported or the text is less than 20 characters, this function will not work")
        languages = gr.Dropdown(
            label="Language", choices=reversed_supported_languages_list, value="English")
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
            length_penalty = gr.Slider(
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

            infer_type = gr.Radio(["api", "local"], value="local", label="Type of Processing",
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
            ref_speaker_list = gr.Dropdown(
                label="Reference Speaker from folder 'speakers'", value=speaker_value, choices=speakers_list)
            show_ref_speaker_from_list = gr.Checkbox(
                value=False, label="Show example", info="This option will allow you to listen to your reference sample")
            update_ref_speaker_list_btn = gr.Button(
                value="Update", elem_classes="speaker-update__btn")
        ref_speaker_example = gr.Audio(
            label="speaker example", sources="upload", visible=False, interactive=False)

        with gr.Tab(label="Single"):
            ref_speaker = gr.Audio(
                label="Reference Speaker (mp3, wav, flac)", editable=False)
        with gr.Tab(label="Multiple"):
            ref_speakers = gr.Files(
                label="Reference Speakers (mp3, wav, flac)", file_types=["audio"])

        with gr.Accordion(label="Reference Speaker settings.", open=True):
            gr.Markdown(
                value="**Note: the settings only work when you enable them and upload files when they are enabled**")
            gr.Markdown(
                value="Take a look at how to create good samples [here](https://github.com/daswer123/xtts-api-server?tab=readme-ov-file#note-on-creating-samples-for-quality-voice-cloning)")
            with gr.Row():
                use_resample = gr.Checkbox(
                    label="Resample reference audio to 22050Hz", info="This is for better processing", value=True)
                improve_reference_audio = gr.Checkbox(
                    label="Clean up reference audio", info="Trim silence, use lowpass and highpass filters", value=False)
                improve_reference_resemble = gr.Checkbox(
                    label="Resemble enhancement (Uses extra 4GB VRAM)", info="You can find the settings next to the settings for the result", value=False)
            auto_cut = gr.Slider(
                label="Automatically trim audio up to x seconds, 0 without trimming ",
                minimum=0,
                maximum=30,
                step=1,
                value=0,
            )
            gr.Markdown(
                value="You can save the downloaded recording or microphone recording to a shared list, you need to set a name and click save")
            speaker_wav_save_name = gr.Textbox(
                label="Speaker save name", value="new_speaker_name")
            save_speaker_btn = gr.Button(
                value="Save a single sample for the speaker", visible=False)
            save_multiple_speaker_btn = gr.Button(
                value="Save multiple samples for the speaker", visible=False)

    with gr.Column():
        status_bar = gr.Label(
            label="Status bar", value="Enter text, select language and reference speaker, and click Generate")
        video_gr = gr.Video(label="Waveform Visual",
                            visible=False, interactive=False)
        audio_gr = gr.Audio(label="Synthesised Audio",
                            interactive=False, autoplay=False)
        generate_btn = gr.Button(
            value="Generate", size="lg", elem_classes="generate-btn")

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

        with gr.Accordion(label="Output settings", open=True):
            with gr.Column():
                with gr.Row():
                    enable_waveform = gr.Checkbox(
                        label="Enable Waveform", info="Create video based on audio in the form of a waveform", value=False)
                    improve_output_audio = gr.Checkbox(
                        label="Improve output quality", info="Reduces noise and makes audio slightly better", value=False)
                    improve_output_resemble = gr.Checkbox(
                        label="Resemble enhancement", info="Uses Resemble enhance to improve sound quality through neural networking. Uses extra 4GB VRAM", value=False)
                with gr.Row():
                    improve_output_voice2voice = gr.Radio(label="Choose RVC or OpenVoice to improve result", visible=RVC_ENABLE,
                                                          info="Uses RVC to convert the output to the RVC model voice, make sure you have a model folder with the pth file inside the rvc folder", choices=["RVC", "OpenVoice", "None"], value="None")
                with gr.Accordion(label="Resemble enhancement Settings", open=False):
                    enhance_resemble_chunk_seconds = gr.Slider(
                        minimum=2, maximum=40, value=8, step=1, label="Chunk seconds (more secods more VRAM usage and faster inference speed)")
                    enhance_resemble_chunk_overlap = gr.Slider(
                        minimum=0.1, maximum=2, value=1, step=0.2, label="Overlap seconds")
                    enhance_resemble_solver = gr.Dropdown(label="CFM ODE Solver (Midpoint is recommended)", choices=[
                                                          "Midpoint", "RK4", "Euler"], value="Midpoint")
                    enhance_resemble_num_funcs = gr.Slider(
                        minimum=1, maximum=128, value=64, step=1, label="CFM Number of Function Evaluations (higher values in general yield better quality but may be slower)")
                    enhance_resemble_temperature = gr.Slider(
                        minimum=0, maximum=1, value=0.5, step=0.01, label="CFM Prior Temperature (higher values can improve quality but can reduce stability)")
                    enhance_resemble_denoise = gr.Checkbox(
                        value=True, label="Denoise Before Enhancement (tick if your audio contains heavy background noise)")

                with gr.Accordion(label="OpenVoice settings", visible=RVC_ENABLE, open=False):
                    open_voice_ref_list = get_openvoice_refs(this_dir)
                    if len(open_voice_ref_list) == 0:
                        open_voice_ref_list = ["None"]

                    gr.Markdown(
                        "**Add samples to the voice2voice/openvoice audio files folder or select from the reference speaker list**")
                    opvoice_ref_list = gr.Dropdown(
                        label="Reference sample", value=open_voice_ref_list[0], choices=open_voice_ref_list)
                    opvoice_show_speakers = gr.Checkbox(
                        value=False, label="Show choises from the speakers folder")

                with gr.Accordion(label="RVC settings", visible=RVC_ENABLE, open=False):
                    # RVC variables
                    with gr.Row():
                        rvc_settings_model_name = gr.Dropdown(
                            label="RVC Model name", info="Create a folder with your model name in the rvc folder and put .pth and .index there , .index optional", choices=rvc_models)
                        rvc_settings_update_btn = gr.Button(
                            value="Update", elem_classes="rvc_update-btn", visible=True)
                    rvc_settings_model_path = gr.Textbox(
                        label="RVC Model", value="", visible=True, interactive=False)
                    rvc_settings_index_path = gr.Textbox(
                        label="Index file", value="", visible=True, interactive=False)
                    rvc_settings_pitch = gr.Slider(
                        minimum=-24, maximum=24, value=0, step=1, label="Pitch")
                    rvc_settings_index_rate = gr.Slider(
                        minimum=0, maximum=1, value=0.75, step=0.01, label="Index rate")
                    rvc_settings_protect_voiceless = gr.Slider(
                        minimum=0, maximum=0.5, value=0.33, step=0.01, label="Protect voiceless")
                    rvc_settings_method = gr.Radio(
                        ["crepe", "pm", "rmvpe", "harvest"], value="rmvpe", label="RVC Method")
                    rvc_settings_filter_radius = gr.Slider(
                        minimum=0, maximum=7, value=3, step=1, label="If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.")
                    rvc_settings_resemple_rate = gr.Slider(
                        minimum=0, maximum=48000, value=0, step=1, label="Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling")
                    rvc_settings_envelope_mix = gr.Slider(
                        minimum=0, maximum=1, value=0.25, step=0.01, label="Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is used")
                with gr.Row():
                    output_type = gr.Radio(
                        ["mp3", "wav"], value="wav", label="Output Type")
            additional_text_input = gr.Textbox(
                label="File Name Value", value="output")

           # Variables
        speaker_value_text = gr.Textbox(
            label="Reference Speaker Name", value=speaker_value, visible=False)
        speaker_path_text = gr.Textbox(
            label="Reference Speaker Path", value="", visible=False)
        speaker_wav_modifyed = gr.Checkbox(
            "Reference Audio", visible=False, value=False)
        speaker_ref_wavs = gr.Text(visible=False)
