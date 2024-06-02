
import gradio as gr
from scripts.voice2voice import get_rvc_models, find_rvc_model_by_name, get_openvoice_refs

# from silero_tts.silero_tts import SileroTTS


from xtts_webui import *

from i18n.i18n import I18nAuto
i18n = I18nAuto()

with gr.Row():
    with gr.Column():
        with gr.Tab(i18n("Text")):
            text = gr.TextArea(label=i18n("Input Text"),
                               placeholder=i18n("Input Text Here..."))
        with gr.Tab(i18n("Batch")):
            batch_generation = gr.Files(
                label=i18n("Upload .txt files"), file_types=["text"])
            batch_generation_path = gr.Textbox(
                label=i18n("Path to folder with .txt files, Has priority over all "), value="")
        with gr.Tab(i18n("Subtitles")):
            batch_sub_generation = gr.Files(
                label=i18n("Upload srt or ass files"), file_types=[".ass",".srt"])
            batch_sub_generation_path = gr.Textbox(
                label=i18n("Path to folder with srt or ass, Has priority over all"), value="")
            sync_sub_generation = gr.Checkbox(label=i18n("Synchronise subtitle timings"),value=False)
        
        with gr.Column():
          voice_engine = gr.Radio(label=i18n("Select Voice Engine"), choices=["XTTS", "SILERO"], value="XTTS", visible=False)
          with gr.Tab("XTTS"):    
            language_auto_detect = gr.Checkbox(
                label=i18n("Enable language auto detect"), info=i18n("If your language is not supported or the text is less than 20 characters, this function will not work"))
            languages = gr.Dropdown(
                label=i18n("Language"), choices=reversed_supported_languages_list, value="English")
        
            speed = gr.Slider(
                label=i18n("speed"),
                minimum=0.1,
                maximum=2,
                step=0.05,
                value=1,
            )
            with gr.Accordion(i18n("Advanced settings"), open=False) as acr:
                temperature = gr.Slider(
                    label=i18n("Temperature"),
                    minimum=0.01,
                    maximum=1,
                    step=0.05,
                    value=0.75,
                )
                length_penalty = gr.Slider(
                    label=i18n("Length Penalty"),
                    minimum=-10.0,
                    maximum=10.0,
                    step=0.5,
                    value=1,
                )
                repetition_penalty = gr.Slider(
                    label=i18n("Repetition Penalty"),
                    minimum=1,
                    maximum=10,
                    step=0.5,
                    value=5,
                )
                top_k = gr.Slider(
                    label=i18n("Top K"),
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=50,
                )
                top_p = gr.Slider(
                    label=i18n("Top P"),
                    minimum=0.01,
                    maximum=1,
                    step=0.05,
                    value=0.85,
                )
                sentence_split = gr.Checkbox(
                    label=i18n("Enable text splitting"),
                    value=True,
                )

                # infer_type = gr.Radio(["api", "local"], value="local", label="Type of Processing",
                #                       info="Defines how the text will be processed,local gives you more options. Api does not allow you to use advanced settings")

            speakers_list = XTTS.get_speakers()
            speaker_value = ""
            if not speakers_list:
                speakers_list = ["None"]
                speaker_value = "None"
            else:
                speaker_value = speakers_list[0]
                XTTS.speaker_wav = speaker_value

            with gr.Row():
                ref_speaker_list = gr.Dropdown(
                    label=i18n("Reference Speaker from folder 'speakers'"), value=speaker_value, choices=speakers_list,allow_custom_value=True)
                show_ref_speaker_from_list = gr.Checkbox(
                    value=False, label=i18n("Show reference sample"), info=i18n("This option will allow you to listen to your reference sample"))
                show_inbuildstudio_speaker = gr.Checkbox(
                    value=False, label=i18n("Show in list avalible speakers inbuild speakers"), info=i18n("This option will allow you to add pre-prepared voices from coqua studio to the list of available voices"))
                update_ref_speaker_list_btn = gr.Button(
                    value=i18n("Update"), elem_classes="speaker-update__btn")
            ref_speaker_example = gr.Audio(
                label=i18n("speaker sample"), sources="upload", visible=False, interactive=False)

            with gr.Tab(label=i18n("Single")):
                ref_speaker = gr.Audio(
                    label=i18n("Reference Speaker (mp3, wav, flac)"), editable=False)
            with gr.Tab(label=i18n("Multiple")):
                ref_speakers = gr.Files(
                    label=i18n("Reference Speakers (mp3, wav, flac)"), file_types=["audio"])

            with gr.Accordion(label=i18n("Reference Speaker settings."), open=False):
                gr.Markdown(
                    value=i18n("**Note: the settings only work when you enable them and upload files when they are enabled**"))
                gr.Markdown(
                    value=i18n("Take a look at how to create good samples [here](https://github.com/daswer123/xtts-api-server?tab=readme-ov-file#note-on-creating-samples-for-quality-voice-cloning)"))
                with gr.Row():
                    use_resample = gr.Checkbox(
                        label=i18n("Resample reference audio to 22050Hz"), info=i18n("This is for better processing"), value=True)
                    improve_reference_audio = gr.Checkbox(
                        label=i18n("Clean up reference audio"), info=i18n("Trim silence, use lowpass and highpass filters"), value=False)
                    improve_reference_resemble = gr.Checkbox(
                        label=i18n("Resemble enhancement (Uses extra 4GB VRAM)"), info=i18n("You can find the settings next to the settings for the result"), value=False)
                auto_cut = gr.Slider(
                    label=i18n("Automatically trim audio up to x seconds, 0 without trimming "),
                    minimum=0,
                    maximum=30,
                    step=1,
                    value=0,
                )
                gr.Markdown(
                    value=i18n("You can save the downloaded recording or microphone recording to a shared list, you need to set a name and click save"))
                speaker_wav_save_name = gr.Textbox(
                    label=i18n("Speaker save name"), value="new_speaker_name")
                save_speaker_btn = gr.Button(
                    value=i18n("Save a single sample for the speaker"), visible=False)
                save_multiple_speaker_btn = gr.Button(
                    value=i18n("Save multiple samples for the speaker"), visible=False)

          with gr.Tab("Silero", render=False):
                with gr.Column():
                    
                    # Get Data
                    silero_avalible_models = list(SILERO.get_available_models()["ru"])
                    silero_avalible_speakers = list(SILERO.get_available_speakers())
                    silero_avalible_sample_rate = list(SILERO.get_available_sample_rates())
                    
                    print(silero_avalible_speakers)
            
                    silero_language = gr.Dropdown(label=i18n("Language Silero"), choices=["ru", "en", "de", "es", "fr", "ba", "xal", "tt", "uz", "ua", "indic"], value="ru")
                    silero_models = gr.Dropdown(label=i18n("Model"), choices=silero_avalible_models, value=silero_avalible_models[0])
                    with gr.Row():
                        silero_speaker = gr.Dropdown(label=i18n("Speaker"), choices=silero_avalible_speakers, value=silero_avalible_speakers[0])
                    # TODO
                    #     siler_show_speaker_sample = gr.Checkbox(label=i18n("Show sample"),info=i18n("This option will allow you to listen to speaker sample"), value=False, interactive=False)
                    # silero_speaker_sample = gr.Audio(label=i18n("Speaker sample"),visible=False, interactive=False)    
                    
                    silero_sample_rate = gr.Radio(label=i18n("Sample rate"), choices=silero_avalible_sample_rate, value=silero_avalible_sample_rate[-1])
                    silero_device = gr.Radio(label=i18n("Device"),info=i18n("Cpu pretty fast"), choices=["cpu", "cuda:0"], value="cpu")

    with gr.Column():
        status_bar = gr.Label(
            label=i18n("Status bar"), value=i18n("Enter text, select language and speaker, then click Generate"))
        video_gr = gr.Video(label=i18n("Waveform Visual"),
                            visible=False, interactive=False)
        audio_gr = gr.Audio(label=i18n("Synthesised Audio"),
                            interactive=False, autoplay=False)
        generate_btn = gr.Button(
            value=i18n("Generate"), size="lg", elem_classes="generate-btn")

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

        with gr.Accordion(label=i18n("Output settings"), open=True):
            with gr.Column():
                with gr.Row():
                    enable_waveform = gr.Checkbox(
                        label=i18n("Enable Waveform"), info=i18n("Create video based on audio in the form of a waveform"), value=False)
                    improve_output_audio = gr.Checkbox(
                        label=i18n("Improve output quality"), info=i18n("Reduces noise and makes audio slightly better"), value=False)
                    improve_output_resemble = gr.Checkbox(
                        label=i18n("Resemble enhancement"), info=i18n("Uses Resemble enhance to improve sound quality through neural networking. Uses extra 4GB VRAM"), value=False)
                with gr.Row():
                    improve_output_voice2voice = gr.Radio(label=i18n("Use RVC or OpenVoice to improve result"), visible=RVC_ENABLE,
                                                          info=i18n("Uses RVC to convert the output to the RVC model voice, make sure you have a model folder with the pth file inside the voice2voice/rvc folder"), choices=["RVC", "OpenVoice", "None"], value="None")
                with gr.Accordion(label=i18n("Resemble enhancement Settings"), open=False):
                    enhance_resemble_chunk_seconds = gr.Slider(
                        minimum=2, maximum=40, value=8, step=1, label=i18n("Chunk seconds (more secods more VRAM usage and faster inference speed)"))
                    enhance_resemble_chunk_overlap = gr.Slider(
                        minimum=0.1, maximum=2, value=1, step=0.2, label=i18n("Overlap seconds"))
                    enhance_resemble_solver = gr.Dropdown(label=i18n("CFM ODE Solver (Midpoint is recommended)"), choices=[
                                                          "Midpoint", "RK4", "Euler"], value="Midpoint")
                    enhance_resemble_num_funcs = gr.Slider(
                        minimum=1, maximum=128, value=64, step=1, label=i18n("CFM Number of Function Evaluations (higher values in general yield better quality but may be slower)"))
                    enhance_resemble_temperature = gr.Slider(
                        minimum=0, maximum=1, value=0.5, step=0.01, label=i18n("CFM Prior Temperature (higher values can improve quality but can reduce stability)"))
                    enhance_resemble_denoise = gr.Checkbox(
                        value=True, label=i18n("Denoise Before Enhancement (tick if your audio contains heavy background noise)"))

                with gr.Accordion(label=i18n("OpenVoice settings"), visible=RVC_ENABLE, open=False):
                    open_voice_ref_list = get_openvoice_refs(this_dir)
                    if len(open_voice_ref_list) == 0:
                        open_voice_ref_list = ["None"]

                    gr.Markdown(
                        i18n("**Add samples to the voice2voice/openvoice audio files folder or select from the reference speaker list**"))
                    opvoice_ref_list = gr.Dropdown(
                        label=i18n("Reference sample"), value=open_voice_ref_list[0], choices=open_voice_ref_list)
                    opvoice_show_speakers = gr.Checkbox(
                        value=False, label=i18n("Show choises from the speakers folder"))

                with gr.Accordion(label=i18n("RVC settings"), visible=RVC_ENABLE, open=False):
                    # RVC variables
                    with gr.Row():
                        rvc_settings_model_name = gr.Dropdown(
                            label=i18n("RVC Model name"), info=i18n("Create a folder with your model name in the rvc folder and put .pth and .index there , .index optional"), choices=rvc_models)
                        rvc_settings_update_btn = gr.Button(
                            value=i18n("Update"), elem_classes="rvc_update-btn", visible=True)
                    rvc_settings_model_path = gr.Textbox(
                        label=i18n("RVC Model"), value="", visible=True, interactive=False)
                    rvc_settings_index_path = gr.Textbox(
                        label=i18n("Index file"), value="", visible=True, interactive=False)
                    rvc_settings_pitch = gr.Slider(
                        minimum=-24, maximum=24, value=0, step=1, label=i18n("Pitch"))
                    rvc_settings_index_rate = gr.Slider(
                        minimum=0, maximum=1, value=0.75, step=0.01, label=i18n("Index rate"))
                    rvc_settings_protect_voiceless = gr.Slider(
                        minimum=0, maximum=0.5, value=0.33, step=0.01, label=i18n("Protect voiceless"))
                    rvc_settings_method = gr.Radio(
                        ["crepe", "pm", "rmvpe", "harvest"], value="rmvpe", label=i18n("RVC Method"))
                    rvc_settings_filter_radius = gr.Slider(
                        minimum=0, maximum=7, value=3, step=1, label=i18n("If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness."))
                    rvc_settings_resemple_rate = gr.Slider(
                        minimum=0, maximum=48000, value=0, step=1, label=i18n("Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling"))
                    rvc_settings_envelope_mix = gr.Slider(
                        minimum=0, maximum=1, value=1, step=0.01, label=i18n("Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is used"))
                with gr.Row():
                    output_type = gr.Radio(
                        ["mp3", "wav"], value="wav", label=i18n("Output Type"))
            additional_text_input = gr.Textbox(
                label=i18n("File Name Value"), value="output")

           # Variables
        speaker_value_text = gr.Textbox(
            label=i18n("Reference Speaker Name"), value=speaker_value, visible=False)
        speaker_path_text = gr.Textbox(
            label=i18n("Reference Speaker Path"), value="", visible=False)
        speaker_wav_modifyed = gr.Checkbox(
            i18n("Reference Audio"), visible=False, value=False)
        speaker_ref_wavs = gr.Text(visible=False)
