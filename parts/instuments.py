import gradio as gr
from xtts_webui import *

from i18n.i18n import I18nAuto
i18n = I18nAuto()

from scripts.funcs import read_key_from_env
from scripts.languages import get_language_names

with gr.Tab(i18n("Resemble Enhance")):
    with gr.Row():
        with gr.Column():
            with gr.Tab(i18n("Single")):
                resemble_audio_single = gr.Audio(
                    label=i18n("Single file"), value=None)
            with gr.Tab(i18n("Batch")):
                resemble_audio_batch = gr.File(
                    file_count="multiple", label=i18n("Batch files"), file_types=["audio"])
                resemble_audio_batch_path = gr.Textbox(
                    label=i18n("Path to folder with audio files (High priority)"), value=None)
            resemble_choose_action = gr.Radio(label=i18n("Choose action"), choices=[
                                              "only_enchance", "only_denoise", "both"], value="both")
            resemble_chunk_seconds = gr.Slider(
                minimum=2, maximum=40, value=8, step=1, label=i18n("Chunk seconds (more secods more VRAM usage and faster inference speed)"))
            resemble_chunk_overlap = gr.Slider(
                minimum=0.1, maximum=2, value=1, step=0.2, label=i18n("Overlap seconds"))
            resemble_solver = gr.Dropdown(label=i18n("CFM ODE Solver (Midpoint is recommended)"), choices=[
                                          "Midpoint", "RK4", "Euler"], value="Midpoint")
            resemble_num_funcs = gr.Slider(minimum=1, maximum=128, value=64, step=1,
                                           label=i18n("CFM Number of Function Evaluations (higher values in general yield better quality but may be slower)"))
            resemble_temperature = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01,
                                             label=i18n("CFM Prior Temperature (higher values can improve quality but can reduce stability)"))
            resemble_denoise = gr.Checkbox(
                value=True, label=i18n("Denoise Before Enhancement (tick if your audio contains heavy background noise)"))
            resemble_output_type = gr.Dropdown(label=i18n("Output type"), choices=[
                                               "wav", "mp3"], value="wav")
        with gr.Column():
            resemble_status_label = gr.Label(
                value=i18n("Upload a file or files and click on the Enhance button"))
            result_denoise = gr.Audio(
                label=i18n("Denoise"), interactive=False, visible=False, value=None)
            result_enhance = gr.Audio(
                label=i18n("Result"), interactive=False, visible=True, value=None)
            resemble_generate_btn = gr.Button(value=i18n("Enhance"))

with gr.Tab("WhisperX"):
    gr.Markdown("""
This project uses whisperX which in turn uses pyannote products. In order for all the functions to work here you need to do 4 things:
1) Create an account on hugginface
2) Agree to use [Segmentation](https://huggingface.co/pyannote/segmentation-3.0)
3) Agree to use [Speaker Diarization](https://huggingface.co/pyannote/speaker-diarization-3.1)
4) Generate a token [here](https://huggingface.co/settings/tokens) and set it in the HF_TOKEN field.
                    """)
    whisper_hf_token = gr.Textbox(label="HF_TOKEN",type="password",value=read_key_from_env("HF_TOKEN"))
    with gr.Row():
        with gr.Column():
            with gr.Tab("Single file"):
                whisperx_audio_single = gr.Audio(
                    label=i18n("Single file"), value=None)
                whisperx_youtube_audio = gr.Textbox(
                    label=i18n("Youtube link"), value="")
            with gr.Tab("Batch"):
                whisperx_audio_batch = gr.File(
                    file_count="multiple", label=i18n("Batch files"), file_types=["audio"])
                whisperx_audio_batch_path = gr.Textbox(
                    label=i18n("Path to folder with audio files (High priority)"), value=None)
            with gr.Accordion(label="Main",open=True):
                whisperx_model = gr.Dropdown(label=i18n("Whisper Model"), choices=["small", "medium", "large-v2", "large-v3"], value="medium")
                whisperx_task = gr.Dropdown(label=i18n("Task"), choices=["transcribe","translate"], value="transcribe")
                whisperx_language = gr.Dropdown(choices=sorted(get_language_names()), label="Language", value="Auto")
            with gr.Accordion(label="Options",open=True):
                whisperx_align = gr.Checkbox(label="Align whisper output",value=True)
                whisperx_timestamp = gr.Checkbox(label="Word Timestamps", value=True)
                whisperx_timestamp_highlight = gr.Checkbox(label="Word Timestamps - Highlight Words", value=False)
            with gr.Accordion(label="Diarize",open=True):
                whisperx_enable_diarize = gr.Checkbox(label="Enable speaker diarization", value=False)
                whisperx_diarize_split = gr.Checkbox(label="Create a separate audio file for each speaker", value=False)
                whisperx_diarize_speakers = gr.Slider(label="Number of speakers (0 to auto)", value=2, minimum=0, maximum=10)
            with gr.Accordion(label="VAD options",open=False):
                whisperx_vad_onset = gr.Slider(label="VAD onset",info="Determines the start of speech activity. The lower the value, the more sensitive it is to detecting the beginning of speech.",value=0.500,maximum=1,minimum=0.001,step=0.001)
                whisperx_vad_offset = gr.Slider(label="VAD offset",info="Sets the end of speech activity. The lower the value, the quicker it stops processing after speech ends.",value=0.363,maximum=1,minimum=0.001,step=0.001)
            with gr.Accordion(label="Diarize options",open=False):
                whisperx_diarize_speakers_max = gr.Slider(label="Max number of speakers", value=1, minimum=1, maximum=10)
                whisperx_diarize_speakers_min = gr.Slider(label="Min number of speakers", value=8, minimum=1, maximum=10)
            with gr.Accordion(label="WhisperX options", open=False):
                whisperx_compute_type = gr.Radio(label="Compute Type",choices=["int8","float16"],value="float16",info="change to 'int8' if low on GPU mem (may reduce accuracy)")
                whisperx_device = gr.Radio(label="Device",choices=["cuda","cpu"],value="cuda",info="change to 'int8' if low on GPU mem (may reduce accuracy)")
                whisperx_batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=32, value=8, step=1,info="reduce if low on GPU mem")
        with gr.Column():
            whisperx_subtitles = gr.Files(interactive=False, label=i18n("Subtitles"))
            with gr.Column():
                whisperx_transcribe = gr.Textbox(label=i18n("Transcription"), interactive=False, visible=True, value=None)
                whisperx_segments = gr.Textbox(label=i18n("Segments"), interactive=False, visible=True, value=None)
            with gr.Column():
                whisperx_diarize_files = gr.Files(interactive=False,visible=False, label=i18n("Speaker diarization files"))
                whisperx_diarize_files_list = gr.Dropdown(visible=False)
                whisperx_diarize_audio_example = gr.Audio(visible=False)
                whisperx_transcribe_btn = gr.Button(value=i18n("Transcribe"))
            
        