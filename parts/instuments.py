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
                # whisperx_task = gr.Dropdown(label=i18n("Task"), choices=["transcribe","translate"], value="transcribe")
                whisperx_language = gr.Dropdown(choices=sorted(get_language_names()),info="Leave empty if you want auto-detection", label="Language", value="Auto")
            with gr.Accordion(label="Options",open=True):
                whisperx_align = gr.Checkbox(label="Align whisper output",value=True)
                whisperx_timestamp = gr.Checkbox(label="Word Timestamps",visible=False, value=True)
                whisperx_timestamp_highlight = gr.Checkbox(label="Word Timestamps - Highlight Words", value=False)
            with gr.Accordion(label="Diarize (need HF_TOKEN)",open=True):
                whisperx_enable_diarize = gr.Checkbox(label="Enable speaker diarization", value=False)
                whisperx_diarize_split = gr.Checkbox(label="Create a separate audio file for each speaker", value=False)
                whisperx_diarize_speakers_max = gr.Slider(label="Max number of speakers", value=1, minimum=1, maximum=10)
                whisperx_diarize_speakers_min = gr.Slider(label="Min number of speakers", value=8, minimum=1, maximum=10)
            with gr.Accordion(label="VAD options",open=False):
                whisperx_vad_onset = gr.Slider(label="VAD onset",info="Onset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected.",value=0.500,maximum=1,minimum=0.001,step=0.001)
                whisperx_vad_offset = gr.Slider(label="VAD offset",info="Offset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected.",value=0.363,maximum=1,minimum=0.001,step=0.001)
                whisperx_vad_chunk_size = gr.Slider(label="Chunk size for merging VAD segments. Default is 30, reduce this if the chunk is too long.", value=30, maximum=100, minimum=1, step=1)
            with gr.Accordion(label="WhisperX options", open=False):
                whisperx_compute_type = gr.Radio(label="Compute Type",choices=["int8","float16","float32"],value="float16",info="change to 'int8' if low on GPU mem (may reduce accuracy)")
                whisperx_device = gr.Radio(label="Device",choices=["cuda","cpu"],value="cuda",info="change to 'int8' if low on GPU mem (may reduce accuracy)")
                whisperx_batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=32, value=8, step=1,info="reduce if low on GPU mem")
                whisperx_beam_size = gr.Slider(label="Beam size",info="number of beams in beam search, only applicable when temperature is zero", minimum=1, maximum=100, value=5, step=1)
            with gr.Accordion(label="Advanced settings", open=False):
                whisperx_device_index = gr.Slider(label="Device Index", minimum=0, maximum=8, value=0, step=1)
                whisperx_threads = gr.Slider(label="Threads",info="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS", minimum=0, maximum=16, value=0, step=1)
                whisperx_align_intropolate = gr.Radio(label="Align interpolate method",info="For word .srt, method to assign timestamps to non-aligned words, or merge them into neighbouring",value="nearest",choices=["nearest", "linear", "ignore"])
                with gr.Accordion("Whisper Settings",open=False):
                    with gr.Column():
                        whisperx_temperature = gr.Slider(label="Temperature to use for sampling", minimum=0, maximum=1, value=0, step=0.01)
                        whisperx_bestof = gr.Slider(label="Number of candidates when sampling with non-zero temperature", minimum=1, maximum=20, value=5, step=1)
                        whisperx_patience = gr.Slider(label="Patience", info="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search", minimum=0.01, maximum=10, value=1, step=0.01)
                        whisperx_lenght_penalty = gr.Slider(label="Length penalty", info="Optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default", minimum=0.01, maximum=10, value=1, step=0.01)
                    with gr.Column():
                        whisperx_suppress_tokens = gr.Textbox(label="Suppress Tokens", info="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations", value="-1")
                        whisperx_suppress_numerals = gr.Checkbox(label="Suppress Numerals",info="whether to suppress numeric symbols and currency symbols during sampling, since wav2vec2 cannot align them correctly", value=False)
                    with gr.Column():
                        whisperx_initial_prompt = gr.Textbox(label="Initial Prompt",value=None, info="optional text to provide as a prompt for the first window")
                        whisperx_condition_on_previous_text = gr.Checkbox(label="Condition on previous text", info="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop", value=False)
                    with gr.Column():
                        whisperx_temperature_increment_on_fallback = gr.Slider(label="Temperature increment on fallback", info="temperature to increase when falling back when the decoding fails to meet either of the thresholds below", minimum=0, maximum=1, value=0.2, step=0.01)
                        whisperx_compression_ratio_threshold = gr.Slider(label="Compression ratio threshold", info="if the gzip compression ratio is higher than this value, treat the decoding as failed", minimum=0, maximum=10, value=2.4, step=0.01)
                        whisperx_logprob_threshold = gr.Slider(label="Logprob threshold", info="if the average log probability is lower than this value, treat the decoding as failed", minimum=0, maximum=10, value=1, step=0.01)
                        whisperx_no_speech_threshold = gr.Slider(label="No speech threshold", info="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence", minimum=0, maximum=1,step=0.01,value=0.6)
                with gr.Accordion("Subtitles settings"):
                    whisperx_max_line_width = gr.Number(label="Max line width",info="the maximum number of characters in a line before breaking the line", minimum=0, maximum=200, value=None, step=1)
                    whisperx_max_line_count = gr.Number(label="Max line count", info="the maximum number of lines in a segment", minimum=0, maximum=200, value=None, step=1)
        with gr.Column():
            whisperx_status_bar = gr.Label(label=i18n("Status"), visible=True, value="Upload audio and click Transcribe")
            whisperx_subtitles = gr.Files(interactive=False, label=i18n("Subtitles"))
            with gr.Column():
                whisperx_transcribe = gr.Textbox(label=i18n("Transcription"), interactive=False, visible=True, value=None)
                whisperx_segments = gr.Textbox(label=i18n("Segments"), interactive=False, visible=True, value=None)
                whisperx_transcribe_btn = gr.Button(value=i18n("Transcribe"))
            with gr.Column():
                whisperx_diarize_files = gr.Files(interactive=False,visible=False, label=i18n("Speaker diarization files"))
                whisperx_diarize_files_list = gr.Dropdown(visible=False)
                whisperx_diarize_audio_example = gr.Audio(visible=False)
            
        