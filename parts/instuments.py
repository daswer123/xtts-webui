import gradio as gr
from scripts.languages import get_language_names
from xtts_webui import *

from i18n.i18n import I18nAuto
from scripts.funcs import read_key_from_env
i18n = I18nAuto()

default_prompt = "Your task is to analyse the subtitles given to you and correct them so that they better fit the context. You must answer in the same language as the subtitles. You must answer in the same format in which the subtitles were given to you. Answer only with the corrected text, without any additional comments."

gpt_4_key = read_key_from_env("GPT_4_KEY")
claude_3_key = read_key_from_env("CLAUDE_3_KEY")


avalible_langiage = get_language_names()
avalible_langiage = ["auto"] + avalible_langiage

with gr.Tab(i18n("Resemble Enhance")):
    gr.Markdown("")
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

with gr.Tab(i18n("Subtitle tools"), render=False):
    with gr.Tab(i18n("Get Subtitle")):
        with gr.Row():
            with gr.Column():
                with gr.Tab(i18n("Single")):
                    subtitle_file_get = gr.Audio(label=i18n("Audo file"))
                    subtitle_file_batch_get = gr.Textbox(visible=False)
                    subtitle_file_batch_get_path = gr.Textbox(visible=False)
                # with gr.Tab(i18n("Batch")):
                #     subtitle_file_batch_get = gr.File(
                #         file_count="multiple", label=i18n("Batch files"))
                #     subtitle_file_batch_get_path = gr.Textbox(
                #         label=i18n("Path to folder with audio files"), value=None)
                    
                get_sub_whisper_model = gr.Dropdown(label=i18n("Whisper Model"), choices=["small", "medium", "large-v2", "large-v3"], value="medium")
                get_sub_source_lang = gr.Dropdown(choices=avalible_langiage, value="auto",label=i18n("Subtitle Target lang"))
                
                with gr.Accordion(label=i18n("Subtitle settings"), open=True):
                    with gr.Row():
                        get_sub_max_width_sub_v2v = gr.Slider(label=i18n("Max width per line"), minimum=1, maximum=100, value=40, step=1)
                        get_sub_max_line_sub_v2v = gr.Slider(label=i18n("Max line"),visible=True, minimum=1, maximum=20, value=2, step=1)
                    with gr.Row():
                        get_sub_highlight_words_v2v = gr.Checkbox(label=i18n("Highlight Words"),visible=False, value=False)
                
                with gr.Accordion(label="Translation settings", open=False,visible=False):
                    get_sub_translator = gr.Radio(label=i18n("Translator"), choices=[
                                                "google", "bing", "baidu","deepl"], value="google")
                    deepl_auth_key_textbox = gr.Textbox(label="Deepl API Key", value="",type="password",visible=False)
                
                with gr.Accordion("Whisper settings",open=False):
                        with gr.Row():
                            get_sub_whisper_compute_time = gr.Radio(label="Compute Type",choices=["int8","float16"],value="float16",info="change to 'int8' if low on GPU mem (may reduce accuracy)")
                            get_sub_whisper_device = gr.Radio(label="Device",choices=["cuda","cpu"],value="cuda",info="change to 'int8' if low on GPU mem (may reduce accuracy)")
                        with gr.Row():
                            get_sub_whisper_batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=32, value=8, step=1,info="reduce if low on GPU mem")
                            get_sub_whisper_aline = gr.Checkbox(value=True,label="Align whisper output")
                    
            with gr.Column():
                get_sub_status_label = gr.Label(value="Status")
                subtitle_output_get = gr.Files(
                    label=i18n("Result"), interactive=False, value=None)
                subtitle_get_btn = gr.Button(value=i18n("Get Subtitles"))
        
    with gr.Tab(i18n("Correct")):
        with gr.Row():
            with gr.Column():
                with gr.Tab(i18n("Single")):
                    subtitle_file_correct = gr.File(label=i18n("Subtitle file, srt"),file_types=["srt,ass,txt"])
                with gr.Tab(i18n("Batch")):
                    subtitle_file_batch_correct = gr.File(
                        file_count="multiple", label=i18n("Batch files"), file_types=["srt,ass,txt"])
                    subtitle_file_batch_correct_path = gr.Textbox(
                        label=i18n("Path to folder with subtitle files"), value=None)
                    
                subtitle_correct_prompt = gr.TextArea(interactive=True,label=i18n("Prompt"),value=default_prompt)
                subbtitle_correct_type = gr.Radio(label=i18n("Choose what model you will use"),choices=["Claude 3 Opus","Claude 3 Sonet", "GPT-4","GPT-4-Turbo"], value="Claude 3 Opus")
                subtitle_api_key_claude = gr.Textbox(label=i18n("API Key Claude"),type="password",interactive=True, value=claude_3_key)
                subtitle_api_key_gpt = gr.Textbox(label=i18n("API Key GPT"), type="password",interactive=True, value=gpt_4_key, visible=False)
            with gr.Column():
                subtitle_output_correct = gr.Files(
                    label=i18n("Result"), interactive=False, value=None)
                subtitle_correct_btn = gr.Button(value=i18n("Get Subtitles"))
    with gr.Tab(i18n("Edit subs")):
      gr.Markdown(i18n("Subtitle Cut"))
      with gr.Row():
        with gr.Column():
            subtitle_file = gr.File(label=i18n("Subtitle file, srt"))
            subtitle_max_width = gr.Slider(minimum=0, maximum=100, value=40, step=1, label=i18n("Max width per line"))
        with gr.Column():
            subtitle_output = gr.Files(
                label=i18n("Result"), value=None)
            subtitle_generate_btn = gr.Button(value=i18n("Change"))



# subtitle_get_btn.click(fn=get_subs)
            
# from scripts.funcs import read_key_from_env
# def switch_deepl_key(get_sub_translator):
#     deepl_key = read_key_from_env("DEEPL_API_KEY")
#     if get_sub_translator == "deepl":
#         return gr.Textbox(label="Deepl API Key", value=deepl_key,type="password",visible=True)
    
#     return gr.Textbox(label="Deepl API Key", value=deepl_key,type="password",visible=False)

# get_sub_translator.change(fn=switch_deepl_key,inputs=[get_sub_translator,deepl_auth_key_textbox],outputs=[deepl_auth_key_textbox])