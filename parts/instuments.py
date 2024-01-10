import gradio as gr
from xtts_webui import *

with gr.Tab("Resemble Enhance"):
    gr.Markdown("")
    with gr.Row():
        with gr.Column():
            with gr.Tab("Single"):
                resemble_audio_single = gr.Audio(
                    label="Single file", value=None)
            with gr.Tab("Batch"):
                resemble_audio_batch = gr.File(
                    file_count="multiple", label="Batch files", file_types=["audio"])
                resemble_audio_batch_path = gr.Textbox(
                    label="Path to folder with audio files (High priority)", value=None)
            resemble_choose_action = gr.Radio(label="Choose action", choices=[
                                              "only_enchance", "only_denoise", "both"], value="both")
            resemble_chunk_seconds = gr.Slider(
                minimum=2, maximum=40, value=8, step=1, label="Chunk seconds (more secods more VRAM usage and faster inference speed)")
            resemble_chunk_overlap = gr.Slider(
                minimum=0.1, maximum=2, value=1, step=0.2, label="Overlap seconds")
            resemble_solver = gr.Dropdown(label="CFM ODE Solver (Midpoint is recommended)", choices=[
                                          "Midpoint", "RK4", "Euler"], value="Midpoint")
            resemble_num_funcs = gr.Slider(minimum=1, maximum=128, value=64, step=1,
                                           label="CFM Number of Function Evaluations (higher values in general yield better quality but may be slower)")
            resemble_temperature = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01,
                                             label="CFM Prior Temperature (higher values can improve quality but can reduce stability)")
            resemble_denoise = gr.Checkbox(
                value=True, label="Denoise Before Enhancement (tick if your audio contains heavy background noise)")
            resemble_output_type = gr.Dropdown(label="Output type", choices=[
                                               "wav", "mp3"], value="wav")
        with gr.Column():
            resemble_status_label = gr.Label(
                value="Upload a file or files and click on the Enhance button")
            result_denoise = gr.Audio(
                label="Denoise", interactive=False, visible=False, value=None)
            result_enhance = gr.Audio(
                label="Result", interactive=False, visible=True, value=None)
            resemble_generate_btn = gr.Button(value="Enhance")
