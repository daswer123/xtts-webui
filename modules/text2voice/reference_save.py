import os

import gradio as gr
from pathlib import Path
from scripts.funcs import move_and_rename_file, str_to_list

from xtts_webui import *

# SAVE FUNCS


def save_speaker(speaker_wav_save_name, speaker_path_text, ref_speaker_list, speaker_ref_wavs):
    move_and_rename_file(
        speaker_path_text, XTTS.speaker_folder, speaker_wav_save_name)

    speakers_list = XTTS.get_speakers()
    speaker_value = speaker_wav_save_name
    speaker_value_text = speaker_wav_save_name

    if speaker_ref_wavs:
        speakers_list.append("multi_reference")

    return None, "", "", speaker_wav_save_name, gr.Dropdown(label="Reference Speaker from folder 'speakers'", value=speaker_value, choices=speakers_list, allow_custom_value=True), gr.Button(visible=False)


def save_speaker_multiple(speaker_wav_save_name, ref_speaker_list, ref_speakers, speaker_ref_wavs, speaker_path_text):
    new_ref_speaker_list = str_to_list(speaker_ref_wavs)
    speaker_dir = Path(XTTS.speaker_folder) / speaker_wav_save_name

    for file in new_ref_speaker_list:
        print(file)
        file = Path(file)
        move_and_rename_file(file, speaker_dir, os.path.basename(file))

    speakers_list = XTTS.get_speakers()
    speaker_value = speaker_wav_save_name
    speaker_value_text = speaker_wav_save_name

    if speaker_path_text:
        speakers_list.append("reference")

    return "", speaker_value, None, "", gr.Dropdown(label="Reference Speaker from folder 'speakers'", value=speaker_value, choices=speakers_list, allow_custom_value=True), gr.Button(visible=False)


# SAVE HANDLERS

save_speaker_btn.click(fn=save_speaker,
                       inputs=[speaker_wav_save_name, speaker_path_text,
                               ref_speaker_list, speaker_ref_wavs],
                       outputs=[ref_speaker, speaker_wav_save_name, speaker_path_text, speaker_value_text, ref_speaker_list, save_speaker_btn])

save_multiple_speaker_btn.click(fn=save_speaker_multiple,
                                inputs=[speaker_wav_save_name, ref_speaker_list,
                                        ref_speakers, speaker_ref_wavs, speaker_path_text],
                                outputs=[speaker_wav_save_name, speaker_value_text, ref_speakers, speaker_ref_wavs, ref_speaker_list])
