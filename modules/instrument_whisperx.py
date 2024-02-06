from xtts_webui import *
import shutil

from datetime import datetime
from scripts.funcs import resemble_enhance_audio, save_audio_to_wav
import glob

from scripts.funcs import write_key_value_to_env

def whisperx_transcribe_func(
    # TOKEN
    whisper_hf_token,
    # FILES
    whisperx_audio_single,
    whisperx_youtube_audio,
    whisperx_audio_batch,
    whisperx_audio_batch_path,
    # WHISPER SETTINGS
    whisperx_model,
    whisperx_compute_type,
    whisperx_device,
    whisperx_batch_size,
    # MAIN SETTINGS
    whisperx_task,
    whisperx_language,
    whisperx_align,
    whisperx_timestamp,
    whisperx_timestamp_highlight,
    # VAD OPTIONS
    whisperx_vad_onset,
    whisperx_vad_offset,
    # DIARIZE
    whisperx_enable_diarize,
    whisperx_diarize_split,
    whisperx_diarize_speakers,
    # DIARIZE SETTINGS
    whisperx_diarize_speakers_max,
    whisperx_diarize_speakers_min
):
    return

def write_hf_token(whisper_hf_token):
    write_key_value_to_env("HF_TOKEN",whisper_hf_token)

whisper_hf_token.change(fn=write_hf_token,inputs=[whisper_hf_token])

whisperx_transcribe_btn.click(
    fn=whisperx_transcribe_func,
    inputs=[
    # TOKEN
    whisper_hf_token,
    # FILES
    whisperx_audio_single,
    whisperx_youtube_audio,
    whisperx_audio_batch,
    whisperx_audio_batch_path,
    # WHISPER SETTINGS
    whisperx_model,
    whisperx_compute_type,
    whisperx_device,
    whisperx_batch_size,
    # MAIN SETTINGS
    whisperx_task,
    whisperx_language,
    whisperx_align,
    whisperx_timestamp,
    whisperx_timestamp_highlight,
    # VAD OPTIONS
    whisperx_vad_onset,
    whisperx_vad_offset,
    # DIARIZE
    whisperx_enable_diarize,
    whisperx_diarize_split,
    whisperx_diarize_speakers,
    # DIARIZE SETTINGS
    whisperx_diarize_speakers_max,
    whisperx_diarize_speakers_min
    ],
    outputs=[
        whisperx_subtitles,
        whisperx_transcribe,
        whisperx_segments,
        # DIARIZE FILES
        whisperx_diarize_files,
        whisperx_diarize_files_list
    ])
