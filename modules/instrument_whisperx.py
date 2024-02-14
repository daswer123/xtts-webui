from xtts_webui import *
import shutil

from datetime import datetime
from scripts.funcs import resemble_enhance_audio, save_audio_to_wav
import glob
import shutil
from datetime import datetime
from pathlib import Path

from modules.text2voice.voice2voice import select_rvc_model, update_rvc_model, update_openvoice_ref_list
from scripts.translate import translate_and_get_voice,translate_advance_stage1
from scripts.voice2voice import infer_rvc, get_openvoice_refs, infer_rvc_batch, infer_openvoice, find_openvoice_ref_by_name
from scripts.funcs import save_audio_to_wav,save_whisper_audio_to_wav


# Constants
WAV_EXTENSION = "*.wav"
MP3_EXTENSION = "*.mp3"
FLAC_EXTENSION = "*.flac"

DATE_FORMAT = "%Y%m%d_%H%M%S"
SPEAKER_PREFIX = "speaker/"
REFERENCE_KEYWORD = "reference"

from scripts.funcs import write_key_value_to_env
from scripts.transcribe import download_audio,whisperx_work

def find_audio_files(batch_path):
    return glob.glob(os.path.join(batch_path, WAV_EXTENSION)) + \
        glob.glob(os.path.join(batch_path, MP3_EXTENSION)) + \
        glob.glob(os.path.join(batch_path, FLAC_EXTENSION))

def whisperx_transcribe_func(
    # TOKEN
    whisper_hf_token,
    # FILES
    whisperx_audio_single,
    whisperx_youtube_audio,
    whisperx_audio_batch,
    whisperx_audio_batch_path,
    # Main
    whisperx_model,
    # whisperx_task,
    whisperx_language,
    # OPTIONS
    whisperx_align,
    whisperx_timestamp,
    whisperx_timestamp_highlight,
    # DIARIZE MAIN
    whisperx_enable_diarize,
    whisperx_diarize_split,
    whisperx_diarize_speakers_max,
    whisperx_diarize_speakers_min,
    # VAD
    whisperx_vad_onset,
    whisperx_vad_offset,
    whisperx_vad_chunk_size,
    # WHISPERX OPTIONS
    whisperx_compute_type,
    whisperx_device,
    whisperx_batch_size,
    whisperx_beam_size,
    # ADVANCE SETTINGS
    whisperx_device_index,
    whisperx_threads,
    whisperx_align_intropolate,
    # ADVANCE WHISPER SETTINGS 
    whisperx_temperature,
    whisperx_bestof,
    whisperx_patience,
    whisperx_lenght_penalty,
    whisperx_suppress_tokens,
    whisperx_suppress_numerals,
    whisperx_initial_prompt,
    whisperx_condition_on_previous_text,
    whisperx_temperature_increment_on_fallback,
    whisperx_compression_ratio_threshold,
    whisperx_logprob_threshold,
    whisperx_no_speech_threshold,
    # SUBTITLES
    whisperx_max_line_width,
    whisperx_max_line_count
):
    if not (whisperx_audio_single or whisperx_audio_batch or whisperx_audio_batch_path or whisperx_youtube_audio):
        return None, None, None, None,None, "Please Upload Audio"
    
    # VAD OPTIONS
    vad_opitons = {
        "vad_onset": whisperx_vad_onset,
        "vad_offset": whisperx_vad_offset
    }
    
    output_folder = this_dir / OUTPUT_FOLDER
    folder_name = f"whisper_{datetime.now().strftime(DATE_FORMAT)}"
    audio_filename = f"{folder_name}.wav"
    output_folder = output_folder / folder_name
    done_message = ""
    
    os.makedirs(output_folder, exist_ok=True)

    input_file = None
    
    if whisperx_youtube_audio and not(whisperx_audio_single or whisperx_audio_batch or whisperx_audio_batch_path):
        input_file = download_audio(whisperx_youtube_audio, output_folder,audio_filename )
    
    if whisperx_audio_single:
        rate, y = whisperx_audio_single
        input_file = save_whisper_audio_to_wav(rate, y, output_folder)

    audio_files = whisperx_audio_batch or []
    if whisperx_audio_batch_path:
        audio_files.extend(find_audio_files(whisperx_audio_batch_path))

    status_bar = gr.Progress(track_tqdm=True)
    print(input_file,audio_files)
    # Process batches of files
    if audio_files:
        # WORK WITH BATCH
        
        # output_folder = output_folder / folder_name / "temp"
        # os.makedirs(output_folder, exist_ok=True)
        # SOMETHING
        done_message = f"Done, file saved in {folder_name} folder"
        return None, None, None, None,None, "Work in progress"
    if input_file:
        # WORK WITH SINGLE FILE
        outputs = whisperx_work(
            
        )
        return None, None, None, None,None, "Work in progress. ASAP"

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
    # Main
    whisperx_model,
    # whisperx_task,
    whisperx_language,
    # OPTIONS
    whisperx_align,
    whisperx_timestamp,
    whisperx_timestamp_highlight,
    # DIARIZE MAIN
    whisperx_enable_diarize,
    whisperx_diarize_split,
    whisperx_diarize_speakers_max,
    whisperx_diarize_speakers_min,
    # VAD
    whisperx_vad_onset,
    whisperx_vad_offset,
    whisperx_vad_chunk_size,
    # WHISPERX OPTIONS
    whisperx_compute_type,
    whisperx_device,
    whisperx_batch_size,
    whisperx_beam_size,
    # ADVANCE SETTINGS
    whisperx_device_index,
    whisperx_threads,
    whisperx_align_intropolate,
    # ADVANCE WHISPER SETTINGS 
    whisperx_temperature,
    whisperx_bestof,
    whisperx_patience,
    whisperx_lenght_penalty,
    whisperx_suppress_tokens,
    whisperx_suppress_numerals,
    whisperx_initial_prompt,
    whisperx_condition_on_previous_text,
    whisperx_temperature_increment_on_fallback,
    whisperx_compression_ratio_threshold,
    whisperx_logprob_threshold,
    whisperx_no_speech_threshold,
    # SUBTITLES
    whisperx_max_line_width,
    whisperx_max_line_count
    ],
    outputs=[
    whisperx_subtitles,
    whisperx_transcribe,
    whisperx_segments,
    # DIARIZE FILES
    whisperx_diarize_files,
    whisperx_diarize_files_list,
    whisperx_status_bar
    ])

