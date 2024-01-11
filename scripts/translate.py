import shutil
from tqdm import tqdm
import gradio as gr
from datetime import datetime
import translators as ts
from pydub import AudioSegment

import torch
import torchaudio

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from pathlib import Path

import ffmpeg
import time
import os
import argparse

import noisereduce
from pedalboard import Pedalboard, NoiseGate, LowpassFilter, Compressor, LowShelfFilter, Gain
import soundfile as sf

from faster_whisper import WhisperModel
import random


def local_generation(xtts, text, speaker_wav, language, output_file, options={}):
    # Log time
    generate_start_time = time.time()  # Record the start time of loading the model

    gpt_cond_latent, speaker_embedding = xtts.get_or_create_latents(
        "", speaker_wav)

    out = xtts.model.inference(
        text,
        language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=options["temperature"],
        length_penalty=options["length_penalty"],
        repetition_penalty=options["repetition_penalty"],
        top_k=options["top_k"],
        top_p=options["top_p"],
        speed=options["speed"],
        enable_text_splitting=False
    )

    torchaudio.save(output_file, torch.tensor(out["wav"]).unsqueeze(0), 24000)

    generate_end_time = time.time()  # Record the time to generate TTS
    generate_elapsed_time = generate_end_time - generate_start_time

    print(f"TTS generated in {generate_elapsed_time:.2f} seconds")


def combine_wav_files(file_list, output_filename):
    combined = AudioSegment.empty()

    for file in file_list:
        sound = AudioSegment.from_wav(file)
        combined += sound

    combined.export(output_filename, format='wav')


def removeTempFiles(file_list):
    for file in file_list:
        os.remove(file)


def segment_audio(start_time, end_time, input_file, output_file):
    # Cut the segment using ffmpeg and convert it to required format and specifications
    # print(start_time, end_time, input_file, output_file,"segment")
    (
        ffmpeg
        .input(input_file)
        .output(output_file,
                format='wav',
                acodec='pcm_s16le',
                ac=1,
                ar='22050',
                ss=start_time,
                to=end_time)
        .run(capture_stdout=True, capture_stderr=True)
    )


def get_suitable_segment(i, segments):
    min_duration = 5

    if i == 0 or (segments[i].end - segments[i].start) >= min_duration:
        print(f"Segment used N-{i}")
        return segments[i]
    else:
        # Iteration through previous segments starting from the current segment
        for prev_i in range(i - 1, -1, -1):
            print(f"Segment used N-{prev_i}")
            if (segments[prev_i].end - segments[prev_i].start) >= min_duration:
                return segments[prev_i]
        # If no suitable previous one is found,
        # return the current one regardless of its duration.
        return segments[i]


def improve_audio(input_file, output_file):
    # Read audio data with soundfile
    audio_data, sample_rate = sf.read(input_file)

    # Assuming that 'audio_data' contains a NumPy array and 'sample_rate' holds the sampling rate

    # Reduce noise from the audio signal
    reduced_noise = noisereduce.reduce_noise(y=audio_data,
                                             sr=sample_rate,
                                             stationary=True,
                                             prop_decrease=0.75)

    # Create a Pedalboard with effects applied to the audio
    board = Pedalboard([
        NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
        Compressor(threshold_db=12, ratio=2.5),
        LowShelfFilter(cutoff_frequency_hz=400, gain_db=5),
        Gain(gain_db=0)
    ])

    # Process the noise-reduced signal through the pedalboard (effects chain)
    result = board(reduced_noise.astype('float32'), sample_rate)

    # Write processed audio data to output file using soundfile
    sf.write(output_file, result.T if result.ndim > 1 else result, sample_rate)


def clean_temporary_files(file_list, directory):
    """
    Remove temporary files from the specified directory
    """
    for temp_file in file_list:
        try:
            temp_filepath = os.path.join(directory, temp_file)
            os.remove(temp_filepath)
        except FileNotFoundError:
            pass  # If file was already removed or doesn't exist


def create_directory_if_not_exists(directory_path):
    """
    Create a directory if it does not exist
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Folder '{directory_path}' has been successfully established.")
    except OSError as e:
        print(
            f"An error occurred while creating folder '{directory_path}': {e}")

def clean_text(text):
    # Replace Dot to .. 
    text = text.replace(".", "..")
    return text

# Assuming translator, segment_audio, get_suitable_segment,
# local_generation, combine_wav_files are defined elsewhere.
def translate_and_get_voice(this_dir, filename, xtts, mode, whisper_model, source_lang, target_lang,
                            speaker_lang, options={}, text_translator="google", translate_mode=True,
                            output_filename="result.mp3", speaker_wavs=None, improve_audio_func=False, progress=None):

    model_size = whisper_model
    print(f"Loading Whisper model {whisper_model}...")
    model = WhisperModel(whisper_model, device="cuda", compute_type="float16")
    print("Whisper model loaded")

    if source_lang == "auto":
        language_for_whisper = None
    else:
        language_for_whisper = source_lang

    segments, info = model.transcribe(
        filename, language=language_for_whisper, beam_size=5)
    print("Detected language '%s' with probability %f" %
          (info.language, info.language_probability))

    if source_lang == "auto":
        source_lang = info.language

    original_segment_files = []
    translated_segment_files = []

    output_folder = os.path.dirname(output_filename)
    output_folder = Path(output_folder)

    temp_folder_name = Path(
        "./temp") / f'translate_{datetime.now().strftime("%Y%m%d%H%M%S")}'
    temp_folder = temp_folder_name

    create_directory_if_not_exists(temp_folder)

    segments = list(segments)  # Убедитесь, что 'segments' является списком
    total_segments = len(segments)

    # Создаем список пустых элементов размером total_segments
    indices = list(range(total_segments))

    if progress is not None:
        tqdm_object = progress.tqdm(
            indices, total=total_segments, desc="Translate and voiceover...")
    else:
        tqdm_object = tqdm(indices, total=total_segments,
                           desc="Translate and voiceover...")

    for i in tqdm_object:  # Используем значения из indices для прогресса
        segment = segments[i]  # Получаем текущий сегмент используя индекс
        text_to_syntez = ""

        cleared_text = clean_text(segment.text)

        if translate_mode:
            text_to_syntez = ts.translate_text(
                query_text=cleared_text, translator=text_translator, from_language=source_lang, to_language=target_lang)
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {text_to_syntez}")
        else:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {cleared_text}")
            text_to_syntez = cleared_text

        original_audio_segment_file = f"original_segment_{i}.wav"
        original_segment_files.append(original_audio_segment_file)

        # Segment audio according to mode
        mode = int(mode)
        if mode == 1 or mode == 2:
            start_time = segment.start if mode == 1 else get_suitable_segment(
                i, segments).start
            end_time = segment.end if mode == 1 else get_suitable_segment(
                i, segments).end

            # Exporting segmented audio; assuming 'filename' is available here
            output_segment_folder = temp_folder / original_audio_segment_file
            segment_audio(start_time=start_time, end_time=end_time,
                          input_file=filename, output_file=str(output_segment_folder))

        # Prepare TTS input based on provided `mode`
        tts_input_wav = speaker_wavs if (
            mode == 3 and speaker_wavs) else temp_folder / original_audio_segment_file

        synthesized_audio_file = ""

        if translate_mode:
            synthesized_audio_file = f"synthesized_segment_{i}.wav"
        else:
            synthesized_audio_file = f"original_segment_{i}.wav"

        current_datetime = str(datetime.now())
        # Start transformation using TTS system; assuming all required fields are correct
        xtts.local_generation(
            this_dir=this_dir,
            text=text_to_syntez,
            ref_speaker_wav=current_datetime,
            speaker_wav=tts_input_wav,
            language=speaker_lang,
            options=options,
            output_file=temp_folder / synthesized_audio_file
        )

        translated_segment_files.append(synthesized_audio_file)
        # tqdm_object.update(1)

    # tqdm_object.close()
    combined_output_filepath = os.path.join(output_filename)
    combine_wav_files([os.path.join(temp_folder, f)
                      for f in translated_segment_files], combined_output_filepath)

    if improve_audio_func:
        out_dir = os.path.dirname(output_filename)
        improved_audio_path = os.path.join(out_dir, "improved.mp3")
        improve_audio(output_filename, improved_audio_path)
        output_filename = improved_audio_path

    # Optionally remove temporary files after combining them into final output.
    clean_temporary_files(translated_segment_files +
                          (original_segment_files if mode != 3 else []), temp_folder)

    new_file_name = output_folder / Path(output_filename).name
    shutil.move(output_filename, new_file_name)
    return new_file_name
