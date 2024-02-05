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
import pysubs2

import deepl
from xtts_webui import deepl_auth_key_textbox,deepl_api_key


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

def save_subs_and_txt(segments, base_folder, base_name):
    # Сохраняем txt
    txt_output_path = os.path.join(base_folder, base_name + '.txt')
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        for segment in segments:
            f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")

    # Создаем и сохраняем srt & ass subtitles
    subs = pysubs2.SSAFile()

    for segment in segments:
        start_time = int(segment.start * 1000)  # Преобразовываем в миллисекунды для pysubs2
        end_time = int(segment.end * 1000)
        text = segment.text
        subs.events.append(pysubs2.SSAEvent(start=start_time, end=end_time, text=text))

    subs.save(os.path.join(base_folder, base_name + '.srt'))
    subs.save(os.path.join(base_folder, base_name + '.ass'))


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

def transcribe_audio(whisper_model, filename, source_lang='auto'):
    model = WhisperModel(whisper_model, device="cuda", compute_type="float16")
    print("Whisper model loaded")

    language_for_whisper = None if source_lang == "auto" else source_lang

    segments, info = model.transcribe(
        filename, language=language_for_whisper,vad_filter=True, beam_size=5)

    print("Detected language '%s' with probability %f" %
          (info.language, info.language_probability))

    if source_lang == "auto":
        detected_language = info.language   # Сохраняем определенный язык
    else:
        detected_language = source_lang

    return segments, detected_language

def accumulate_segments(segments, start_index, segment_filenames, temp_folder, desired_duration=20):
    accumulated_duration = 0
    accumulated_segment_files = []

    for i in range(start_index, len(segments)):
        if accumulated_duration >= desired_duration:
            break

        segment = segments[i]
        duration_of_segment = segment.end - segment.start
        original_audio_segment_file = temp_folder / segment_filenames[i]

        # Add file name to list of files we'll pass to TTS.
        accumulated_segment_files.append(original_audio_segment_file)

        accumulated_duration += duration_of_segment

    return accumulated_segment_files


# Assuming translator, segment_audio, get_suitable_segment,
# local_generation, combine_wav_files are defined elsewhere.
def translate_and_get_voice(this_dir, filename, xtts, mode, whisper_model, source_lang, target_lang,
                            speaker_lang, options={}, text_translator="google", translate_mode=True,
                            output_filename="result.mp3", speaker_wavs=None, improve_audio_func=False, progress=None):

    # STAGE - 1 TRANSCRIBE
    segments, detected_language = transcribe_audio(whisper_model,filename,source_lang)

    if source_lang == "auto":
        source_lang = detected_language

    output_folder = os.path.dirname(output_filename)
    output_folder = Path(output_folder)

    temp_folder_name = Path(
        "./temp") / f'translate_{datetime.now().strftime("%Y%m%d%H%M%S")}'
    temp_folder = temp_folder_name
    create_directory_if_not_exists(temp_folder)

    # STAGE 1.5 CREATE LIST OF SEGMENTS
    segments = list(segments)  # Убедитесь, что 'segments' является списком
    total_segments = len(segments)
    # Создаем список пустых элементов размером total_segments
    indices = list(range(total_segments))

    # STAGE 2: CUT ALL SEGMENTS
    segment_filenames = []
    mode = int(mode)

    for i in range(total_segments):
        segment = segments[i]

        original_audio_segment_file = f"original_segment_{i}.wav"
        segment_filenames.append(original_audio_segment_file)

        # start_time = segment.start if mode == 1 else get_suitable_segment(i, segments).start
        # end_time = segment.end if mode == 1 else get_suitable_segment(i, segments).end

        start_time = segment.start
        end_time = segment.end

        output_segment_folder = temp_folder / original_audio_segment_file
        segment_audio(start_time=start_time,
                      end_time=end_time,
                      input_file=filename,
                      output_file=str(output_segment_folder))

    # CREATE PROGRESS BAR
    if progress is not None:
        tqdm_object = progress.tqdm(
            indices, total=total_segments, desc="Translate and voiceover...")
    else:
        tqdm_object = tqdm(indices, total=total_segments,
                           desc="Translate and voiceover...")

    # original_segment_files = []
    translated_segment_files = []

    # STAGE 3: VOICEOVER
    for i in tqdm_object:  
        segment = segments[i]
        tts_input_wavs = []  
        text_to_syntez = ""
        
        if text_translator == "deepl":
            api_key = os.getenv("DEEPL_API_KEY")
            print("api key =",api_key)
            if not api_key:
                return
            deepl_translator = deepl.Translator(api_key)

        if translate_mode:
            if text_translator == "deepl":
                if target_lang == "en":
                    target_lang = "en-US"
                    
                text_to_syntez = deepl_translator.translate_text(segment.text,source_lang=source_lang, target_lang=target_lang)
                text_to_syntez = text_to_syntez.text
                print("Deepl:",text_to_syntez)
            else:
              text_to_syntez = ts.translate_text(
                  query_text=segment.text, translator=text_translator, from_language=source_lang, to_language=target_lang)
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {text_to_syntez}")
        else:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            text_to_syntez = clean_text(text_to_syntez)

        synthesized_audio_file = f"synthesized_segment_{i}.wav"
        if mode == 1:
            # Use single file for TTS input as before.
            
            original_audio_segment_file = segment_filenames[i]

            tts_input_wavs.append(temp_folder / original_audio_segment_file)

        elif mode == 2:
            # Accumulate multiple files until reaching at least 20 seconds.

            synthesized_audio_starting_from_i_wav_name_pattern = f"synthesized_segments_starting_from_{i}_combined.wav"

            accumulated_files_for_tts_input=accumulate_segments(
                segments,start_index=i,
                segment_filenames=segment_filenames,
                temp_folder=temp_folder)

            tts_input_combined_path_temporary_output_filename=temp_folder/synthesized_audio_starting_from_i_wav_name_pattern

            tts_input_wavs = accumulated_files_for_tts_input

            # if len(accumulated_files_for_tts_input) > 1:
            #     # If more than one file was found, combine them into one WAV file for TTS processing.
            #     combine_wav_files([os.path.join(temp_folder,f)
            #         for f in accumulated_files_for_tts_input],
            #         str(tts_input_combined_path_temporary_output_filename))

            #     tts_input_wavs.append(tts_input_combined_path_temporary_output_filename)

            # elif len(accumulated_files_for_tts_input) ==1:
            #     # If only one file was found (less than 20 seconds), use it directly.
            #     tts_input_wavs.append(os.path.join(temp_folder,
            #                                        accumulated_files_for_tts_input[0]))

        current_datetime = str(datetime.now())
        # Start transformation using TTS system; assuming all required fields are correct
        print(tts_input_wavs)
        xtts.local_generation(
            this_dir=this_dir,
            text=text_to_syntez,
            ref_speaker_wav=current_datetime,
            speaker_wav=tts_input_wavs,
            language=speaker_lang,
            options=options,
            output_file=temp_folder / synthesized_audio_file
        )

        translated_segment_files.append(synthesized_audio_file)

    # Обрабатываем output_filename и создаем необходимую структуру каталога
    base_name_with_ext = os.path.basename(output_filename)  # Имя файла с расширением
    base_directory = os.path.dirname(output_filename)  # Директория файла

    # Определяем базовое имя без расширения
    base_name_without_ext = os.path.splitext(base_name_with_ext)[0]

    # Создаем целевую папку если она ещё не существует
    target_folder_path = os.path.join(base_directory, base_name_without_ext)
    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)

    # Теперь выполним функцию сохранения всего: .txt, .srt и .ass файлов.
    save_subs_and_txt(segments, target_folder_path, base_name_without_ext)

    combined_audio_output_filepath = os.path.join(target_folder_path, base_name_with_ext)

    # Передаем корректный путь для объединенного wav файла - уже внутри целевой директории.
    combine_wav_files([os.path.join(temp_folder,f) for f in translated_segment_files], combined_audio_output_filepath)

    # Optionally remove temporary files after combining them into final output.
    clean_temporary_files(translated_segment_files +
                          (segment_filenames if mode != 3 else []), temp_folder)

    new_file_name = output_folder / Path(target_folder_path) / Path(output_filename).name
    # print(new_file_name)
    # shutil.move(output_filename, new_file_name)
    return new_file_name
