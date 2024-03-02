
import os
import langid

import gradio as gr
from pathlib import Path
from scripts.funcs import improve_and_convert_audio, resemble_enhance_audio, str_to_list
from scripts.voice2voice import infer_rvc, infer_openvoice, find_openvoice_ref_by_name

import uuid

from xtts_webui import *
import shutil
from datetime import datetime

import re

import ffmpeg
from pathlib import Path
import shutil

def concatenate_audios(folder_path, output_filename="result.wav"):
    # Получаем список всех .wav файлов в папке.
    print(folder_path)
    audio_files_paths = sorted(Path(folder_path).glob("*.wav"))

    # Убеждаемся, что есть хотя бы два файла для объединения.
    if len(audio_files_paths) < 2:
        print("Нужно как минимум два аудиофайла для объединения.")
        return

    # Создаем комплексный фильтр для соединения аудиофайлов.
    inputs = []
    for audio_file in audio_files_paths:
        inputs.append(ffmpeg.input(str(audio_file)))

    # Используем фильтр concat с параметром 'n' равным количеству файлов и 'a' показывающим,
    # что это аудиофайлы ('v=0' означало бы видео).
    output_filename_path = os.path.join(folder_path,output_filename)
    merged_audio = ffmpeg.concat(*inputs, v=0, a=1).output(output_filename_path)

    # Запускаем процесс и сохраняем результат в файл result.wav.
    merged_audio.run()


def move_and_replace(original_file_path):
    not_sync_dir = original_file_path.parent / "not_sync"
    not_sync_dir.mkdir(exist_ok=True)

    # Строим имя временного файла из оригинального.
    temp_filename = f"temp_{original_file_path.stem}{original_file_path.suffix}"
    temp_output_filepath = original_file_path.parent / temp_filename

    if original_file_path.exists():
        # Путь к оригинальному файлу в папке 'not_sync'.
        new_location = not_sync_dir / original_file_path.name
        shutil.move(str(original_file_path), str(new_location))  # Перемещаем оригинальный файл.

    if temp_output_filepath.exists():
        # Если временный файл существует, переименовываем его для замены оригинала.
        temp_output_filepath.rename(original_file_path)



def get_audio_duration(file_path):
    probe = ffmpeg.probe(file_path)
    duration = float(probe['format']['duration'])
    return duration

def apply_atempo_filter(input_filepath, output_filepath, speed_factor):
    # Создаем комплексный фильтр для atempo если это требуется.
    filters = []
    while speed_factor > 2:
        filters.append('atempo=2.0')
        speed_factor /= 2
    while speed_factor < 0.5:
        filters.append('atempo=0.5')
        speed_factor *= 2

    # Добавляем последнее значение atempo к фильтрам.
    filters.append(f'atempo={speed_factor}')

    filter_str = ','.join(filters)

    # Применяем фильтры и записываем вывод в файл.
    (
        ffmpeg.input(str(input_filepath))
             .filter_('atempo', f'"{filter_str}"')
             .output(str(output_filepath))
             .run(overwrite_output=True)
    )

def adjust_audio_speed(input_file_path, target_duration):
    actual_duration = get_audio_duration(input_file_path)
    speed_factor = actual_duration / target_duration

    # Указываем путь к временному файлу.
    temp_output_filepath = input_file_path.with_name(f"temp_{input_file_path.name}")

    if 0.5 <= speed_factor <= 2:
        (
            ffmpeg
            .input(str(input_file_path))
            .filter('atempo', speed_factor)
            .output(str(temp_output_filepath))
            .run(overwrite_output=True)
        )
        # move_and_replace(input_file_path)  # Вызывается с одним аргументом.

    else:
         filters_chain = []
         while speed_factor > 2 or speed_factor < 0.5:
             if speed_factor > 2:
                 filters_chain.append('atempo=2.0')
                 speed_factor /= 2
             elif speed_factor < 0.5:
                 filters_chain.append('atempo=0.5')
                 speed_factor *= 2

         filters_chain.append(f'atempo={speed_factor}')

         stream = ffmpeg.input(str(input_file_path))
         for filter_ in filters_chain:
             stream = stream.filter('atempo', filter_.split('=')[1])

         stream.output(str(temp_output_filepath)).run(overwrite_output=True)

    # move_and_replace(temp_output_filepath)   # Теперь передаем сформированный путь к временному файлу.


# Не забудьте также обновить функцию move_and_replace, чтобы она могла принимать два параметра.

# def move_and_replace(original_file_path, temp_output_filepath):
#      not_sync_dir = original_file_path.parent / "not_sync"
#      not_sync_dir.mkdir(exist_ok=True)

#      new_location = not_sync_dir / original_file_path.name
#      shutil.move(str(original_file_path), str(new_location))  # Перемещаем оригинальный файл

#      temp_output_filepath.replace(original_file_path)  # Заменяем оригинальный новым

        

# HELP FUNCS

def extract_text_from_srt(content):
    # Define a regex pattern to match the timecodes and text
    pattern = re.compile(r'\d+\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\Z)', re.DOTALL)
    entries = re.findall(pattern, content)

    return [(start.replace(',', '_'), end.replace(',', '_'), text.replace('\n', ' ')) for start, end, text in entries]

def save_lines_to_files(subtitle_entries, base_dir_path,subtitle_file):
    # Create temp directory using subtitle file name (without extension)
    temp_dir_path = base_dir_path / subtitle_file.stem
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for start_timecode, end_timecode, text in subtitle_entries:
        # Replace colons and commas with underscores so it's a valid filename
        safe_start_timecode = start_timecode.replace(':', '_').replace(',', '_')
        safe_end_timecode = end_timecode.replace(':', '_').replace(',', '_')

        filename = f"{safe_start_timecode}-{safe_end_timecode}.txt"
        filepath = temp_dir_path / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)

        saved_paths.append(str(filepath))

    return saved_paths



def predict_lang(text, selected_lang):
    # strip need as there is space at end!
    language_predicted = langid.classify(text)[0].strip()

    # tts expects chinese as zh-cn
    if language_predicted == "zh":
        # we use zh-cn
        language_predicted = "zh-cn"

    # Check if language in supported langs
    if language_predicted not in supported_languages:
        language_predicted = selected_lang
        logger.warning(
            f"Language {language_predicted} not supported, using {supported_languages[selected_lang]}")

    return language_predicted

# GENERATION AND GENERATION OPTIONS


def switch_waveform(enable_waveform, video_gr):
    if enable_waveform:
        return gr.Video(label="Waveform Visual", visible=True, interactive=False)
    else:
        return gr.Video(label="Waveform Visual", interactive=False, visible=False)

def generate_audio(
    # Resemble enhance Settings
    enhance_resemble_chunk_seconds,
    enhance_resemble_chunk_overlap,
    enhance_resemble_solver,
    enhance_resemble_num_funcs,
    enhance_resemble_temperature,
    enhance_resemble_denoise,
    # RVC settings
    rvc_settings_model_path,
    rvc_settings_index_path,
    rvc_settings_model_name,
    rvc_settings_pitch,
    rvc_settings_index_rate,
    rvc_settings_protect_voiceless,
    rvc_settings_method,
    rvc_settings_filter_radius,
    rvc_settings_resemple_rate,
    rvc_settings_envelope_mix,
    # OpenVoice Setting
    opvoice_ref_list,
    # Batch
    batch_generation,
    batch_generation_path,
    ref_speakers,
    # Batch Subtitles
    batch_sub_generation,
    batch_sub_generation_path,
    sync_sub_generation,
    # Features
    language_auto_detect,
    enable_waveform,
    improve_output_audio,
    improve_output_resemble,
    improve_output_voice2voice,
    #  Default settings
    output_type,
    text,
    languages,
    # Help variables
    speaker_value_text,
    speaker_path_text,
    additional_text,
    # TTS settings
    temperature, length_penalty,
    repetition_penalty,
    top_k,
    top_p,
    speed,
    sentence_split,
    # STATUS
        status_bar):

    resemble_enhance_settings = {
        "chunk_seconds": enhance_resemble_chunk_seconds,
        "chunks_overlap": enhance_resemble_chunk_overlap,
        "solver": enhance_resemble_solver,
        "nfe": enhance_resemble_num_funcs,
        "tau": enhance_resemble_temperature,
        "denoising": enhance_resemble_denoise,
        "use_enhance": improve_output_resemble
    }

    ref_speaker_wav = ""

    print("Using ready reference")
    ref_speaker_wav = speaker_value_text

    if speaker_path_text and speaker_value_text == "reference":
        print("Using folder reference")
        ref_speaker_wav = speaker_path_text

    if ref_speakers and speaker_value_text == "multi_reference":
        print("Using multiple reference")
        ref_speakers_list = str_to_list(ref_speakers)
        ref_speaker_wav = Path(ref_speakers_list[0]).parent.absolute()
        ref_speaker_wav = str(ref_speaker_wav)

    lang_code = reversed_supported_languages[languages]

    if language_auto_detect:
        lang_code = predict_lang(text, lang_code)

    options = {
        "temperature": temperature,
        "length_penalty": length_penalty,
        "repetition_penalty": float(repetition_penalty),
        "top_k": top_k,
        "top_p": top_p,
        "speed": speed,
        "sentence_split": sentence_split,
    }

    status_bar = gr.Progress(track_tqdm=True)
    
    if batch_sub_generation_path and Path(batch_sub_generation_path).exists():
        # Search for both .srt and .ass files within the directory
        batch_sub_generation = [f for f in Path(batch_sub_generation_path).glob('*.srt')]
        batch_sub_generation += [f for f in Path(batch_sub_generation_path).glob('*.ass')]

    # Find all .txt files in the filder and write this to batch_generation
    if batch_generation_path and Path(batch_generation_path).exists():
        batch_generation = [f for f in Path(
            batch_generation_path).glob('*.txt')]
    
    # batch_sub_generation,
    # batch_sub_generation_path,
    print(batch_sub_generation,batch_sub_generation_path,"test")
    
    new_batch_sub = {}
    saved_files = []
    if batch_sub_generation:
        test_folder = Path(f"temp/sub_batch_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
        batch_generation = []
        for subtitle_file in batch_sub_generation:
            with open(subtitle_file, 'r', encoding='utf-8') as file:
                content = file.read()
                subtitle_file = Path(subtitle_file)

                if subtitle_file.suffix == '.srt':
                    # Process .srt subtitles here...
                    extracted_texts = extract_text_from_srt(content)

                elif subtitle_file.suffix == '.ass':
                    # Process .ass subtitles here...
                    extracted_texts = extract_text_from_srt(content)
                    # pass  # You would implement similar logic for .ass files
                
                saved_files = save_lines_to_files(extracted_texts,test_folder,subtitle_file)
                
                new_batch_sub[subtitle_file.stem] = saved_files
                
        print(new_batch_sub,"test")
        # if status_bar is not None:
        #     tqdm_object = status_bar.tqdm(batch_sub_generation, desc="Generate...")
        # else:
        #     tqdm_object = tqdm(batch_sub_generation)
    
    if new_batch_sub:
        final_output_folders = []
        if status_bar is not None:
            tqdm_object = status_bar.tqdm(new_batch_sub.keys(), desc="Generate...")
        else:
            tqdm_object = tqdm(new_batch_sub.keys())

        batch_all_dirname = f"output/subs_"+datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(batch_all_dirname, exist_ok=True)
        status_message = f"Done, generation subtitles saved in {batch_all_dirname}"

        for sub in tqdm_object:
          files = new_batch_sub[sub]
          sub_dirname = os.path.join(batch_all_dirname,f"sub_"+sub)
          
          final_output_folders.append(sub_dirname)
          
          os.makedirs(sub_dirname, exist_ok=True)
          for file_path in files:
            with open(file_path, encoding="utf-8", mode="r") as f:
                text = f.read()

                if language_auto_detect:
                    lang_code = predict_lang(text, lang_code)

                filename = os.path.basename(file_path)
                filename = filename.split(".")[0]

                output_file_path = f"{filename}_{additional_text}_{speaker_value_text}.{output_type}"
                output_file = XTTS.process_tts_to_file(
                    this_dir,text, lang_code, ref_speaker_wav, options, output_file_path)

                if improve_output_audio:
                    output_file = improve_and_convert_audio(
                        output_file, output_type)

                if improve_output_voice2voice == "RVC" and not rvc_settings_model_path:
                    status_message += "\nPlease select RVC model before generate"

                if improve_output_voice2voice == "RVC" and rvc_settings_model_path:
                    temp_dir = this_dir / "output"
                    result = temp_dir / \
                        f"{speaker_value_text}_{rvc_settings_model_name}_{Path(file_path).stem}.{output_type}"
                    infer_rvc(rvc_settings_pitch,
                              rvc_settings_index_rate,
                              rvc_settings_protect_voiceless,
                              rvc_settings_method,
                              rvc_settings_model_path,
                              rvc_settings_index_path,
                              output_file,
                              result,
                              filter_radius=rvc_settings_filter_radius,
                              resemple_rate=rvc_settings_resemple_rate,
                              envelope_mix=rvc_settings_envelope_mix
                              )
                    output_file = result.absolute()

                if improve_output_voice2voice == "OpenVoice" and opvoice_ref_list != "None":
                    temp_dir = this_dir / "output"
                    result = temp_dir / \
                        f"{speaker_value_text}_tuned_{filename}.{output_type}"
                    allow_infer = True

                    if (len(text) < 150):
                        allow_infer = False
                        status_message += "\nYour text should be longer, more than 150 symblos to use OpenVoice Tunner"

                    # Initialize ref_path to None to ensure it has a value in all branches
                    ref_opvoice_path = None

                    # Check if starts with "speaker/"
                    if opvoice_ref_list.startswith("speaker/"):
                        speaker_wav = opvoice_ref_list.split("/")[-1]

                        if speaker_wav == "reference" and speaker_path_text:
                            ref_opvoice_path = speaker_path_text
                        else:
                            ref_opvoice_path = XTTS.get_speaker_path(
                                speaker_wav)
                            if type(ref_opvoice_path) == list:
                                ref_opvoice_path = ref_opvoice_path[0]

                        if speaker_wav == "reference" and not speaker_path_text:
                             allow_infer = False
                             status_message += "\nReference for OpenVoice not found, Skip tunning"
                             print("Referenc not found, Skip")
                    else:
                        ref_opvoice_path = find_openvoice_ref_by_name(
                            this_dir, opvoice_ref_list)

                    if allow_infer:
                        infer_openvoice(
                            input_path=output_file, ref_path=ref_opvoice_path, output_path=result)

                        # Update the output_file with the absolute path to the result
                        output_file = result.absolute()

                if improve_output_resemble:
                    output_file = resemble_enhance_audio(
                        **resemble_enhance_settings, audio_path=output_file, output_type=output_type)[1]

                new_output_file = os.path.join(
                    sub_dirname, os.path.basename(output_file_path))
                shutil.move(output_file, new_output_file)
                output_file = new_output_file
        if True:
          for sub_folder in final_output_folders:
              audio_files_paths = list(Path(sub_folder).glob("*.wav"))
              for audio_file in audio_files_paths:
                  filename_without_ext = audio_file.stem
          
                  # Расчет продолжительности тайминга на основе имени файла. Нужно добавить соответствующую логику.
                  timing_parts= filename_without_ext.split('-')
                  start_time_str, end_time_str= timing_parts[0], timing_parts[1]
          
                   # Преобразование строковых таймингов в секунды.
                  hh_mm_ss_to_seconds = lambda t: sum(x * int(t) for x, t in zip([3600, 60, 1], t.split('_')))
          
                  start_seconds=hh_mm_ss_to_seconds(start_time_str)
                  end_seconds=hh_mm_ss_to_seconds(end_time_str)
          
                  time_difference=end_seconds-start_seconds
          
                  adjust_audio_speed(audio_file,time_difference)
                  move_and_replace(audio_file)
        
        for sub_folder in final_output_folders:
            concatenate_audios(sub_folder)
            if sync_sub_generation:
                no_sync_path = os.path.join(sub_folder,"not_sync")
                concatenate_audios(no_sync_path)
                
        return None, None, status_message
        # if enable_waveform:
        #     return gr.make_waveform(audio=output_file), output_file, status_message
        # else:
        #     return None, output_file, status_message
    
            
    if batch_generation:
        if status_bar is not None:
            tqdm_object = status_bar.tqdm(batch_generation, desc="Generate...")
        else:
            tqdm_object = tqdm(batch_generation)

        batch_dirname = f"output/batch_"+datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(batch_dirname, exist_ok=True)
        status_message = f"Done, generation saved in {batch_dirname}"

        for file_path in tqdm_object:
            with open(file_path, encoding="utf-8", mode="r") as f:
                text = f.read()

                if language_auto_detect:
                    lang_code = predict_lang(text, lang_code)

                filename = os.path.basename(file_path)
                filename = filename.split(".")[0]

                output_file_path = f"{filename}_{additional_text}_{speaker_value_text}.{output_type}"
                output_file = XTTS.process_tts_to_file(
                    this_dir,text, lang_code, ref_speaker_wav, options, output_file_path)

                if improve_output_audio:
                    output_file = improve_and_convert_audio(
                        output_file, output_type)

                if improve_output_voice2voice == "RVC" and not rvc_settings_model_path:
                    status_message += "\nPlease select RVC model before generate"

                if improve_output_voice2voice == "RVC" and rvc_settings_model_path:
                    temp_dir = this_dir / "output"
                    result = temp_dir / \
                        f"{speaker_value_text}_{rvc_settings_model_name}_{Path(file_path).stem}.{output_type}"
                    infer_rvc(rvc_settings_pitch,
                              rvc_settings_index_rate,
                              rvc_settings_protect_voiceless,
                              rvc_settings_method,
                              rvc_settings_model_path,
                              rvc_settings_index_path,
                              output_file,
                              result,
                              filter_radius=rvc_settings_filter_radius,
                              resemple_rate=rvc_settings_resemple_rate,
                              envelope_mix=rvc_settings_envelope_mix
                              )
                    output_file = result.absolute()

                if improve_output_voice2voice == "OpenVoice" and opvoice_ref_list != "None":
                    temp_dir = this_dir / "output"
                    result = temp_dir / \
                        f"{speaker_value_text}_tuned_{filename}.{output_type}"
                    allow_infer = True

                    if (len(text) < 150):
                        allow_infer = False
                        status_message += "\nYour text should be longer, more than 150 symblos to use OpenVoice Tunner"

                    # Initialize ref_path to None to ensure it has a value in all branches
                    ref_opvoice_path = None

                    # Check if starts with "speaker/"
                    if opvoice_ref_list.startswith("speaker/"):
                        speaker_wav = opvoice_ref_list.split("/")[-1]

                        if speaker_wav == "reference" and speaker_path_text:
                            ref_opvoice_path = speaker_path_text
                        else:
                            ref_opvoice_path = XTTS.get_speaker_path(
                                speaker_wav)
                            if type(ref_opvoice_path) == list:
                                ref_opvoice_path = ref_opvoice_path[0]

                        if speaker_wav == "reference" and not speaker_path_text:
                             allow_infer = False
                             status_message += "\nReference for OpenVoice not found, Skip tunning"
                             print("Referenc not found, Skip")
                    else:
                        ref_opvoice_path = find_openvoice_ref_by_name(
                            this_dir, opvoice_ref_list)

                    if allow_infer:
                        infer_openvoice(
                            input_path=output_file, ref_path=ref_opvoice_path, output_path=result)

                        # Update the output_file with the absolute path to the result
                        output_file = result.absolute()

                if improve_output_resemble:
                    output_file = resemble_enhance_audio(
                        **resemble_enhance_settings, audio_path=output_file, output_type=output_type)[1]

                new_output_file = os.path.join(
                    batch_dirname, os.path.basename(output_file_path))
                shutil.move(output_file, new_output_file)
                output_file = new_output_file

        if enable_waveform:
            return gr.make_waveform(audio=output_file), output_file, status_message
        else:
            return None, output_file, status_message

    # Check if the file already exists, if yes, add a number to the filename
    count = 1
    output_file_path = f"{additional_text}_({count})_{speaker_value_text}.{output_type}"
    while os.path.exists(os.path.join('output', output_file_path)):
        count += 1
        output_file_path = f"{additional_text}_({count})_{speaker_value_text}.{output_type}"

    status_message = "Done"
    # Perform TTS and save to the generated filename
    output_file = XTTS.process_tts_to_file(
       this_dir, text, lang_code, ref_speaker_wav, options, output_file_path)

    if improve_output_audio:
        output_file = improve_and_convert_audio(output_file, output_type)

    if improve_output_voice2voice == "RVC" and not rvc_settings_model_path:
        status_message = "Please select RVC model before generate"

    if improve_output_voice2voice == "RVC" and rvc_settings_model_path:
        temp_dir = this_dir / "output"
        result = temp_dir / \
            f"{speaker_value_text}_{rvc_settings_model_name}_{count}.{output_type}"
        infer_rvc(rvc_settings_pitch,
                  rvc_settings_index_rate,
                  rvc_settings_protect_voiceless,
                  rvc_settings_method,
                  rvc_settings_model_path,
                  rvc_settings_index_path,
                  output_file,
                  result,
                  filter_radius=rvc_settings_filter_radius,
                  resemple_rate=rvc_settings_resemple_rate,
                  envelope_mix=rvc_settings_envelope_mix
                  )
        output_file = result.absolute()

    if improve_output_voice2voice == "OpenVoice" and opvoice_ref_list != "None":
        temp_dir = this_dir / "output"
        result = temp_dir / f"{speaker_value_text}_tuned_{count}.{output_type}"
        allow_infer = True

        # Initialize ref_path to None to ensure it has a value in all branches
        ref_opvoice_path = None

        if (len(text) < 150):
            allow_infer = False
            status_message = "Your text should be longer, more than 150 symblos to use OpenVoice Tunner"

        # Check if starts with "speaker/"
        if opvoice_ref_list.startswith("speaker/"):
            speaker_wav = opvoice_ref_list.split("/")[-1]

            if speaker_wav == "reference" and speaker_path_text:
                ref_opvoice_path = speaker_path_text
            else:
                ref_opvoice_path = XTTS.get_speaker_path(speaker_wav)
                if type(ref_opvoice_path) == list:
                    ref_opvoice_path = ref_opvoice_path[0]

            if speaker_wav == "reference" and not speaker_path_text:
                allow_infer = False
                print("Referenc not found, Skip")
        else:
                ref_opvoice_path = find_openvoice_ref_by_name(
                    this_dir, opvoice_ref_list)

        if allow_infer:
            infer_openvoice(input_path=output_file,
                            ref_path=ref_opvoice_path, output_path=result)

            # Update the output_file with the absolute path to the result
            output_file = result.absolute()

    if improve_output_resemble:
        output_file = resemble_enhance_audio(
            **resemble_enhance_settings, audio_path=output_file, output_type=output_type)[1]

    if enable_waveform:
        return gr.make_waveform(audio=output_file), output_file, status_message
    else:
        return None, output_file, status_message


# GENERATION HANDLERS
generate_btn.click(
    fn=generate_audio,
    inputs=[
        # Resemble enhance Settings
        enhance_resemble_chunk_seconds,
        enhance_resemble_chunk_overlap,
        enhance_resemble_solver,
        enhance_resemble_num_funcs,
        enhance_resemble_temperature,
        enhance_resemble_denoise,
        # RVC settings
        rvc_settings_model_path,
        rvc_settings_index_path,
        rvc_settings_model_name,
        rvc_settings_pitch,
        rvc_settings_index_rate,
        rvc_settings_protect_voiceless,
        rvc_settings_method,
        rvc_settings_filter_radius,
        rvc_settings_resemple_rate,
        rvc_settings_envelope_mix,
        # OpenVoice Setting
        opvoice_ref_list,
        # Batch
        batch_generation,
        batch_generation_path,
        speaker_ref_wavs,
        # Batch Subtitles
        batch_sub_generation,
        batch_sub_generation_path,
        sync_sub_generation,
        # Features
        language_auto_detect,
        enable_waveform,
        improve_output_audio,
        improve_output_resemble,
        improve_output_voice2voice,
        #  Default settings
        output_type,
        text,
        languages,
        # Help variables
        speaker_value_text,
        speaker_path_text,
        additional_text_input,
        # TTS settings
        temperature,
        length_penalty,
        repetition_penalty,
        top_k,
        top_p,
        speed,
        sentence_split,
        # STATUS
        status_bar
    ],
    outputs=[video_gr, audio_gr, status_bar]
)


enable_waveform.change(
    fn=switch_waveform,
    inputs=[enable_waveform, video_gr],
    outputs=[video_gr]
)
