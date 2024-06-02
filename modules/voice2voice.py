import glob
import shutil
from datetime import datetime
from pathlib import Path

from modules.text2voice.voice2voice import select_rvc_model, update_rvc_model, update_openvoice_ref_list
from scripts.translate import translate_and_get_voice,translate_advance_stage1
from scripts.voice2voice import infer_rvc, get_openvoice_refs, infer_rvc_batch, infer_openvoice, find_openvoice_ref_by_name
from scripts.funcs import save_audio_to_wav

from xtts_webui import *

import ffmpeg

def convert_to_mp3(input_file, output_file):
  """Конвертирует входной аудиофайл в формат MP3 с помощью ffmpeg"""
  try:
    stream = ffmpeg.input(str(input_file))
    stream = ffmpeg.output(stream, str(output_file))
    ffmpeg.run(stream)
  except ffmpeg.Error as e:
    print(f'Ошибка при конвертации аудио: {e.stderr}')

# Constants
WAV_EXTENSION = "*.wav"
MP3_EXTENSION = "*.mp3"
FLAC_EXTENSION = "*.flac"

DATE_FORMAT = "%Y%m%d_%H%M%S"
SPEAKER_PREFIX = "speaker/"
REFERENCE_KEYWORD = "reference"

prepare_segments = None

# Auxiliary functions
def translate_and_voiceover_advance(
    translate_audio_single,
    translate_audio_batch,
    translate_audio_batch_path,
    # WHISPER
    translate_whisper_compute_time,
    translate_whisper_batch_size,
    translate_whisper_aline,
    translate_whisper_device,
    translate_whisper_model,
    translate_audio_mode,
    translate_source_lang,
    translate_target_lang,
    translate_speaker_lang,
    translate_num_sent,
    translate_max_reference_seconds,
    translate_translator,
    translate_speed,
    translate_temperature,
    translate_length_penalty,
    translate_repetition_penalty,
    translate_top_k,
    translate_top_p,
    translate_sentence_split,
    translate_status_bar,
    # Sub settings
    max_line_sub_v2v,
    max_width_sub_v2v,
    highlight_words_v2v
):
    print("Hello world")
    if not translate_audio_single and not translate_audio_batch and not translate_audio_batch_path:
        return None, None, "Please load audio"

    options = {
        "temperature": float(translate_temperature),
        "length_penalty": float(translate_length_penalty),
        "repetition_penalty": float(translate_repetition_penalty),
        "top_k": translate_top_k,
        "top_p": float(translate_top_p),
        "speed": float(translate_speed),
    }
    output_folder = this_dir / OUTPUT_FOLDER
    folder_name = f"translated_from_{translate_source_lang}_to_{translate_target_lang}" + \
        datetime.now().strftime(DATE_FORMAT)

    # Save Audio
    input_file = None
    if translate_audio_single:
        rate, y = translate_audio_single
        input_file = save_audio_to_wav(rate, y, Path.cwd())

    audio_files = translate_audio_batch or []
    if translate_audio_batch_path:
        audio_files.extend(find_audio_files(translate_audio_batch_path))

    current_date = datetime.now().strftime(DATE_FORMAT)
    tranlsated_filename = f"translated_from_{translate_source_lang}_to_{translate_target_lang}_{current_date}.wav"
    translate_audio_file = translate_advance_stage1(
        this_dir=this_dir,
        filename=input_file,
        xtts=XTTS,
        options=options,
        text_translator=translate_translator,
        translate_mode=True,
        whisper_model=translate_whisper_model,
        mode=translate_audio_mode,
        source_lang=translate_source_lang,
        target_lang=translate_target_lang,
        speaker_lang=translate_speaker_lang,
        num_sen=translate_num_sent,
        ref_seconds=translate_max_reference_seconds,
        output_filename=output_folder / tranlsated_filename,
        progress=gr.Progress(track_tqdm=True),
        # Sub settings
        max_line_sub_v2v = max_line_sub_v2v,
        max_width_sub_v2v = max_width_sub_v2v,
        highlight_words_v2v = highlight_words_v2v,
        whisper_align=translate_whisper_aline,
        whisper_device = translate_whisper_device,
        whisper_batch_size=translate_whisper_batch_size,
        whisper_compute_type=translate_whisper_compute_time,
    )
    openvoice_status_bar = gr.Progress(track_tqdm=True)
    
    global prepare_segments
        
    translate_audio_file_ready = translate_audio_file[0]
    translate_audio_file_ready = "\n".join(translate_audio_file_ready)
    number_stork = len(translate_audio_file[0])
    prepare_segments = translate_audio_file[1]
        
    print("Translate",translate_audio_file)
    return gr.TextArea(value=translate_audio_file_ready,lines=number_stork,interactive=True,visible=True), gr.Button("Stage - 2 dub",visible=True),i18n("Now edit the text and click on the 'Voice Text' button")

def translate_and_voiceover_advance_stage2(
    translate_audio_single,
    translate_audio_batch,
    translate_audio_batch_path,
    # WHISPER
    translate_whisper_compute_time,
    translate_whisper_batch_size,
    translate_whisper_aline,
    translate_whisper_device,
    translate_whisper_model,
    translate_audio_mode,
    translate_source_lang,
    translate_target_lang,
    translate_speaker_lang,
    translate_num_sent,
    translate_max_reference_seconds,
    translate_translator,
    translate_speed,
    translate_temperature,
    translate_length_penalty,
    translate_repetition_penalty,
    translate_top_k,
    translate_top_p,
    translate_sentence_split,
    translate_status_bar,
    translate_advance_stage1_text,
    translate_ref_speaker_list,
    sync_original,
    # Sub settings
    max_line_sub_v2v,
    max_width_sub_v2v,
    highlight_words_v2v
    
):
    print("Hello world")
    if not translate_audio_single and not translate_audio_batch and not translate_audio_batch_path:
        return None, None, "Please load audio"

    options = {
        "temperature": float(translate_temperature),
        "length_penalty": float(translate_length_penalty),
        "repetition_penalty": float(translate_repetition_penalty),
        "top_k": translate_top_k,
        "top_p": float(translate_top_p),
        "speed": float(translate_speed),
    }
    output_folder = this_dir / OUTPUT_FOLDER
    folder_name = f"translated_from_{translate_source_lang}_to_{translate_target_lang}" + \
        datetime.now().strftime(DATE_FORMAT)

    # Save Audio
    input_file = None
    if translate_audio_single:
        rate, y = translate_audio_single
        input_file = save_audio_to_wav(rate, y, Path.cwd())

    audio_files = translate_audio_batch or []
    if translate_audio_batch_path:
        audio_files.extend(find_audio_files(translate_audio_batch_path))
    
    speaker_wavs = None
    if translate_ref_speaker_list:
        speaker_wavs = XTTS.get_speaker_path(translate_ref_speaker_list)
    
    global prepare_segments

    current_date = datetime.now().strftime(DATE_FORMAT)
    tranlsated_filename = f"translated_from_{translate_source_lang}_to_{translate_target_lang}_{current_date}.wav"
    translate_audio_file = translate_and_get_voice(
        this_dir=this_dir,
        filename=input_file,
        xtts=XTTS,
        options=options,
        sync_original=sync_original,
        text_translator=translate_translator,
        translate_mode=True,
        whisper_model=translate_whisper_model,
        mode=translate_audio_mode,
        source_lang=translate_source_lang,
        target_lang=translate_target_lang,
        speaker_lang=translate_speaker_lang,
        num_sen=translate_num_sent,
        ref_seconds=translate_max_reference_seconds,
        output_filename=output_folder / tranlsated_filename,
        prepare_text=translate_advance_stage1_text,
        prepare_segments=prepare_segments,
        speaker_wavs=speaker_wavs,
        progress=gr.Progress(track_tqdm=True),
        # Sub settings
        max_line_sub_v2v = max_line_sub_v2v,
        max_width_sub_v2v = max_width_sub_v2v,
        highlight_words_v2v = highlight_words_v2v,
        whisper_align=translate_whisper_aline,
        whisper_batch_size=translate_whisper_batch_size,
        whisper_compute_type=translate_whisper_compute_time,
        whisper_device = translate_whisper_device,
    )
    openvoice_status_bar = gr.Progress(track_tqdm=True)
    
    
    print("Translate",translate_and_voiceover)
    return None, translate_audio_file[0],translate_audio_file[1], "Done"

def translate_and_voiceover(
    translate_audio_single,
    translate_audio_batch,
    translate_audio_batch_path,
    translate_whisper_compute_time,
    translate_whisper_batch_size,
    translate_whisper_aline,
    translate_whisper_device,
    translate_whisper_model,
    translate_audio_mode,
    translate_source_lang,
    translate_target_lang,
    translate_speaker_lang,
    translate_num_sent,
    translate_max_reference_seconds,
    translate_translator,
    translate_speed,
    translate_temperature,
    translate_length_penalty,
    translate_repetition_penalty,
    translate_top_k,
    translate_top_p,
    translate_sentence_split,
    translate_status_bar,
    translate_ref_speaker_list,
    sync_original,
    # Sub settings
    max_line_sub_v2v ,
    max_width_sub_v2v,
    highlight_words_v2v
):
    print("Hello world")
    if not translate_audio_single and not translate_audio_batch and not translate_audio_batch_path:
        return None, None, "Please load audio"

    options = {
        "temperature": float(translate_temperature),
        "length_penalty": float(translate_length_penalty),
        "repetition_penalty": float(translate_repetition_penalty),
        "top_k": translate_top_k,
        "top_p": float(translate_top_p),
        "speed": float(translate_speed),
    }
    output_folder = this_dir / OUTPUT_FOLDER
    folder_name = f"translated_from_{translate_source_lang}_to_{translate_target_lang}" + \
        datetime.now().strftime(DATE_FORMAT)

    # Save Audio
    input_file = None
    if translate_audio_single:
        rate, y = translate_audio_single
        input_file = save_audio_to_wav(rate, y, Path.cwd())

    audio_files = translate_audio_batch or []
    if translate_audio_batch_path:
        audio_files.extend(find_audio_files(translate_audio_batch_path))

    speaker_wavs = None
    if translate_ref_speaker_list:
        speaker_wavs = XTTS.get_speaker_path(translate_ref_speaker_list)
        
    current_date = datetime.now().strftime(DATE_FORMAT)
    tranlsated_filename = f"translated_from_{translate_source_lang}_to_{translate_target_lang}_{current_date}.wav"
    translate_audio_file = translate_and_get_voice(
        this_dir=this_dir,
        filename=input_file,
        xtts=XTTS,
        sync_original=sync_original,
        options=options,
        text_translator=translate_translator,
        translate_mode=True,
        whisper_model=translate_whisper_model,
        mode=translate_audio_mode,
        source_lang=translate_source_lang,
        target_lang=translate_target_lang,
        speaker_lang=translate_speaker_lang,
        num_sen=translate_num_sent,
        ref_seconds=translate_max_reference_seconds,
        output_filename=output_folder / tranlsated_filename,
        speaker_wavs=speaker_wavs,
        progress=gr.Progress(track_tqdm=True),
        # Sub settings
        max_line_sub_v2v = max_line_sub_v2v,
        max_width_sub_v2v = max_width_sub_v2v,
        highlight_words_v2v = highlight_words_v2v,
        whisper_device = translate_whisper_device,
        whisper_align=translate_whisper_aline,
        whisper_batch_size=translate_whisper_batch_size,
        whisper_compute_type=translate_whisper_compute_time,
        
    )
    openvoice_status_bar = gr.Progress(track_tqdm=True)
    
    print("Translate",translate_and_voiceover)
    return None, translate_audio_file[0],translate_audio_file[1], "Done"


def get_reference_path(speaker_wav, speaker_path_text):
    if speaker_wav == REFERENCE_KEYWORD:
        return speaker_path_text if speaker_path_text else None
    else:
        ref_path = XTTS.get_speaker_path(speaker_wav)
        return ref_path[0] if isinstance(ref_path, list) else ref_path


def find_audio_files(batch_path):
    return glob.glob(os.path.join(batch_path, WAV_EXTENSION)) + \
        glob.glob(os.path.join(batch_path, MP3_EXTENSION)) + \
        glob.glob(os.path.join(batch_path, FLAC_EXTENSION))

# Main optimization function


def infer_openvoice_audio(openvoice_audio_single, openvoice_audio_batch, openvoice_audio_batch_path,
                          openvoice_voice_ref_list, openvoice_status_bar, speaker_path_text):
    print("hello world")

    if openvoice_voice_ref_list == "None":
        return None, None, "Please select Reference audio"

    if not openvoice_audio_single and not openvoice_audio_batch and not openvoice_audio_batch_path:
        return None, None, "Please load audio"

    output_folder = this_dir / OUTPUT_FOLDER
    folder_name = "openvoice_" + datetime.now().strftime(DATE_FORMAT)

    # Save Audio
    input_file = None
    if openvoice_audio_single:
        rate, y = openvoice_audio_single
        input_file = save_audio_to_wav(rate, y, Path.cwd())

    audio_files = openvoice_audio_batch or []
    if openvoice_audio_batch_path:
        audio_files.extend(find_audio_files(openvoice_audio_batch_path))

    openvoice_status_bar = gr.Progress(track_tqdm=True)
    if audio_files:
        output_folder = output_folder / folder_name
        os.makedirs(output_folder, exist_ok=True)
        tqdm_object = openvoice_status_bar.tqdm(
            audio_files, desc="Tuning Files...")

        for audio_file in tqdm_object:
            ref_voice_opvoice_path = None
            allow_infer = True

            if openvoice_voice_ref_list.startswith(SPEAKER_PREFIX):
                speaker_wav = openvoice_voice_ref_list.split("/")[-1]
                ref_voice_opvoice_path = get_reference_path(
                    speaker_wav, speaker_path_text)

                if not ref_voice_opvoice_path:
                    allow_infer = False
                    print("Reference not found")
            else:
                ref_voice_opvoice_path = find_openvoice_ref_by_name(
                    Path.cwd(), openvoice_voice_ref_list)

            if allow_infer and ref_voice_opvoice_path:

                output_filename = output_folder / \
                    f"openvoice_{Path(audio_file).stem}.wav"
                infer_openvoice(
                    input_path=audio_file, ref_path=ref_voice_opvoice_path, output_path=output_filename)

        return None, None, f"Files saved in {output_folder} folder"

    elif openvoice_audio_single:
        temp_dir = Path.cwd() / "output"
        filename_openvoice = Path(ref_voice_opvoice_path).stem
        output_filename = temp_dir / \
            f"openvoice_{filename_openvoice}_{datetime.now().strftime(DATE_FORMAT)}.wav"

        if openvoice_voice_ref_list.startswith(SPEAKER_PREFIX):
            speaker_wav = openvoice_voice_ref_list.split("/")[-1]
            ref_voice_opvoice_path = get_reference_path(
                speaker_wav, speaker_path_text)
        else:
            ref_voice_opvoice_path = find_openvoice_ref_by_name(
                Path.cwd(), openvoice_voice_ref_list)

        if ref_voice_opvoice_path:
            infer_openvoice(
                input_path=input_file, ref_path=ref_voice_opvoice_path, output_path=output_filename)
            output_audio = output_filename
            done_message = "Done"
        else:
            output_audio = None
            done_message = "Reference not found"

        return None, gr.Audio(label="Result", value=output_audio), done_message

    # If none of the conditions are met, return an error message
    return None, None, "An unexpected error occurred during processing"


# Main optimization function
def infer_rvc_audio(
        rvc_audio_single,
        rvc_audio_batch,
        rvc_audio_batch_path,
        rvc_voice_settings_output_type,
        rvc_voice_settings_model_name,
        rvc_voice_settings_model_path,
        rvc_voice_settings_index_path,
        rvc_voice_settings_pitch,
        rvc_voice_settings_index_rate,
        rvc_voice_settings_protect_voiceless,
        rvc_voice_settings_method,
        rvc_voice_filter_radius,
        rvc_voice_resemple_rate,
        rvc_voice_envelope_mix,
        rvc_voice_status_bar
):
    if not rvc_voice_settings_model_name:
        return None, None, "Please select RVC model"

    if not (rvc_audio_single or rvc_audio_batch or rvc_audio_batch_path):
        return None, None, "Please load audio"

    output_folder = this_dir / OUTPUT_FOLDER
    folder_name = f"rvc_{datetime.now().strftime(DATE_FORMAT)}"
    done_message = ""

    input_file = None
    if rvc_audio_single:
        rate, y = rvc_audio_single
        input_file = save_audio_to_wav(rate, y, Path.cwd())

    audio_files = rvc_audio_batch or []
    if rvc_audio_batch_path:
        audio_files.extend(find_audio_files(rvc_audio_batch_path))

    rvc_voice_status_bar = gr.Progress(track_tqdm=True)
    # Process batches of files
    # Обработка пакетов файлов
    if audio_files:
        output_folder = output_folder / folder_name
        os.makedirs(output_folder, exist_ok=True)
        temp_folder = output_folder / "temp"
        os.makedirs(temp_folder, exist_ok=True)

        infer_rvc_batch(
            model_name=rvc_voice_settings_model_name,
            pitch=rvc_voice_settings_pitch,
            index_rate=rvc_voice_settings_index_rate,
            protect_voiceless=rvc_voice_settings_protect_voiceless,
            method=rvc_voice_settings_method,
            index_path=rvc_voice_settings_model_path,
            model_path=rvc_voice_settings_index_path,
            paths=audio_files,
            opt_path=temp_folder,
            filter_radius=rvc_voice_filter_radius,
            resemple_rate=rvc_voice_resemple_rate,
            envelope_mix=rvc_voice_envelope_mix,
        )

        if rvc_voice_settings_output_type == "mp3":
            for file in os.listdir(temp_folder):
                if file.endswith(".wav"):
                    input_file = temp_folder / file
                    output_file = output_folder / f"{file[:-4]}.mp3"
                    convert_to_mp3(input_file, output_file)

            # shutil.rmtree(temp_folder)  # Удаляем временную папку с wav файлами
        else:
            for file in os.listdir(temp_folder):
                shutil.move(str(temp_folder / file), str(output_folder / file))

        # shutil.rmtree(temp_folder)

        done_message = f"Готово, файлы сохранены в папке {folder_name}"
        return None, None, done_message


    # Process single file
    elif rvc_audio_single:
        output_file_name = output_folder / \
            f"rvc_{rvc_voice_settings_model_name}_{datetime.now().strftime(DATE_FORMAT)}.wav"
        infer_rvc(
            pitch=rvc_voice_settings_pitch,
            index_rate=rvc_voice_settings_index_rate,
            protect_voiceless=rvc_voice_settings_protect_voiceless,
            method=rvc_voice_settings_method,
            index_path=rvc_voice_settings_model_path,
            model_path=rvc_voice_settings_index_path,
            input_path=input_file,
            opt_path=output_file_name,
            filter_radius=rvc_voice_filter_radius,
            resemple_rate=rvc_voice_resemple_rate,
            envelope_mix=rvc_voice_envelope_mix
        )

        done_message = "Done"
        output_audio = output_file_name
        
        if rvc_voice_settings_output_type == "wav":
            output_audio = output_file_name
        
        if rvc_voice_settings_output_type == "mp3":
            output_file = output_folder / f"rvc_{rvc_voice_settings_model_name}_{datetime.now().strftime(DATE_FORMAT)}.mp3"
            convert_to_mp3(output_file_name, output_file)
            output_audio = output_file
        
        return None, gr.Audio(label="Result", value=output_audio), done_message

    # If none of the conditions are met, return an error message
    return None, None, "An unexpected error occurred during processing"

from scripts.funcs import write_key_value_to_env, read_key_from_env 

def save_auth_key(deepl_auth_key_textbox):
    # Write to env
    global deepl_api_key
    deepl_api_key = deepl_auth_key_textbox
    print("API key =",deepl_auth_key_textbox)
    write_key_value_to_env("DEEPL_API_KEY",deepl_auth_key_textbox)
    return

def show_deepl_api_key(translate_translator):
    if translate_translator == "deepl":
        deepl_api_from_env = read_key_from_env("DEEPL_API_KEY")
        return gr.Textbox(label="Deepl Api Key", value=deepl_api_from_env,type="password",visible=True)
    return gr.Textbox(visible=False)

def switch_visible_mode1_params(translate_audio_mode):
    if translate_audio_mode == 1:
        return gr.Markdown(value="**Attention, the number of rows must not be changed, otherwise an error will occur**",visible=True)
    return gr.Markdown(visible=False)

def switch_visible_mode2_params(translate_audio_mode):
    if translate_audio_mode == 2:
        return gr.Slider(label=i18n("Number of sentences that will be voiced at one time"),minimum=1,maximum=6,step=1,visible=True),gr.Slider(label=i18n("Number of reference seconds that will be used"),minimum=10,maximum=600,value=20,step=1,visible=True)
    if translate_audio_mode == 3:
        return gr.Slider(label=i18n("Number of sentences that will be voiced at one time"),minimum=1,maximum=6,step=1,visible=True),gr.Slider(visible=False)
    return gr.Slider(visible=False), gr.Slider(visible=False)

def switch_visible_mode3_params(translate_audio_mode):
    if translate_audio_mode == 3:
        speakers_list = XTTS.get_speakers()
        speaker_value = ""
        if not speakers_list:
            speakers_list = ["None"]
            speaker_value = "None"
        else:
            speaker_value = speakers_list[0]
            XTTS.speaker_wav = speaker_value

        translate_ref_speaker_list = gr.Dropdown(
            label=i18n("Reference Speaker from folder 'speakers'"),visible=True, value=speaker_value, choices=speakers_list)
        translate_show_ref_speaker_from_list = gr.Checkbox(
            value=False, label=i18n("Show reference sample"),visible=True, info=i18n("This option will allow you to listen to your reference sample"))
        translate_update_ref_speaker_list_btn = gr.Button(
                value=i18n("Update"), elem_classes="speaker-update__btn",visible=True)
        translate_ref_speaker_example = gr.Audio(
            label=i18n("speaker sample"), sources="upload", visible=False, interactive=False)
        
        return translate_ref_speaker_list,translate_show_ref_speaker_from_list,translate_update_ref_speaker_list_btn,translate_ref_speaker_example
    return gr.Dropdown(visible=False,value=None),gr.Checkbox(visible=False),gr.Button(visible=False),gr.Audio(visible=False)

def check_whisper_cpu(translate_whisper_device):
    if translate_whisper_device == "cpu":
        return gr.Radio(choices=["float16"],value="float16",label="compute type")
    return gr.Radio(label="Compute Type",choices=["int8","float16"],value="float16",info="change to 'int8' if low on GPU mem (may reduce accuracy)")

translate_whisper_device.change(fn=check_whisper_cpu,inputs=[translate_whisper_device],outputs=[translate_whisper_compute_time])


translate_audio_mode.change(fn=switch_visible_mode1_params,inputs=[translate_audio_mode],outputs=[transalte_advance_markdown])
translate_audio_mode.change(fn=switch_visible_mode2_params,inputs=[translate_audio_mode],outputs=[translate_num_sent,translate_max_reference_seconds])
translate_audio_mode.change(fn=switch_visible_mode3_params,inputs=[translate_audio_mode],outputs=[translate_ref_speaker_list,translate_show_ref_speaker_from_list,translate_update_ref_speaker_list_btn,translate_ref_speaker_example])

translate_translator.change(fn=show_deepl_api_key,inputs=[translate_translator],outputs=[deepl_auth_key_textbox])
deepl_auth_key_textbox.change(fn=save_auth_key,inputs=[deepl_auth_key_textbox])

# def switch_translate_modes(translate_audio_mode):
#     if translate_audio_mode == 2 or translate_audio_mode == 3:
#         return gr.Button(value=i18n("Stage 1 - Translate and edit Text"), visible=True), gr.TextArea(label="Translated Text", value=None, visible=False), gr.Button(value=i18n("Stage 2 - Dubbing"), visible=False), gr.Markdown(i18n("Work in progress..."), visible=False)
#     else:
#         return gr.Button(visible=False),gr.TextArea(visible=False), gr.Button(visible=False),gr.Markdown(i18n("Work in progress..."), visible=True)
# translate_advance_stage1_btn = gr.Button(value=i18n("Stage 1 - Translate and edit Text"))
                # translate_advance_stage1_text = gr.TextArea(label="Translated Text",value=None,visible=False)
                # translate_advance_stage2_btn = gr.Button(value=i18n("Stage 2 - Dubbing"),visible=False)
                # transalte_advance_markdown = gr.Markdown(i18n("Work in progress..."),visible=None)

# translate_audio_mode.change(fn=switch_translate_modes,inputs=[translate_audio_mode],
#                             outputs=[translate_advance_stage1_btn,translate_advance_stage1_text,translate_advance_stage2_btn,transalte_advance_markdown])

# max_width_sub_v2v = gr.Slider(label=i18n("Max width per line"), minimum=1, maximum=100, value=40, step=1)
#                         max_line_sub_v2v = gr.Slider(label=i18n("Max line"), minimum=1, maximum=20, value=2, step=1)

translate_btn.click(fn=translate_and_voiceover, inputs=[
                                                        # INPUTS
                                                        translate_audio_single,
                                                        translate_audio_batch,
                                                        translate_audio_batch_path,
                                                        # WHISPER
                                                        translate_whisper_compute_time,
                                                        translate_whisper_batch_size,
                                                        translate_whisper_aline,
                                                        translate_whisper_device,
                                                        # TRANSLATE SETTIGNS
                                                        translate_whisper_model,
                                                        translate_audio_mode,
                                                        translate_source_lang,
                                                        translate_target_lang,
                                                        translate_speaker_lang,
                                                        translate_num_sent,
                                                        translate_max_reference_seconds,
                                                        # XTTS SETTINGS
                                                        translate_translator,
                                                        translate_speed,
                                                        translate_temperature,
                                                        translate_length_penalty,
                                                        translate_repetition_penalty,
                                                        translate_top_k,
                                                        translate_top_p,
                                                        translate_sentence_split,
                                                        # STATUS BAR
                                                        translate_status_bar,
                                                        translate_ref_speaker_list,
                                                        sync_with_original_checkbox,
                                                        # Sub settings
                                                        max_line_sub_v2v,
                                                        max_width_sub_v2v,
                                                        highlight_words_v2v
                                                        ], outputs=[translate_video_output, translate_voice_output,translate_files_output, translate_status_bar])

if RVC_ENABLE:
    rvc_voice_settings_model_name.change(fn=select_rvc_model, inputs=[rvc_voice_settings_model_name], outputs=[
        rvc_voice_settings_model_path, rvc_voice_settings_index_path])
    rvc_voice_settings_update_btn.click(fn=update_rvc_model, inputs=[rvc_voice_settings_model_name], outputs=[
        rvc_voice_settings_model_name, rvc_voice_settings_model_path, rvc_voice_settings_index_path])
    
    rvc_voice_infer_btn.click(fn=infer_rvc_audio, inputs=[
        # INPUT
        rvc_audio_single,
        rvc_audio_batch,
        rvc_audio_batch_path,
        rvc_voice_settings_output_type,
        # PATH
        rvc_voice_settings_model_name,
        rvc_voice_settings_model_path,
        rvc_voice_settings_index_path,
        # SETTINGS
        rvc_voice_settings_pitch,
        rvc_voice_settings_index_rate,
        rvc_voice_settings_protect_voiceless,
        rvc_voice_settings_method,
        rvc_voice_filter_radius,
        rvc_voice_resemple_rate,
        rvc_voice_envelope_mix,
        # STATUS
        rvc_voice_status_bar
    ], outputs=[
        rvc_video_output,
        rvc_voice_output,
        rvc_voice_status_bar
    ])
    
    opvoice_voice_show_speakers.change(fn=update_openvoice_ref_list, inputs=[
        opvoice_voice_ref_list, opvoice_voice_show_speakers], outputs=[opvoice_voice_ref_list])
    
    openvoice_voice_infer_btn.click(fn=infer_openvoice_audio, inputs=[openvoice_audio_single, openvoice_audio_batch, openvoice_audio_batch_path,
                                                                      opvoice_voice_ref_list, openvoice_status_bar, speaker_path_text], outputs=[openvoice_video_output, openvoice_voice_output, openvoice_status_bar])


translate_advance_stage1_btn.click(fn=translate_and_voiceover_advance, inputs=[
                                                        # INPUTS
                                                        translate_audio_single,
                                                        translate_audio_batch,
                                                        translate_audio_batch_path,
                                                        # WHISPER
                                                        translate_whisper_compute_time,
                                                        translate_whisper_batch_size,
                                                        translate_whisper_aline,
                                                        translate_whisper_device,
                                                        # TRANSLATE SETTIGNS
                                                        translate_whisper_model,
                                                        translate_audio_mode,
                                                        translate_source_lang,
                                                        translate_target_lang,
                                                        translate_speaker_lang,
                                                        translate_num_sent,
                                                        translate_max_reference_seconds,
                                                        # XTTS SETTINGS
                                                        translate_translator,
                                                        translate_speed,
                                                        translate_temperature,
                                                        translate_length_penalty,
                                                        translate_repetition_penalty,
                                                        translate_top_k,
                                                        translate_top_p,
                                                        translate_sentence_split,
                                                        # STATUS BAR
                                                        translate_status_bar,
                                                        # Sub settings
                                                        max_line_sub_v2v,
                                                        max_width_sub_v2v,
                                                        highlight_words_v2v
                                                        ], outputs=[translate_advance_stage1_text,translate_advance_stage2_btn,translate_status_bar])

translate_advance_stage2_btn.click(fn=translate_and_voiceover_advance_stage2, inputs=[
                                                        # INPUTS
                                                        translate_audio_single,
                                                        translate_audio_batch,
                                                        translate_audio_batch_path,
                                                        # WHISPER
                                                        translate_whisper_compute_time,
                                                        translate_whisper_batch_size,
                                                        translate_whisper_aline,
                                                        translate_whisper_device,
                                                        # TRANSLATE SETTIGNS
                                                        translate_whisper_model,
                                                        translate_audio_mode,
                                                        translate_source_lang,
                                                        translate_target_lang,
                                                        translate_speaker_lang,
                                                        translate_num_sent,
                                                        translate_max_reference_seconds,
                                                        # XTTS SETTINGS
                                                        translate_translator,
                                                        translate_speed,
                                                        translate_temperature,
                                                        translate_length_penalty,
                                                        translate_repetition_penalty,
                                                        translate_top_k,
                                                        translate_top_p,
                                                        translate_sentence_split,
                                                        # STATUS BAR
                                                        translate_status_bar,
                                                        translate_advance_stage1_text,
                                                        translate_ref_speaker_list,
                                                        sync_with_original_checkbox,
                                                        # Sub settings
                                                        max_line_sub_v2v,
                                                        max_width_sub_v2v,
                                                        highlight_words_v2v
                                                        ], outputs=[translate_video_output, translate_voice_output,translate_files_output, translate_status_bar])