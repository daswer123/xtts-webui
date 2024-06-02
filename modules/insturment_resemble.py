from tqdm import tqdm
from modules.voice2voice import find_audio_files
from scripts.languages import get_language_from_name
from xtts_webui import *
import shutil

from datetime import datetime
from scripts.funcs import resemble_enhance_audio, save_audio_to_wav
import glob

from scripts.funcs import read_key_from_env, write_key_value_to_env

# Constants
WAV_EXTENSION = "*.wav"
MP3_EXTENSION = "*.mp3"
FLAC_EXTENSION = "*.flac"

DATE_FORMAT = "%Y%m%d_%H%M%S"
SPEAKER_PREFIX = "speaker/"
REFERENCE_KEYWORD = "reference"

def instrument_enchane_audio(
    resemble_audio_single,
    resemble_audio_batch,
    resemble_audio_batch_path,
    resemble_choose_action,
    resemble_chunk_seconds,
    resemble_chunk_overlap,
    resemble_solver,
    resemble_num_funcs,
    resemble_temperature,
    resemble_denoise,
    resemble_output_type,
    resemble_status_label
):

    resemble_status_label = gr.Progress(track_tqdm=True)

    output_folder = this_dir / OUTPUT_FOLDER
    folder_name = ""

    done_message = ""

    # Define folder name
    if resemble_choose_action == "both":
        # Current date
        folder_name = "resemble_enhance_both"
    else:
        folder_name = f"resemble_enhance_{resemble_choose_action}"

    folder_name += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output dir

    # Save Audio
    if resemble_audio_single is not None:
        rate, y = resemble_audio_single
        input_file = save_audio_to_wav(rate, y, this_dir)

    use_enhance = resemble_choose_action == "both" or resemble_choose_action == "only_enchance"
    use_denoise = resemble_choose_action == "both" or resemble_choose_action == "only_denoise"

    audio_files = resemble_audio_batch

    if resemble_audio_batch_path:
        audio_files = glob.glob(resemble_audio_batch_path + "/*.wav")
        audio_files += glob.glob(resemble_audio_batch_path + "/*.mp3")
        audio_files += glob.glob(resemble_audio_batch_path + "/*.flac")

    if resemble_audio_batch or resemble_audio_batch_path:
        output_dir = output_folder / folder_name
        os.makedirs(output_dir, exist_ok=True)

        if resemble_status_label is not None:
            tqdm_object = resemble_status_label.tqdm(
                audio_files, desc="Enchance files...")
        else:
            tqdm_object = tqdm(audio_files)

        for file in tqdm_object:
            output_audio = resemble_enhance_audio(
                audio_path=file,
                use_enhance=use_enhance,
                use_denoise=use_denoise,
                solver=resemble_solver,
                nfe=resemble_num_funcs,
                tau=resemble_temperature,
                chunk_seconds=resemble_chunk_seconds,
                chunks_overlap=resemble_chunk_overlap,
                denoising=resemble_denoise,
                output_type="wav",
                output_folder=folder_name
            )
        done_message = f"Done, file saved in {folder_name} folder"
    else:
        output_audio = resemble_enhance_audio(
            audio_path=input_file,
            use_enhance=use_enhance,
            use_denoise=use_denoise,
            solver=resemble_solver,
            nfe=resemble_num_funcs,
            tau=resemble_temperature,
            chunk_seconds=resemble_chunk_seconds,
            chunks_overlap=resemble_chunk_overlap,
            denoising=resemble_denoise,
            output_type="wav",
            output_folder=""
        )
        done_message = f"Done"

        # With glob collect audio files and create list with path to each file ( wav, mp3, flac)

    return gr.Audio(label="Denoise", visible=use_denoise, value=output_audio[0]), gr.Audio(label="Enhance", visible=use_enhance, value=output_audio[1]), done_message
    # return result_denoise,result_enhance,resemble_status_label
    
    
def switch_api_field(subbtitle_correct_type):
    gpt_4_key = read_key_from_env("GPT_4_KEY")
    claude_3_key = read_key_from_env("CLAUDE_3_KEY")
    
    if subbtitle_correct_type == "Claude 3 Opus" or subbtitle_correct_type == "Claude 3 Sonet":
        return gr.Textbox(label=i18n("API Key Claude"),type="password", value=claude_3_key, visible=True), gr.Textbox(label=i18n("API Key GPT"),type="password",interactive=True, value=gpt_4_key,visible=False)
    
    if subbtitle_correct_type == "GPT-4" or subbtitle_correct_type == "GPT-4-Turbo":
        return gr.Textbox(label=i18n("API Key Claude"),type="password", value=claude_3_key, visible=False), gr.Textbox(label=i18n("API Key GPT"), type="password",interactive=True, value=gpt_4_key,visible=True)
    

def change_env_gpt_key(field_value):
    write_key_value_to_env("GPT_4_KEY", field_value)
    return field_value

def change_env_claude_key(field_value):
    write_key_value_to_env("CLAUDE_3_KEY", field_value)
    return field_value

from scripts.llm import process_subtitle_chunk, send_req_llm
from pathlib import Path

def get_key_and_model(type, api_claude, api_gpt):
    if type == "Claude 3 Opus" or type == "Claude 3 Sonet":
        api_key = api_claude
        llm_type = "claude"
        
        if type == "Claude 3 Opus":
            model = "claude-3-opus-20240229"
            
        if type == "Claude 3 Sonet":
            model = "claude-3-sonnet-20240229"
        
    if type == "GPT-4" or type == "GPT-4-Turbo":
        api_key = api_gpt
        llm_type = "gpt"
        
        if type == "GPT-4":
            model = "gpt-4"
            
        if type == "GPT-4-Turbo":
            model = "gpt-4-turbo-preview"
            
    return [api_key, model,type]

async def correct_subs(subtitle_file_correct,subtitle_file_batch_correct,subtitle_file_batch_correct_path, subtitle_correct_prompt, subbtitle_correct_type, subtitle_api_key_claude, subtitle_api_key_gpt, chunk_size=1024):
    [api_key, model, llm_type] = get_key_and_model(subbtitle_correct_type, subtitle_api_key_claude, subtitle_api_key_gpt)
    
    corrected_files = []
    
    if subtitle_file_batch_correct_path:
        # Обработка файлов из указанной папки
        subtitle_files = [str(f) for f in Path(subtitle_file_batch_correct_path).glob("*") if f.suffix.lower() in [".srt", ".txt", ".ass"]]
    elif subtitle_file_batch_correct:
        # Обработка батча файлов
        subtitle_files = subtitle_file_batch_correct
    elif subtitle_file_correct:
        # Обработка единичного файла
        subtitle_files = [subtitle_file_correct]
    else:
        print("No subtitle files provided.")
        return []
    
    progress = gr.Progress(track_tqdm=True)
    
    for subtitle_file in progress.tqdm(subtitle_files):
        # subtitle_file = subtitle_files[i]
        with open(subtitle_file, "r", encoding="utf-8") as f:
            subtitle_text = f.read()
        
        # Separate subs for every chunk_size chars 
        output_chunks = []
        subtitle_text_split = [subtitle_text[i:i+chunk_size] for i in range(0, len(subtitle_text), chunk_size)]
        
        for chunk in subtitle_text_split:
            res_chunk = await process_subtitle_chunk(api_key, chunk, subtitle_correct_prompt, model, llm_type)
            if res_chunk is not None:
                output_chunks.append(res_chunk)
            else:
                return None
        
        # Save to output
        result = "\n".join(output_chunks)
        
        if subtitle_file_batch_correct_path or subtitle_file_batch_correct:
            # Сохранение батча файлов
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = Path(OUTPUT_FOLDER) / f"corrected_subs_batch_{timestamp}"
        else:
            # Сохранение единичного файла
            output_folder = Path(OUTPUT_FOLDER) / f"{Path(subtitle_file).stem}_corrected"
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Same format as original
        subtitle_file_output = output_folder / f"{Path(subtitle_file).stem}_corrected.{Path(subtitle_file).suffix.replace('.','')}"
        
        with open(subtitle_file_output, "w", encoding="utf-8") as f:
            f.write(result)
            print(f"Saved to {subtitle_file_output}")
        
        corrected_files.append(str(subtitle_file_output))
    
    return corrected_files



from scripts.translate import parse_srt, save_old_subs_and_txt,save_subs_and_txt, transcribe_for_gradio


def cut_subtitles(subtitle_file, subtitle_max_width):
        # Read name
        current_timestamp_formated = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Output folder 
        output_name = f"subtitles_{subtitle_max_width}_{current_timestamp_formated}"
        output_folder = this_dir / OUTPUT_FOLDER / f"subtitles_{subtitle_max_width}_{current_timestamp_formated}"
        os.makedirs(output_folder, exist_ok=True)
        
        subtitles = parse_srt(subtitle_file)        
        subtitle_output = save_old_subs_and_txt(subtitles, output_folder,output_name,subtitle_max_width,save_only_src=True)
        
        return subtitle_output

resemble_generate_btn.click(fn=instrument_enchane_audio,
                            inputs=[
                                resemble_audio_single,
                                resemble_audio_batch,
                                resemble_audio_batch_path,
                                resemble_choose_action,
                                resemble_chunk_seconds,
                                resemble_chunk_overlap,
                                resemble_solver,
                                resemble_num_funcs,
                                resemble_temperature,
                                resemble_denoise,
                                resemble_output_type,
                                resemble_status_label
                            ],
                            outputs=[result_denoise, result_enhance, resemble_status_label])


subtitle_generate_btn.click(fn=cut_subtitles,inputs=[subtitle_file,subtitle_max_width],outputs=[subtitle_output])
subbtitle_correct_type.change(fn=switch_api_field, inputs=[subbtitle_correct_type], outputs=[subtitle_api_key_claude,subtitle_api_key_gpt])


subtitle_api_key_claude.change(fn=change_env_claude_key, inputs=[subtitle_api_key_claude])
subtitle_api_key_gpt.change(fn=change_env_gpt_key, inputs=[subtitle_api_key_gpt])
subtitle_correct_btn.click(fn=correct_subs,inputs=[
    subtitle_file_correct,subtitle_file_batch_correct,subtitle_file_batch_correct_path,subtitle_correct_prompt,subbtitle_correct_type,subtitle_api_key_claude,subtitle_api_key_gpt],
                           outputs=[subtitle_output_correct])


def get_subs(subtitle_file_get, subtitle_file_batch_get, subtitle_file_batch_get_path, get_sub_whisper_model, 
             get_sub_source_lang, get_sub_max_width_sub_v2v, get_sub_max_line_sub_v2v, get_sub_highlight_words_v2v,
             get_sub_translator, deepl_auth_key_textbox, get_sub_whisper_compute_time, get_sub_whisper_device,
             get_sub_whisper_batch_size, get_sub_whisper_aline, get_sub_status_label):

    output_folder = this_dir / OUTPUT_FOLDER
    folder_name = f"subs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = output_folder / folder_name
    
    if get_sub_source_lang != "auto":
        get_sub_source_lang = get_language_from_name(get_sub_source_lang).code
        # print(get_sub_source_lang)

    # Save Audio
    input_file = None
    if subtitle_file_get:
        rate, y = subtitle_file_get
        input_file = save_audio_to_wav(rate, y, Path.cwd())

    audio_files = subtitle_file_batch_get or []
    if subtitle_file_batch_get_path:
        audio_files.extend(find_audio_files(subtitle_file_batch_get_path))

    if audio_files:
        os.makedirs(output_dir, exist_ok=True)
        
        get_sub_status_label = gr.Progress(track_tqdm=True)

        if get_sub_status_label is not None:
            tqdm_object = get_sub_status_label.tqdm(audio_files, desc="Transcribing files...")
        else:
            tqdm_object = tqdm(audio_files)
            
        subs = []

        for file in tqdm_object:
            transcribe_for_gradio(
                subtitle_file_get=file,
                get_sub_whisper_model=get_sub_whisper_model,
                get_sub_source_lang=get_sub_source_lang,
                get_sub_max_width_sub_v2v=get_sub_max_width_sub_v2v,
                get_sub_max_line_sub_v2v=get_sub_max_line_sub_v2v,
                get_sub_highlight_words_v2v=get_sub_highlight_words_v2v,
                get_sub_translator=get_sub_translator,
                deepl_auth_key_textbox=deepl_auth_key_textbox,
                get_sub_whisper_compute_time=get_sub_whisper_compute_time,
                get_sub_whisper_device=get_sub_whisper_device,
                get_sub_whisper_batch_size=get_sub_whisper_batch_size,
                get_sub_whisper_aline=get_sub_whisper_aline,
                output_folder=output_dir
            )
        done_message = f"Done, subtitles saved in {folder_name} folder"
    else:
        subs = transcribe_for_gradio(
            subtitle_file_get=input_file,
            get_sub_whisper_model=get_sub_whisper_model,
            get_sub_source_lang=get_sub_source_lang,
            get_sub_max_width_sub_v2v=get_sub_max_width_sub_v2v,
            get_sub_max_line_sub_v2v=get_sub_max_line_sub_v2v,
            get_sub_highlight_words_v2v=get_sub_highlight_words_v2v,
            get_sub_translator=get_sub_translator,
            deepl_auth_key_textbox=deepl_auth_key_textbox,
            get_sub_whisper_compute_time=get_sub_whisper_compute_time,
            get_sub_whisper_device=get_sub_whisper_device,
            get_sub_whisper_batch_size=get_sub_whisper_batch_size,
            get_sub_whisper_aline=get_sub_whisper_aline,
            output_folder=output_folder
        )
        done_message = "Done"

    return [subs,done_message]


subtitle_get_btn.click(fn=get_subs, 
                       inputs=[subtitle_file_get, subtitle_file_batch_get, subtitle_file_batch_get_path, 
                               get_sub_whisper_model, get_sub_source_lang, get_sub_max_width_sub_v2v, 
                               get_sub_max_line_sub_v2v, get_sub_highlight_words_v2v, get_sub_translator,
                               deepl_auth_key_textbox, get_sub_whisper_compute_time, get_sub_whisper_device,
                               get_sub_whisper_batch_size, get_sub_whisper_aline,get_sub_status_label],
                       outputs=[subtitle_output_get,get_sub_status_label])