
import os
import langid

import gradio as gr
from pathlib import Path
from scripts.funcs import improve_and_convert_audio,resemble_enchance_audio,str_to_list
from scripts.rvc_scripts import infer_rvc

import uuid

from xtts_webui import *

# HELP FUNCS
def predict_lang(text,selected_lang):
    language_predicted = langid.classify(text)[0].strip()  # strip need as there is space at end!

    # tts expects chinese as zh-cn
    if language_predicted == "zh":
            # we use zh-cn
            language_predicted = "zh-cn"
    
    # Check if language in supported langs
    if language_predicted not in supported_languages:
        language_predicted = selected_lang
        logger.warning(f"Language {language_predicted} not supported, using {supported_languages[selected_lang]}")

    return language_predicted

# GENERATION AND GENERATION OPTIONS
def switch_waveform(enable_waveform,video_gr):
    if enable_waveform:
        return gr.Video(label="Waveform Visual",visible=True,interactive=False)
    else:
        return  gr.Video(label="Waveform Visual",interactive=False,visible=False)

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
    # Batch
    batch_generation,
    batch_generation_path,
    ref_speakers,
    # Features
    language_auto_detect,
    enable_waveform,
    improve_output_audio,
    improve_output_resemble,
    improve_output_rvc,
    #  Default settings
    output_type,
    text,
    languages,
    # Help variables
    speaker_value_text,
    speaker_path_text,
    additional_text,
    # TTS settings
    temperature,length_penalty,
    repetition_penalty,
    top_k,
    top_p,
    speed,
    sentence_split):

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
        lang_code = predict_lang(text,lang_code)

    options = {
        "temperature": temperature,
        "length_penalty": length_penalty,
        "repetition_penalty": float(repetition_penalty),
        "top_k": top_k,
        "top_p": top_p,
        "speed": speed,
        "sentence_split": sentence_split,
    }
    # Find all .txt files in the filder and write this to batch_generation
    if batch_generation_path and Path(batch_generation_path).exists():
        batch_generation = [f for f in Path(batch_generation_path).glob('*.txt')]
        
    if batch_generation:
        for file_path in batch_generation:
            with open(file_path,encoding="utf-8",mode="r") as f:
                text = f.read()

                if language_auto_detect:
                    lang_code = predict_lang(text,lang_code)

                filename = os.path.basename(file_path)
                filename = filename.split(".")[0]

                output_file_path = f"{filename}_{additional_text}_{speaker_value_text}.{output_type}"
                output_file = XTTS.process_tts_to_file(text, lang_code, ref_speaker_wav, options, output_file_path)

                if improve_output_audio:
                   output_file = improve_and_convert_audio(output_file,output_type)

                if improve_output_rvc and rvc_settings_model_path:
                            temp_dir = this_dir / "output"
                            result = temp_dir / f"{speaker_value_text}_{rvc_settings_model_name}_{count}.{output_type}"
                            infer_rvc(rvc_settings_pitch,
                                rvc_settings_index_rate,
                                rvc_settings_protect_voiceless,
                                rvc_settings_method,
                                rvc_settings_model_path,
                                rvc_settings_index_path,
                                output_file,
                                result,
                                )
                output_file = result.absolute()

                if improve_output_resemble:
                    output_file = resemble_enchance_audio(**resemble_enhance_settings,audio_path=output_file,output_type=output_type)

        if enable_waveform:
            return gr.make_waveform(audio=output_file),output_file
        else:
            return None,output_file

    # Check if the file already exists, if yes, add a number to the filename
    count = 1
    output_file_path = f"{additional_text}_({count})_{speaker_value_text}.{output_type}"
    while os.path.exists(os.path.join('output', output_file_path)):
        count += 1
        output_file_path = f"{additional_text}_({count})_{speaker_value_text}.{output_type}"
    
    # Perform TTS and save to the generated filename
    output_file = XTTS.process_tts_to_file(text, lang_code, ref_speaker_wav, options, output_file_path)

    if improve_output_audio:
        output_file = improve_and_convert_audio(output_file,output_type)

    if improve_output_rvc and rvc_settings_model_path:
        temp_dir = this_dir / "output"
        result = temp_dir / f"{speaker_value_text}_{rvc_settings_model_name}_{count}.{output_type}"
        infer_rvc(rvc_settings_pitch,
                                rvc_settings_index_rate,
                                rvc_settings_protect_voiceless,
                                rvc_settings_method,
                                rvc_settings_model_path,
                                rvc_settings_index_path,
                                output_file,
                                result,
                                )
        output_file = result.absolute()

    if improve_output_resemble:
        output_file = resemble_enchance_audio(**resemble_enhance_settings,audio_path=output_file,output_type=output_type)

    if enable_waveform:
        return gr.make_waveform(audio=output_file),output_file
    else:
        return None,output_file



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
                        # Batch
                        batch_generation,
                        batch_generation_path,
                        speaker_ref_wavs,
                        # Features
                        language_auto_detect,
                        enable_waveform,
                        improve_output_audio,
                        improve_output_resemble,
                        improve_output_rvc,
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
                        sentence_split
                        ],
                    outputs=[video_gr, audio_gr]
                )


enable_waveform.change(
                    fn=switch_waveform,
                    inputs=[enable_waveform,video_gr],
                    outputs=[video_gr]
                )