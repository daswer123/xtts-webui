
import os
import langid

import gradio as gr
from pathlib import Path
from scripts.funcs import improve_and_convert_audio, resemble_enhance_audio, str_to_list
from scripts.voice2voice import infer_rvc, infer_openvoice, find_openvoice_ref_by_name

import uuid

from xtts_webui import *
import shutil

# HELP FUNCS


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

    # Find all .txt files in the filder and write this to batch_generation
    if batch_generation_path and Path(batch_generation_path).exists():
        batch_generation = [f for f in Path(
            batch_generation_path).glob('*.txt')]

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
