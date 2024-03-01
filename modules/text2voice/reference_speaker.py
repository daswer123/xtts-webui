import os

import gradio as gr
from pathlib import Path

import shutil
import uuid

from scripts.funcs import save_audio_to_wav, resample_audio, improve_ref_audio, resemble_enhance_audio, cut_audio

from xtts_webui import *

# FUNCS
# SPEAKERS LIST FUNCS

def update_speakers_list(speakers_list, speaker_value_text, speaker_path_text, speaker_ref_wavs):
    speakers_list = XTTS.get_speakers()
    speaker_value = ""
    if not speakers_list:
        speakers_list = ["None"]
        speaker_value = "None"
    else:
        speaker_value = speakers_list[0]
        speaker_value_text = speaker_value

    if speaker_path_text:
        speakers_list.append("reference")

    if speaker_ref_wavs:
        speakers_list.append("multi_reference")

    return gr.Dropdown(label="Reference Speaker from folder 'speakers'", value=speaker_value, choices=speakers_list), speaker_value_text


def switch_speaker_example_visibility(ref_speaker_example, show_ref_speaker_from_list, ref_speaker_list, speaker_value_text):
    if show_ref_speaker_from_list and speaker_value_text != "reference" and speaker_value_text != "multi_reference":
        speaker_path = XTTS.get_speaker_sample(ref_speaker_list)
        if speaker_path == None:
            return gr.Audio(visible=False, value=None)
        return gr.Audio(label="Speaker Example", value=speaker_path, visible=True)
    else:
        return gr.Audio(label="Speaker Example", visible=False, value=None)

# SINGLE REFERENCE FUNCS


def change_current_speaker(ref_speaker_list, speaker_value_text, show_ref_speaker_from_list):
    XTTS.speaker_wav = ref_speaker_list
    speaker_value_text = ref_speaker_list

    if show_ref_speaker_from_list:
        speaker_path = XTTS.get_speaker_sample(ref_speaker_list)
        print(speaker_path)
        if speaker_value_text == None:
            return ref_speaker_list, speaker_value_text, gr.Audio(visible=False, label="Speaker Example")
        return ref_speaker_list, speaker_value_text, gr.Audio(label="Speaker Example", value=speaker_path, visible=True)

    return ref_speaker_list, speaker_value_text, None


def clear_current_speaker_audio(ref_speaker_audio, speaker_path_text, ref_speaker_list, speaker_value_text, speaker_ref_wavs):
    speakers_list = XTTS.get_speakers()
    speaker_value = ""
    if not speakers_list:
        speakers_list = ["None"]
        speaker_value = "None"
    else:
        speaker_value = speakers_list[0]
        speaker_value_text = speaker_value

    if speaker_ref_wavs:
        speakers_list.append("multi_reference")
        speaker_value = "multi_reference"

    return "", speaker_value, gr.Dropdown(label="Reference Speaker from folder 'speakers'", value=speaker_value, choices=speakers_list), gr.Button(visible=False)


def change_current_speaker_audio(improve_reference_resemble, improve_reference_audio, auto_cut, ref_speaker_audio, speaker_path_text, ref_speaker_list, speaker_value_text, use_resample, speaker_ref_wavs):
    #  Get audio, save it to tempo and resample it.
    output_path = ""
    rate, y = ref_speaker_audio
    output_path = save_audio_to_wav(rate, y, this_dir, auto_cut)

    if improve_reference_resemble:
        output_path = resemble_enhance_audio(output_path, True)
    if use_resample:
        output_path = resample_audio(output_path, this_dir)

    if improve_reference_audio:
        output_path = improve_ref_audio(output_path, this_dir)

#  Assign the invisible variables the values we will use for generation
    XTTS.speaker_wav = output_path
    speaker_path_text = output_path

  # Get the list of speakers to assign a reference and select it in the list
  #  Add a reference item that will operate on the example given by the user
    speakers_list = XTTS.get_speakers()
    speaker_value_text = "reference"

    speakers_list.append(speaker_value_text)

    if speaker_ref_wavs:
        speakers_list.append("multi_reference")

    return output_path, speaker_path_text, "reference", gr.Dropdown(label="Reference Speaker from folder 'speakers'", value="reference", choices=speakers_list), gr.Button(value="Save a single sample for the speaker", visible=True)


# MULTIPLE REFERENCE FUNCS
def create_multiple_reference(ref_speakers, use_resample=False, improve_reference_audio=False, auto_cut=0, improve_reference_resemble=False, speaker_value_text=""):
    ref_dir = Path(f'temp/reference_speaker_{uuid.uuid4()}')
    ref_dir.mkdir(parents=True, exist_ok=True)

    # Processing and moving new files.
    processed_files = []
    for index, speaker_file in enumerate(ref_speakers):
        current_file = Path(speaker_file)

        # Save the original file name without extension
        original_filename = current_file.stem

        if auto_cut > 0:
            current_file = cut_audio(current_file, duration=auto_cut)

        if improve_reference_resemble:
            current_file = Path(resemble_enchance_audio(current_file, True))

        # Apply resampling if the use_resample flag is set.
        if use_resample:
            current_file = Path(resample_audio(current_file, this_dir=ref_dir))

        # Improve audio if the improve_reference_audio flag is set.
        if improve_reference_audio:
            current_file = Path(improve_ref_audio(
                current_file, this_dir=ref_dir))

        # The first file will be called preview.wav
        # new_location_name = "preview.wav" if index == 0 else f"{original_filename}.wav"
        new_location_name = f"{original_filename}.wav"
        new_location = ref_dir / new_location_name
        processed_files.append(new_location)
        try:
            # Move the prepared file to the target directory.
            shutil.move(str(current_file), str(new_location))
            print(f'Processed and moved {current_file} to {new_location}')
        except Exception as e:
            print(f'Failed to process or move {current_file}. Reason: {e}')
    print(f'Processed {len(processed_files)} files.')
    processed_files = [str(path) for path in processed_files]

    # Add ref to list and select it
    speakers_list = XTTS.get_speakers()
    speaker_value = "multi_reference"
    speakers_list.append("multi_reference")

    if speaker_value_text:
        speakers_list.append("reference")

    return processed_files, processed_files, speaker_value, gr.Dropdown(label="Reference Speaker from folder 'speakers'", value=speaker_value, choices=speakers_list), gr.Button(value="Save multiple samples for the speaker", visible=True)


def clear_multiple_reference(ref_speaker):
    speakers_list = XTTS.get_speakers()
    speaker_value = ""
    if not speakers_list:
        speakers_list = ["None"]
        speaker_value = "None"
    else:
        speaker_value = speakers_list[0]
        speaker_value_text = speaker_value

    if ref_speaker:
        speakers_list.append("reference")
        speaker_value = "reference"

    return None, "", speaker_value, gr.Dropdown(label="Reference Speaker from folder 'speakers'", value=speaker_value, choices=speakers_list), gr.Button(visible=False)


def show_inbuild_voices(show_inbuildstudio_speaker):
    speakers_list = XTTS.get_speakers(True)
    speaker_value = ""
    
    if not speakers_list:
        speakers_list = ["None"]
        speaker_value = "None"
    else:
        speaker_value = speakers_list[0]
        XTTS.speaker_wav = speaker_value
                        
    return gr.Dropdown(label=i18n("Reference Speaker from folder 'speakers'"), value=speaker_value, choices=speakers_list)


# HANDLERS
# REFERENCE LIST
show_inbuildstudio_speaker.change(fn=show_inbuild_voices,inputs=[show_inbuildstudio_speaker],outputs=[ref_speaker_list])

ref_speaker_list.change(fn=change_current_speaker,
                        inputs=[ref_speaker_list, speaker_value_text,
                                show_ref_speaker_from_list],
                        outputs=[ref_speaker_list, speaker_value_text, ref_speaker_example])

update_ref_speaker_list_btn.click(fn=update_speakers_list,
                                  inputs=[ref_speaker_list, speaker_value_text,
                                          speaker_path_text, speaker_ref_wavs],
                                  outputs=[ref_speaker_list, speaker_value_text])

show_ref_speaker_from_list.change(fn=switch_speaker_example_visibility,
                                  inputs=[ref_speaker_example, show_ref_speaker_from_list,
                                          ref_speaker_list, speaker_value_text],
                                  outputs=[ref_speaker_example])

# Translate Ref list
# REFERENCE LIST
translate_ref_speaker_list.change(fn=change_current_speaker,
                        inputs=[translate_ref_speaker_list, speaker_value_text,
                                translate_show_ref_speaker_from_list],
                        outputs=[translate_ref_speaker_list, speaker_value_text, translate_ref_speaker_example])

translate_update_ref_speaker_list_btn.click(fn=update_speakers_list,
                                  inputs=[translate_ref_speaker_list, speaker_value_text,
                                          speaker_path_text, speaker_ref_wavs],
                                  outputs=[translate_ref_speaker_list, speaker_value_text])

translate_show_ref_speaker_from_list.change(fn=switch_speaker_example_visibility,
                                  inputs=[translate_ref_speaker_example, translate_show_ref_speaker_from_list,
                                          translate_ref_speaker_list, speaker_value_text],
                                  outputs=[translate_ref_speaker_example])

# REFERENCE SINGLE UPLOAD OR MICROPHONE
ref_speaker.stop_recording(fn=change_current_speaker_audio,
                           inputs=[improve_reference_resemble, improve_reference_audio, auto_cut, ref_speaker,
                                   speaker_path_text, speaker_value_text, ref_speaker_list, use_resample, speaker_ref_wavs],
                           outputs=[ref_speaker, speaker_path_text, speaker_value_text, ref_speaker_list, save_speaker_btn])

ref_speaker.upload(fn=change_current_speaker_audio,
                   inputs=[improve_reference_resemble, improve_reference_audio, auto_cut, ref_speaker,
                           speaker_path_text, speaker_value_text, ref_speaker_list, use_resample, speaker_ref_wavs],
                   outputs=[ref_speaker, speaker_path_text, speaker_value_text, ref_speaker_list, save_speaker_btn])

ref_speaker.clear(fn=clear_current_speaker_audio,
                  inputs=[ref_speaker, speaker_path_text,
                          speaker_value_text, ref_speaker_list, speaker_ref_wavs],
                  outputs=[speaker_path_text, speaker_value_text, ref_speaker_list, save_speaker_btn])

# REFERENCE MULTIPLE UPLOAD
ref_speakers.upload(fn=create_multiple_reference,
                    inputs=[ref_speakers, use_resample, improve_reference_audio,
                            auto_cut, improve_reference_resemble, speaker_value_text],
                    outputs=[ref_speakers, speaker_ref_wavs, speaker_value_text, ref_speaker_list, save_multiple_speaker_btn])

ref_speakers.clear(fn=clear_multiple_reference,
                   inputs=[ref_speaker],
                   outputs=[ref_speakers, speaker_ref_wavs, speaker_value_text, ref_speaker_list, save_multiple_speaker_btn])
