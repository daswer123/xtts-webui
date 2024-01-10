from xtts_webui import *
import shutil

from datetime import datetime
from scripts.funcs import resemble_enhance_audio, save_audio_to_wav
import glob


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
