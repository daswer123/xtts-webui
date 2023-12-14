from scipy.io import wavfile
import numpy as np
import os
import ffmpeg
import uuid
from pathlib import Path

# import noisereduce
# from pedalboard import Pedalboard,NoiseGate,LowpassFilter,Compressor,LowShelfFilter,Gain
# import soundfile as sf


def save_audio_to_wav(rate, y, this_dir, max_duration=None):
    audio_data = np.asarray(y, dtype=np.int16)
    temp_folder = this_dir / 'temp'

    os.makedirs(temp_folder, exist_ok=True)

    wav_name = f'speaker_ref_{uuid.uuid4()}.wav'

    original_wav_path = str(temp_folder / wav_name)  

     # Save the audio data to a file without changing the sampling rate.
    wavfile.write(original_wav_path, rate, audio_data)

    if max_duration is not None and max_duration != 0:
         output_wav_path = str(temp_folder / f'cut_{wav_name}')
         (
             ffmpeg.input(original_wav_path)
             .output(output_wav_path, to=max_duration)
             .run(overwrite_output=True)
         )
         os.remove(original_wav_path)  
         return output_wav_path

    return original_wav_path

def resample_audio(input_wav_path, this_dir, target_rate=24000):
    temp_folder = Path(this_dir) / 'temp'
    temp_folder.mkdir(parents=True, exist_ok=True)

    output_wav_name = f"resampled_audio_{uuid.uuid4()}.wav"
    output_wav_path = temp_folder / output_wav_name

    (
        ffmpeg
        .input(str(input_wav_path))
        .output(str(output_wav_path), ar=target_rate, acodec='pcm_s16le', ac=1)
        .run(overwrite_output=True)
     )

    return str(output_wav_path)

def move_and_rename_file(file_path, target_folder_path, new_file_name):
    # Make sure that the new file name contains the correct .wav extension
    if not new_file_name.lower().endswith('.wav'):
        new_file_name += '.wav'

    # Create Path objects for easy handling of paths
    file_path = Path(file_path)
    target_folder_path = Path(target_folder_path)

    # Creating a target directory if it does not exist
    target_folder_path.mkdir(parents=True, exist_ok=True)

    # Full path to the new file in the destination folder
    target_file_path = target_folder_path / new_file_name

    # Move and rename a file
    file_path.rename(target_file_path)

# WIP
# def improve_audio( input_file, output_file):
#         # Read audio data with soundfile
#         audio_data, sample_rate = sf.read(input_file)

#         # Assuming that 'audio_data' contains a NumPy array and 'sample_rate' holds the sampling rate

#         # Reduce noise from the audio signal
#         reduced_noise = noisereduce.reduce_noise(y=audio_data,
#                                                  sr=sample_rate,
#                                                  stationary=True,
#                                                  prop_decrease=0.75)

#         # Create a Pedalboard with effects applied to the audio
#         board = Pedalboard([
#             NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
#             Compressor(threshold_db=12, ratio=2.5),
#             LowShelfFilter(cutoff_frequency_hz=400, gain_db=5),
#             Gain(gain_db=0)
#         ])

#         # Process the noise-reduced signal through the pedalboard (effects chain)
#         result = board(reduced_noise.astype('float32'), sample_rate)

#         # Write processed audio data to output file using soundfile
#         sf.write(output_file, result.T if result.ndim > 1 else result , sample_rate)