import whisperx
from yt_dlp import YoutubeDL
import os

def download_audio(url, output_folder, filename=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Подготовим шаблон имени выходного файла
    outtmpl = f'{filename}.%(ext)s' if filename else '%(title)s.%(ext)s'
    outtmpl = os.path.join(output_folder, outtmpl)

    options = {
        'format': 'bestaudio/best',
        'extractaudio': True,
        'audioformat': "mp3",  # Запрос на конвертацию в mp3
        'outtmpl': outtmpl,
        'noplaylist': True,
        # Добавляем пост-процессор для принудительной конвертации в mp3
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',  # Можешь выбрать другое качество аудио
         }],
    }

    with YoutubeDL(options) as ydl:
        info_dict = ydl.extract_info(url, download=True)

    # Теперь мы уверены что файл будет сохранен именно как mp3
    audio_title = filename if filename else info_dict.get('title', None)

    saved_path = f"{output_folder}/{audio_title}.mp3"

    return saved_path



def whisperx_work(
# TOKEN
  hf_token,
# INPUT DATA
  input_audio,
  output_folder=".",
  output_format="all",
# WHISPER SETTINGS
  model="medium",
  model_dir=None,
  compute_type="float16",
  device="cuda",
  device_index=0,
  batch_size=8,
#   task="transcribe",
  timestamp_enable = True,
  timestamp_highlight_enable = False,
  language=None,
#   ALIGN SETTINGS
  align_enable=False,
  interpolate_method="nearest",
  return_char_alignments=False,
#   VAD
  vad_onset=0.500,
  vad_offset=0.363,
  chunk_size=30,
# DIARIZE PARAMS
  diarize_enable = False,
  diarize_split_enable = False,
  diarize_max_speakers = None,
  dizarize_min_speakers = None,
# Advance Wihsper settings
  temperature = 0,
  best_of =  5,
  beam_size = 5,
  patience = 1.0,
  length_penalty = 1.0,
  suppress_tokens="-1",
  suppress_numerals=False,
  initial_prompt=None,
  condition_on_previous_text=False,
  temperature_increment_on_fallback=0.2,
  compression_ratio_threshold = 2.4,
  logprob_threshold = 1.0,
  no_speech_threshold = 0.6,
  max_line_width = None,
  max_line_count = None,
  threads = 4,
):
    return False