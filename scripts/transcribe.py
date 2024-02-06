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



def whisperx_work(hf_token,
  input_audio,
  output_folder,
  model_name,
  compute_type,
  device,
  vad_options,
  batch_size,
  language,
  align_enable,
  timestamp_enable,
  timestamp_highlight_enable,
  diarize_enable,
  diarize_split_enable,
  diarize_speakers,
  diarize_max_speakers,
  dizarize_min_speakers,
):
    model = whisperx.load_model(model_name,device=device,compute_type=compute_type,language=language,vad_options=vad_options)
    
    print("Whisper model loaded")
    audio = whisperx.load_audio(input_audio)
    print("Audio loaded")
    # self, audio: Union[str, np.ndarray], batch_size=None, num_workers=0, language=None, task=None, chunk_size=30, print_progress = False, combined_progress=False
    result = whisperx.transcribe(audio, batch_size=batch_size,language=language)
    
    language = result["language"]
    
    if align_enable:
        print("Start aling result")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=whisper_device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, whisper_device, return_char_alignments=False)
    
    if diarize_enable:
        # 3. Assign speaker labels
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device,min_speakers=dizarize_min_speakers, max_speakers=diarize_max_speakers)

        # add min/max number of speakers if known
        diarize_segments = diarize_model(audio)
        # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        result = whisperx.assign_word_speakers(diarize_segments, result)

    print(result)