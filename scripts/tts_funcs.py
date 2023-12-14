# tts.py

import torch
import torchaudio

from TTS.api import TTS

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from pathlib import Path

from scripts.modeldownloader import download_model,check_tts_version

from loguru import logger
import os
import time 
import re


# List of supported language codes
supported_languages = {
    "ar":"Arabic",
    "pt":"Brazilian Portuguese",
    "zh-cn":"Chinese",
    "cs":"Czech",
    "nl":"Dutch",
    "en":"English",
    "fr":"French",
    "de":"German",
    "it":"Italian",
    "pl":"Polish",
    "ru":"Russian",
    "es":"Spanish",
    "tr":"Turkish",
    "ja":"Japanese",
    "ko":"Korean",
    "hu":"Hungarian",
    "hi":"Hindi"
}

reversed_supported_languages = {name: code for code, name in supported_languages.items()}

class TTSWrapper:
    def __init__(self,output_folder = "./output", speaker_folder="./speakers",lowvram = False,model_source = "local",model_version = "2.0.2",device = "cuda"):

        self.cuda = device # If the user has chosen what to use, we rewrite the value to the value we want to use
        self.device = 'cpu' if lowvram else (self.cuda if torch.cuda.is_available() else "cpu")
        self.lowvram = lowvram  # Store whether we want to run in low VRAM mode.

        self.latents_cache = {} 

        self.model_source = model_source
        self.model_version = model_version

        self.speaker_folder = speaker_folder
        self.output_folder = output_folder

        self.language = "en"
        self.speaker_wav = ""
        self.available_model_versions = ["v2.0.3", "v2.0.2", "v2.0.1", "v2.0.0", "main"]

        
        self.create_directories()
        check_tts_version()
    
    def unload_model(self):
        self.model = None
        torch.cuda.empty_cache()
        logger.info("Model unloaded")
        
    def load_model(self,this_dir):
        if self.model_source == "api":
            # this_dir = Path(__file__).parent.resolve()
            if self.model_version in self.available_model_versions:
              download_model(this_dir,self.model_version)

            # this_dir = Path(__file__).parent.resolve()
            config_path = this_dir / 'models' / f'{self.model_version}' / 'config.json'
            checkpoint_dir = this_dir / 'models' / f'{self.model_version}'

            self.model = TTS(model_path=checkpoint_dir,config_path=config_path).to(self.device)

        if self.model_source == "local":
          self.load_local_model(this_dir)
          if self.lowvram == False:
            # Due to the fact that we create latents on the cpu and load them from the cuda we get an error
            logger.info("Pre-create latents for all current speakers")
            self.create_latents_for_all() 
          
        logger.info("Model successfully loaded ")
    
    def load_local_model(self,this_dir):
        # this_dir = Path(__file__).parent.resolve()
        if self.model_version in self.available_model_versions:
          download_model(this_dir,self.model_version)

        config = XttsConfig()
        config_path =  this_dir / 'models' / f'{self.model_version}' / 'config.json'
        checkpoint_dir = this_dir / 'models' / f'{self.model_version}'

        config.load_json(str(config_path))
        
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=str(checkpoint_dir))
        self.model.to(self.device)

    def switch_model_device(self):
        # We check for lowram and the existence of cuda
        if self.lowvram and torch.cuda.is_available() and self.cuda != "cpu":
            with torch.no_grad():
                if self.device == self.cuda:
                    self.device = "cpu"
                else:
                    self.device = self.cuda

                self.model.to(self.device)

            if self.device == 'cpu':
                # Clearing the cache to free up VRAM
                torch.cuda.empty_cache()

    def get_or_create_latents(self, speaker_wav):
        if speaker_wav not in self.latents_cache:
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(speaker_wav)
            self.latents_cache[speaker_wav] = (gpt_cond_latent, speaker_embedding)
        return self.latents_cache[speaker_wav]

    def create_latents_for_all(self):
        speakers_list = self.get_speakers()

        for speaker_name in speakers_list:
            speaker_wav = os.path.join(self.speaker_folder, speaker_name+".wav")

            self.get_or_create_latents(speaker_wav)

        logger.info(f"Latents created for all {len(speakers_list)} speakers.")

    def create_directories(self):
        directories = [self.output_folder, self.speaker_folder]

        for sanctuary in directories:
            # List of folders to be checked for existence
            absolute_path = os.path.abspath(os.path.normpath(sanctuary))

            if not os.path.exists(absolute_path):
                # If the folder does not exist, create it
                os.makedirs(absolute_path)
                print(f"Folder in the path {absolute_path} has been created")

    def set_speaker_folder(self, folder):
        if os.path.exists(folder) and os.path.isdir(folder):
            self.speaker_folder = folder
            self.create_directories()
            logger.info(f"Speaker folder is set to {folder}")
        else:
            raise ValueError("Provided path is not a valid directory")

    def set_out_folder(self, folder):
        if os.path.exists(folder) and os.path.isdir(folder):
            self.output_folder = folder
            self.create_directories()
            logger.info(f"Output folder is set to {folder}")
        else:
            raise ValueError("Provided path is not a valid directory")

    def list_speakers(self):
        speakers_list = [f for f in os.listdir(self.speaker_folder) if f.endswith('.wav')]
        return speakers_list

    def get_speakers(self):
        # Use os.path.splitext to split off the extension and take only the name
        speakers_list = [os.path.splitext(f)[0] for f in os.listdir(self.speaker_folder) if f.endswith('.wav')]
        return speakers_list
    # Special format for SillyTavern
    def get_speakers_special(self):
        speakers_list = []
        BASE_URL = os.getenv('BASE_URL', '127.0.0.1:8020')
        TUNNEL_URL = os.getenv('TUNNEL_URL', '')

        preview_url = "" 
        for file in os.listdir(self.speaker_folder):
            
            if TUNNEL_URL == "":
                preview_url = f"{BASE_URL}/sample/{file}"
            else:
                preview_url = f"{TUNNEL_URL}/sample/{file}"

            if file.endswith('.wav'):
                speaker_name = os.path.splitext(file)[0]
                speaker = {
                    'name': speaker_name,
                    'voice_id': speaker_name,
                    'preview_url': preview_url
                }
                speakers_list.append(speaker)
        return speakers_list
    
    def list_languages(self):
        return reversed_supported_languages

    def clean_text(self,text):
        # Remove asterisks and line breaks
        text = re.sub(r'[\*\r\n]', '', text)
        # Replace double quotes with single quotes and correct punctuation around quotes
        text = re.sub(r'"\s?(.*?)\s?"', r"'\1'", text)
        return text

    def local_generation(self,text,speaker_wav,language,options,output_file):
        # Log time
        generate_start_time = time.time()  # Record the start time of loading the model

        gpt_cond_latent, speaker_embedding = self.get_or_create_latents(speaker_wav)


        out = self.model.inference(
            text,
            language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=options["temperature"],
            length_penalty=options["length_penalty"],
            repetition_penalty=options["repetition_penalty"],
            top_k=options["top_k"],
            top_p=options["top_p"],
            enable_text_splitting=True,
            speed=options["speed"]
        )

        torchaudio.save(output_file, torch.tensor(out["wav"]).unsqueeze(0), 24000)

        generate_end_time = time.time()  # Record the time to generate TTS
        generate_elapsed_time = generate_end_time - generate_start_time

        logger.info(f"Processing time: {generate_elapsed_time:.2f} seconds.")

    def api_generation(self,text,speaker_wav,language,options,output_file):
        self.model.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                file_path=output_file,
                speed=options["speed"],
        )
    
    def get_speaker_path(self,speaker_name_or_path):
        # Check if the speaker path is a .wav file or just the name
        if speaker_name_or_path.endswith('.wav'):
                if os.path.isabs(speaker_name_or_path):
                    # If it's an absolute path for the speaker file
                    speaker_wav = speaker_name_or_path
                else:
                    # It's just a filename; append it to the speakers folder
                    speaker_wav = os.path.join(self.speaker_folder, speaker_name_or_path)
        else:
                # Look for the corresponding .wav in our list of speakers
                speakers_list = self.list_speakers()
                if f"{speaker_name_or_path}.wav" in speakers_list:
                    speaker_wav = os.path.join(self.speaker_folder, f"{speaker_name_or_path}.wav")
                else:
                    raise ValueError(f"Speaker {speaker_name_or_path} not found.")
        return speaker_wav


    def process_tts_to_file(self, text, language,ref_speaker_wav, options, file_name_or_path="out.wav"):
        try:
            speaker_wav = self.get_speaker_path(ref_speaker_wav)
            
            # Determine output path based on whether a full path or a file name was provided
            if os.path.isabs(file_name_or_path):
                # An absolute path was provided by user; use as is.
                output_file = file_name_or_path
            else:
                # Only a filename was provided; prepend with output folder.
                output_file = os.path.join(self.output_folder, file_name_or_path)

            # Replace double quotes with single, asterisks, carriage returns, and line feeds
            clear_text = self.clean_text(text)

            self.switch_model_device() # Load to CUDA if lowram ON

            # Define generation if model via api or locally
            if self.model_source == "local":
                self.local_generation(clear_text,speaker_wav,language,options,output_file)
            else:
                self.api_generation(clear_text,speaker_wav,language,options,output_file)
            
            self.switch_model_device() # Unload to CPU if lowram ON
            return output_file

        except Exception as e:
            raise e  # Propagate exceptions for endpoint handling.



if __name__ == "__main__":
    print("Nothing")
    # demo.launch(debug=True)   