from scripts.modeldownloader import install_deepspeed_based_on_python_version
from argparse import ArgumentParser
import os

parser = ArgumentParser(description="Run the Uvicorn server.")
parser.add_argument("-hs", "--host", default="localhost", help="Host to bind")
parser.add_argument("-p", "--port", default=8020, type=int, help="Port to bind")
parser.add_argument("-d", "--device", default="cuda", type=str, help="Device that will be used, you can choose cpu or cuda")
parser.add_argument("-sf", "--speaker_folder", default="speakers/", type=str, help="The folder where you get the samples for tts")
parser.add_argument("-o", "--output", default="output/", type=str, help="Output folder")
parser.add_argument("-ms", "--model-source", default="local", choices=["api","local"],
                    help="Define the model source: 'api' for latest version from repository, api inference or 'local' for using local inference and model v2.0.2.")
parser.add_argument("-v", "--version", default="v2.0.2", type=str, help="You can specify which version of xtts to use,This version will be used everywhere in local, api and apiManual.")
parser.add_argument("--lowvram", action='store_true', help="Enable low vram mode which switches the model to RAM when not actively processing.")
parser.add_argument("--deepspeed", action='store_true', help="Enable deepspeed acceleration, works on windows on python 3.10 and 3.11.")
parser.add_argument("--share", action='store_true', help="Allows the interface to be used outside the local computer.")

args = parser.parse_args()

os.environ['DEVICE'] = args.device  # Set environment variable for output folder.
os.environ['OUTPUT'] = args.output  # Set environment variable for output folder.
os.environ['SPEAKER'] = args.speaker_folder  # Set environment variable for speaker folder.
os.environ['BASE_URL'] = "http://" + args.host + ":" + str(args.port)  # Set environment variable for base url."
os.environ['MODEL_SOURCE'] = args.model_source  # Set environment variable for the model source
os.environ["LOWVRAM_MODE"] = str(args.lowvram).lower() # Set lowvram mode
os.environ["DEEPSPEED"] = str(args.deepspeed).lower() # Enable Streaming mode
os.environ["MODEL_VERSION"] = args.version # Specify version of XTTS model


# Check deepspeed
install_deepspeed_based_on_python_version()

from xtts_webui import demo   

demo.launch(share=args.share)   
    