import sys
sys.path.append("..")
from umbrella.speculation.auto_engine import AutoEngine
from umbrella.logging_config import setup_logger
from umbrella.utils import TextColors
from umbrella.templates import Prompts, SysPrompts
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--configuration', type=str, default="../configs/chat_config_24gb.json",help='the configuration of the chatbot')
args = parser.parse_args()
logger = setup_logger()


DEVICE = "cuda:0"
with open(args.configuration, "r") as f:
    config = json.load(f)

GEN_LEN = config.pop("generation_length", 256)
MAX_TURNS = config.pop("max_turns", 16)
template = config.pop("template", "meta-llama3")
system_prompt = SysPrompts[template]
user_prompt = Prompts[template]

engine = AutoEngine.from_config(DEVICE, **config)
engine.initialize()

for i in range(MAX_TURNS):
    if i == 0:
        logger.info(TextColors.colorize("Start Infini AI Chatbot", "cyan"))
        prompt = input(TextColors.colorize("User: ", "blue"))
        if prompt == "BYE":
            logger.info(TextColors.colorize("Terminate Infini AI Chatbot. Thanks for using. Bye!", "cyan"))
            break
        prompt = user_prompt.format(prompt)
        prompt = system_prompt + prompt
        print(TextColors.colorize("Assistant:", "blue"), end=" ")
        engine.prefill(prompt)
        engine.speculative_decoding(max_new_tokens=GEN_LEN)
    else:
        prompt = input(TextColors.colorize("User: ", "blue"))
        if prompt == "BYE":
            logger.info(TextColors.colorize("Terminate Infini AI Chatbot. Thanks for using. Bye!", "cyan"))
            break
        prompt = user_prompt.format(prompt)
        print(TextColors.colorize("Assistant:", "blue"), end=" ")
        engine.append(prompt)
        engine.speculative_decoding(max_new_tokens=GEN_LEN)
    
    if not engine.validate_status():
        logger.info(TextColors.colorize("Exceeding Maximum Contexts. Terminate Infini AI Chatbot. Thanks for using. Bye!", "cyan"))
        break
    
