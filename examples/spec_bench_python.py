import sys
sys.path.append("..")
import json
import os.path as osp
import ssl
import urllib.request
import os
from umbrella.speculation.dynamic_speculation_engine import DynamicSpeculationEngine
from umbrella.speculation.static_speculation_engine import StaticSpeculationEngine
from umbrella.speculation.auto_engine import AutoEngine
from umbrella.logging_config import setup_logger
from umbrella.utils import TextColors
from umbrella.templates import Prompts, SysPrompts
from datasets import load_dataset
logger = setup_logger()

import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--configuration', type=str, default="../configs/code_config_24gb.json",help='the configuration of the chatbot')
args = parser.parse_args()

DEVICE = "cuda:0"
with open(args.configuration, "r") as f:
    config = json.load(f)
GEN_LEN = config.pop("generation_length", 256)
MAX_TURNS = config.pop("max_turns", 16)
template = config.pop("template", "meta-llama3")
system_prompt = SysPrompts[template]
user_prompt = Prompts[template]

prompts = load_dataset("ananyarn/Algorithm_and_Python_Source_Code", split="train[:300]")


engine = AutoEngine.from_config(device=DEVICE, **config)
engine.initialize()


large_model_steps = 0
total_time = 0
total_decode_tokens = 0
filter_length = 20


for idx, prompt in enumerate(prompts):
    print(TextColors.colorize("Question ID: {}".format(idx), 'white'))
    
    inputs = user_prompt.format(prompt["Algorithm"])
    inputs = system_prompt + inputs
    
    engine.prefill(inputs)
    print(TextColors.colorize(prompt["Algorithm"], 'green'))
    num_tokens, decode_time, step = engine.speculative_decoding(max_new_tokens=GEN_LEN)
    if num_tokens >= filter_length:
        total_time += decode_time
        total_decode_tokens += num_tokens
        large_model_steps += step
    

        
    engine.reset()

        
logger.info(TextColors.colorize("Summary | Avg Accept Tokens {:.2f} | TPOT {:.2f} ms ".format(total_decode_tokens/large_model_steps, 1000 * total_time/total_decode_tokens), "cyan"))
