import sys
sys.path.append("..")
import json
import os.path as osp
import ssl
import urllib.request
import os
from umbrella.speculation.speculation_engine import SpeculationEngine
from umbrella.logging_config import setup_logger
from umbrella.utils import TextColors
from umbrella.templates import Prompts, SysPrompts
logger = setup_logger()

import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--configuration', type=str, default="../configs/chat_config.json",help='the configuration of the chatbot')
args = parser.parse_args()

DEVICE = "cuda:0"
with open(args.configuration, "r") as f:
    config = json.load(f)
GEN_LEN = config.pop("generation_length", 256)
MAX_LEN = config.pop("max_length", 8192)
MAX_TURNS = config.pop("max_turns", 16)
draft_model_name = config.pop("draft_model", "meta-llama/Llama-3.2-1B-Instruct")
target_model_name = config.pop("model", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
template = config.pop("template", "meta-llama3")
system_prompt = SysPrompts[template]
user_prompt = Prompts[template]

def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_jsonl(
    file_path,
):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict

test_filepath = os.path.join("data/", "question.jsonl")
print(f"Loading data from {test_filepath} ...")

if not os.path.exists(test_filepath):
    download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            "data/",
        )
    os.rename(os.path.join("data/", "question.jsonl"), test_filepath)

prompts = load_jsonl(test_filepath)

engine = SpeculationEngine(
    draft_model_name=draft_model_name,
    target_model_name=target_model_name,
    device=DEVICE,
    max_length=MAX_LEN,
    **config
)
engine.initialize()


large_model_steps = 0
total_time = 0
total_decode_tokens = 0

category_large_model_steps = {}
category_total_time = {}
category_total_decode_tokens = {}
for prompt in prompts:
    category_large_model_steps[prompt['category']] = 0
    category_total_time[prompt['category']] = 0
    category_total_decode_tokens[prompt['category']] = 0

for idx, prompt in enumerate(prompts):
    print(TextColors.colorize("Question ID: {}".format(prompt["question_id"]), 'white'))
    inputs = user_prompt.format(prompt["turns"][0])
    inputs = system_prompt + inputs
    engine.prefill(inputs)
    print(TextColors.colorize(prompt["turns"][0], 'green'))
    num_tokens, decode_time, step = engine.speculative_decoding(max_new_tokens=GEN_LEN)
    total_time += decode_time
    total_decode_tokens += num_tokens
    large_model_steps += step
    
    category_large_model_steps[prompt['category']] += step
    category_total_time[prompt['category']] += decode_time
    category_total_decode_tokens[prompt['category']] += num_tokens
    
    if len(prompt["turns"]) > 1:
        
        inputs = user_prompt.format(prompt["turns"][1])
        engine.append(inputs)
        print(TextColors.colorize(prompt["turns"][1], 'green'))
        num_tokens, decode_time, step = engine.speculative_decoding(max_new_tokens=GEN_LEN)
        total_time += decode_time
        total_decode_tokens += num_tokens
        large_model_steps += step
        
        category_large_model_steps[prompt['category']] += step
        category_total_time[prompt['category']] += decode_time
        category_total_decode_tokens[prompt['category']] += num_tokens
        
    engine.reset()

for category in category_large_model_steps.keys():

        logger.info(TextColors.colorize("{} | Avg Accept Tokens {:.2f} | TPOT {:.2f} ms ".format(
        category,
        category_total_decode_tokens[category]/category_large_model_steps[category], 
        1000 * category_total_time[category]/category_total_decode_tokens[category]
        ), "cyan"))
        
logger.info(TextColors.colorize("Summary | Avg Accept Tokens {:.2f} | TPOT {:.2f} ms ".format(total_decode_tokens/large_model_steps, 1000 * total_time/total_decode_tokens), "cyan"))
