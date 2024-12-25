import sys
sys.path.append("..")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from umbrella.speculation.speculation_engine import SpeculationEngine
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",help='model')
parser.add_argument('--draft_model', type=str, default="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",help='draft model')
parser.add_argument('--G', type=int, default=64, help='generation length')
args = parser.parse_args()
print(args)

MODEL_NAME = args.model
DEVICE = "cuda:0"
GEN_LEN = args.G
draft_model_name = args.draft_model
target_model_name = args.model

engine = SpeculationEngine(
    draft_model_name=draft_model_name,
    target_model_name=target_model_name,
    device=DEVICE,
    max_length=8192,
    num_cache_layers=16
)


text1 = "<|begin_of_text|><|start_header_id|>User: <|end_header_id|>Hello, tell me what you know about China in 200 words. <|eot_id|>\n<|start_header_id|>Assistant:<|end_header_id|>"

text2 = "\n\n<|start_header_id|>User: <|end_header_id|>Then tell me the relationship between China and Japan in 200 words. <|eot_id|>\n<|start_header_id|>Assistant:<|end_header_id|>"

engine.initialize()

engine.prefill(text2)
engine.speculative_decoding(max_new_tokens=512)

engine.append(text1)
engine.speculative_decoding(max_new_tokens=512)

engine.reset()

