import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from umbrella.speculation.dynamic_speculation_engine import DynamicSpeculationEngine
from umbrella.speculation.static_speculation_engine import StaticSpeculationEngine
from umbrella.templates import Prompts, SysPrompts
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",help='model')
parser.add_argument('--draft_model', type=str, default="meta-llama/Llama-3.2-1B-Instruct",help='draft model')
parser.add_argument('--template', type=str, default="meta-llama3",help='prompt template')
parser.add_argument('--G', type=int, default=512, help='generation length')
parser.add_argument('--offload', action='store_true', help="offload the model")
args = parser.parse_args()
print(args)

template = args.template
system_prompt = SysPrompts[template]
user_prompt = Prompts[template]

MODEL_NAME = args.model
DEVICE = "cuda:0"
GEN_LEN = args.G
draft_model_name = args.draft_model
target_model_name = args.model

engine = DynamicSpeculationEngine(
    draft_model_name=draft_model_name,
    target_model_name=target_model_name,
    device=DEVICE,
    max_length=8192,
    num_cache_layers=24
) if args.offload else StaticSpeculationEngine(
    draft_model_name=draft_model_name,
    target_model_name=target_model_name,
    device=DEVICE,
    max_length=2048,
    cuda_graph=True,
    growmap_path="../umbrella/trees/sequoia_tree-3x4.json"
)


text1 = "Tell me what you know about Reinforcement Learning in 100 words."
text2 = "Tell me what you know about LSH in 100 words."

text1 = user_prompt.format(text1)
text1 = system_prompt + text1
text2 = user_prompt.format(text2)

engine.initialize()

engine.prefill(text1)
engine.speculative_decoding(max_new_tokens=GEN_LEN)

engine.append(text2)
engine.speculative_decoding(max_new_tokens=GEN_LEN)

engine.reset()

