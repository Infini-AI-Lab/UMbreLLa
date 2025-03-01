import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from umbrella.engine.ar_engine import AREngine
from umbrella.templates import Prompts, SysPrompts
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct",help='model')
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
model_name = args.model

engine = AREngine(
    model_name=model_name,
    device=DEVICE,
    max_length=8192
)

engine.initialize()

text1 = "Tell me what you know about Reinforcement Learning in 100 words."
text2 = "Tell me what you know about LSH in 100 words."

text1 = user_prompt.format(text1)
text1 = system_prompt + text1
text2 = user_prompt.format(text2)

engine.prefill(text1)
engine.decoding(max_new_tokens=GEN_LEN)

engine.append(text2)
engine.decoding(max_new_tokens=GEN_LEN)