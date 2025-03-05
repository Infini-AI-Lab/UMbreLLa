import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from umbrella.engine.ar_engine import AREngine
from umbrella.templates import Prompts, SysPrompts
import argparse
import torch
from transformers import AutoTokenizer
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct",help='model')
parser.add_argument('--template', type=str, default="meta-llama3",help='prompt template')
parser.add_argument('--G', type=int, default=512, help='generation length')
parser.add_argument('--offload', action='store_true', help="offload the model")
args = parser.parse_args()
print(args)


MODEL_NAME = args.model
DEVICE = "cuda:0"
torch.cuda.set_device(DEVICE)
GEN_LEN = args.G

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
engine = AREngine(
    model_name=MODEL_NAME,
    device=DEVICE,
    max_length=35000,
    kv_budget = 256,
    local_budget = 1024,
    full_layers = [0,1,14,21],
    cache_config="h2o"
)

engine.initialize()

MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

text = r"Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."

coversation = [
    {"role": "user", "content":MATH_QUERY_TEMPLATE.format(Question=text)}
]

templated_texts = tokenizer.apply_chat_template(conversation=coversation, add_generation_prompt=True, tokenize=False)
engine.prefill(text)
engine.decoding(max_new_tokens=GEN_LEN)
