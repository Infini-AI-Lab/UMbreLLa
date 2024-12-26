import sys
sys.path.append("..")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from umbrella.models.auto_model import AutoModelLM
import torch
from umbrella.templates import Prompts, SysPrompts
from transformers import AutoTokenizer
from umbrella.speculation.speculation_utils import make_causal_mask_hf, make_causal_mask
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct",help='model')
parser.add_argument('--template', type=str, default="meta-llama3",help='prompt template')
parser.add_argument('--offload', action='store_true', help="offload the model")
parser.add_argument('--cuda_graph', action='store_true', help="whether use cuda graph")
args = parser.parse_args()
DEVICE = "cuda:0"
DTYPE = torch.float16
MAX_LEN = 2048
GEN_LEN = 256
template = args.template
system_prompt = SysPrompts[template]
user_prompt = Prompts[template]

text = "Tell me what you know about Reinforcement Learning in 100 words."
text = user_prompt.format(text)
text = system_prompt + text

tokenizer = AutoTokenizer.from_pretrained(args.model)
tokens = tokenizer.encode(text=text, return_tensors="pt").to(DEVICE)

llm = AutoModelLM.from_pretrained(
    model_name=args.model,
    offload=args.offload,
    cuda_graph=args.cuda_graph,
    batch_size=1,
    max_length=MAX_LEN,
    dtype=DTYPE,
    device=DEVICE
)
llm.alloc()
if args.cuda_graph:
    llm.initialize_cuda_graph([1])
attention_mask = make_causal_mask((MAX_LEN, MAX_LEN), DEVICE)
storage_ids = torch.arange(MAX_LEN, device=DEVICE)
position_ids = torch.arange(MAX_LEN, device=DEVICE).unsqueeze(0)

prefix_len = tokens.shape[1]
logits = llm.graph_inference(input_ids=tokens, position_ids=position_ids[:,:prefix_len], 
              storage_ids=storage_ids[:prefix_len], attention_mask=attention_mask[:prefix_len])[0]

generated_tokens = []
for i in range(GEN_LEN):
    next_token = logits[-1:].argmax(dim=-1, keepdim=True)
    generated_tokens.append(next_token.item())
    logits = llm.graph_inference(input_ids=next_token, position_ids=position_ids[:,prefix_len+i:prefix_len+i+1], 
              storage_ids=storage_ids[prefix_len+i : prefix_len+i+1], attention_mask=attention_mask[prefix_len+i:prefix_len+i+1])[0]

output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(output)