import sys
sys.path.append("..")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset
from umbrella.sequoia_utils import measure_acceptance_rate, generate_sequoia_tree
from umbrella.templates import SysPrompts, Prompts
from umbrella.models.auto_model import AutoModelLM
from umbrella.speculation.speculation_utils import make_causal_mask
import argparse
import time
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct",help='model')
parser.add_argument('--draft_model', type=str, default="Zhuominc/FastCode-500M",help='model')
parser.add_argument('--offload', action='store_true', help="offload the model")
parser.add_argument('--cuda_graph', action='store_true', help="whether use cuda graph")
parser.add_argument('--w', type=int, default=8, help="whether use cuda graph")
args = parser.parse_args()

system_prompt = SysPrompts['llama3-code']
prompt = Prompts['llama3-code']
data = load_dataset("openai/openai_humaneval")



DEVICE = "cuda:0"
DTYPE = torch.float16
MAX_LEN = 2048

draft_model = AutoModelLM.from_pretrained(
    model_name=args.draft_model,
    offload=False,
    cuda_graph=True,
    batch_size=1,
    max_length=MAX_LEN,
    dtype=DTYPE,
    device=DEVICE
)
draft_model.alloc()
target_model = AutoModelLM.from_pretrained(
    model_name=args.model,
    offload=False,
    cuda_graph=False,
    batch_size=1,
    max_length=MAX_LEN,
    dtype=DTYPE,
    device=DEVICE
)
target_model.alloc()
tokenizer = AutoTokenizer.from_pretrained(args.model)
attention_mask = make_causal_mask((MAX_LEN, MAX_LEN), DEVICE)
storage_ids = torch.arange(MAX_LEN, device=DEVICE)
position_ids = torch.arange(MAX_LEN, device=DEVICE).unsqueeze(0)

total_tokens = 0
acceptance_rate = torch.zeros((args.w,), device=DEVICE)
for d in tqdm(data['test']):
    context = system_prompt + prompt.format("Help me complete the code.") + d['prompt']
    solution = d['canonical_solution']
    input_ids_context = tokenizer.encode(context, return_tensors="pt").to(DEVICE)
    input_ids_solution = tokenizer.encode(solution, return_tensors="pt").to(DEVICE)
    input_ids = torch.cat([input_ids_context, input_ids_solution[:,1:]], dim=-1)
    
    
    prefix_len = input_ids.shape[1]
    solution_len = input_ids_solution.shape[1] - 1
    
    logits_target = target_model.graph_inference(input_ids=input_ids, position_ids=position_ids[:,:prefix_len], 
              storage_ids=storage_ids[:prefix_len], attention_mask=attention_mask[:prefix_len])[0, -solution_len - 1: -1]
    
    logits_draft = draft_model.graph_inference(input_ids=input_ids, position_ids=position_ids[:,:prefix_len], 
              storage_ids=storage_ids[:prefix_len], attention_mask=attention_mask[:prefix_len])[0, -solution_len - 1: -1]
    
    total_tokens += solution_len
    
    topk_indices = logits_draft.topk(dim=-1, k=args.w).indices
    target_indices = logits_target.topk(dim=-1, k=1).indices
    
    acceptance = (topk_indices == target_indices)
    acceptance_rate += acceptance.float().sum(dim=0)
    
    target_model.clear()
    draft_model.clear()
    


generate_sequoia_tree(width=5, depth=6, acc=acceptance_rate.tolist(), json_file="../umbrella/trees/8b_sequoia_tree-5x6.json")
generate_sequoia_tree(width=5, depth=8, acc=acceptance_rate.tolist(), json_file="../umbrella/trees/8b_sequoia_tree-5x8.json")
generate_sequoia_tree(width=6, depth=6, acc=acceptance_rate.tolist(), json_file="../umbrella/trees/8b_sequoia_tree-6x6.json")
generate_sequoia_tree(width=6, depth=7, acc=acceptance_rate.tolist(), json_file="../umbrella/trees/8b_sequoia_tree-6x7.json")