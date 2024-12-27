import sys
sys.path.append("..")
from umbrella.models.auto_model import AutoModelLM
import argparse
import time
import torch
import os
from umbrella.speculation.speculation_utils import make_causal_mask
os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct",help='model')
parser.add_argument('--T', type=int, default=200, help='repeat times')
parser.add_argument('--P', type=int, default=512, help='prefix length')
parser.add_argument('--M', type=int, default=2048, help='max length')
parser.add_argument('--D', type=int, default=1, help='dec length')
parser.add_argument('--offload', action='store_true', help="offload the model")
parser.add_argument('--cuda_graph', action='store_true', help="whether use cuda graph")
args = parser.parse_args()
print(args)
PREFIX_LEN = args.P
MAX_LEN = args.M
DEC_LEN = args.D
MODEL_NAME = args.model
DTYPE = torch.float16
DEVICE = "cuda:0"
T = args.T
WARM_UP = 3

llm = AutoModelLM.from_pretrained(model_name=args.model, offload=args.offload, cuda_graph=args.cuda_graph, max_length=MAX_LEN,dtype=DTYPE, device=DEVICE)
llm.alloc()
if args.cuda_graph:
    llm.initialize_cuda_graph([DEC_LEN])
input_ids = torch.randint(low=3, high=30000, size=(1, PREFIX_LEN), device=DEVICE)
attention_mask = make_causal_mask((MAX_LEN, MAX_LEN), device=DEVICE)
position_ids = torch.arange(PREFIX_LEN, device=DEVICE).unsqueeze(0)
prefix_storage_ids = torch.arange(PREFIX_LEN, device=DEVICE)
llm.graph_inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[:PREFIX_LEN,:], storage_ids=prefix_storage_ids)

input_ids = torch.randint(low=3, high=30000, size=(1, DEC_LEN), device=DEVICE)
storage_ids = torch.arange(DEC_LEN, device=DEVICE) + PREFIX_LEN
position_ids = storage_ids.clone().unsqueeze(0)
attention_mask = attention_mask[PREFIX_LEN: PREFIX_LEN + DEC_LEN,:].clone()

for _ in range(WARM_UP):
    llm.graph_inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, storage_ids=storage_ids)
torch.cuda.synchronize()
t1 = time.time()
for _ in range(T):
    llm.graph_inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, storage_ids=storage_ids)
torch.cuda.synchronize()
t2 = time.time()

print("Max Length :{}, Decode Length :{}, Prefix Length :{}, inference time:{}s".format(MAX_LEN, DEC_LEN, PREFIX_LEN, (t2 - t1)/ T))