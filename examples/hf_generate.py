# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct",help='model')
parser.add_argument('--G', type=int, default=512, help='generation length')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, _attn_implementation="eager").to("cuda:0")
text = "Tell me what you know about Reinforcement Learning in 100 words."
input_ids = tokenizer.encode(text=text, return_tensors="pt").to("cuda:0")

output = model.generate(input_ids, do_sample=False, max_new_tokens=args.G)
print(tokenizer.decode(output[0], skip_special_tokens=True))