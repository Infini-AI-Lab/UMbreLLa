# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, MistralForCausalLM
from umbrella.speculation.speculation_utils import make_causal_mask, is_sentence_complete_regex, find_first_element_position

DEVICE = "cuda:0"
MAX_LEN = 2048

attention_mask = make_causal_mask((MAX_LEN, MAX_LEN), DEVICE)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3", torch_dtype=torch.float16, _attn_implementation="eager").to(DEVICE)

# # tokenizer.add_special_tokens({'pad_token_id': '[PAD]'})
# tokenizer.padding_side = 'right'
# tokenizer.add_eos_token = True
# tokenizer.pad_token_id=2041
# eos_token_id=tokenizer.eos_token_id
# model.resize_token_embeddings(len(tokenizer))
# model.config.pad_token_id = tokenizer.pad_token_id

# model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3", torch_dtype=torch.float16, '''_attn_implementation="eager"''' max_length=MAX_LEN, attention_mask=attention_mask).to("cuda:0")
# text = "Tell me what you know about Reinforcement Learning in 100 words."
text = "<s>[INST] Tell me what you know about Reinforcement Learning in 100 words.[/INST]"

# messages = [{"role": "user", "content": text}]

# # Modified template application
# prompt = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True  # Critical for response triggering
# )


input_ids = tokenizer.encode(text=text, return_tensors="pt").to(DEVICE)

# input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# prefix_len = input_ids.shape[1]

output = model.generate(input_ids, do_sample=False, max_new_tokens=512)
print(tokenizer.decode(output[0], skip_special_tokens=True))
