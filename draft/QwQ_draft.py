import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer', type=str, default="Qwen/QwQ-32B",help='tokenizer')
parser.add_argument('--output_dir', type=str, default="QwQ-1.5B",help='output directory')
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("llamafactory/OpenR1-Math-94k", split="train")


def format_prompt(data):
    template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}"
    return template.format(data["messages"][1]["content"], data["messages"][2]["content"])

def tokenize_function(examples):
    return tokenizer(format_prompt(examples), truncation=True, padding="max_length", max_length=32768)

tokenized_dataset = dataset.map(tokenize_function, remove_columns=["messages"], num_proc=32)

model = AutoModelForCausalLM.from_pretrained("agentica-org/DeepScaleR-1.5B-Preview", _attn_implementation="flash_attention_2")

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    weight_decay=1e-4,
    lr_scheduler_type="cosine",
    logging_steps=1,
    save_only_model=True,
    save_safetensors=True,
    save_steps=1000,
    bf16=True,
    save_strategy="steps",
    warmup_ratio=0.01
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 开始训练
trainer.train()