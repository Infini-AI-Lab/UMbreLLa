import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer', type=str, default="mistralai/Mistral-Small-24B-Instruct-2501",help='tokenizer')
parser.add_argument('--config', type=str, default="./config.json",help='model config')
parser.add_argument('--bsz', type=int, default=4, help='generation length')
args = parser.parse_args()

config = AutoConfig.from_pretrained(args.config)
model_name = args.tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_config(config)

train_data_files= [
        "train/chunk1/example_train_10*.jsonl.zst", 
        "train/chunk1/example_train_11*.jsonl.zst", 
        "train/chunk1/example_train_12*.jsonl.zst",
        "train/chunk1/example_train_13*.jsonl.zst",
        "train/chunk1/example_train_14*.jsonl.zst"
]
train_raw_datasets = load_dataset("cerebras/SlimPajama-627B", data_files=train_data_files, split="train")

eval_data_files= ["validation/chunk1/example_holdout_*.jsonl.zst"]
eval_raw_datasets = load_dataset("cerebras/SlimPajama-627B", data_files=eval_data_files, split="train")

# 定义预处理函数：对句子对进行编码
def preprocess_function(examples):
    
    
    output = tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )

    return output

train_tokenized_datasets = train_raw_datasets.map(preprocess_function, batched=True, num_proc=8)
eval_tokenized_datasets = eval_raw_datasets.map(preprocess_function, batched=True, num_proc=8)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-4,
    per_device_train_batch_size=args.bsz,
    per_device_eval_batch_size=args.bsz,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    logging_dir='./logs',
    logging_steps=10,
    bf16=True,
    save_only_model=True,
    save_steps=5000,
    save_total_limit=2,
    eval_strategy="steps",
    save_strategy="steps"
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=eval_tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 开始训练
trainer.train()
