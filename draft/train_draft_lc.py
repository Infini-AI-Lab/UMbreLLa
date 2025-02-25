import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoModelForCausalLM, AutoConfig, LlamaForCausalLM
from datasets import load_dataset
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="mistralai/Mistral-Small-24B-Instruct-2501",help='tokenizer')
parser.add_argument('--output_dir', type=str, default="mistral",help='output directory')
parser.add_argument('--bsz', type=int, default=4, help='generation length')
args = parser.parse_args()

model_name = args.model
tokenizer = AutoTokenizer.from_pretrained(model_name, _attn_implementation="flash_attention_2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
total_params = sum(p.numel() for p in model.parameters())
print(f"total_params: {total_params:,}")


train_raw_datasets = load_dataset("qywu/slimpajama_long",split="train")

eval_raw_datasets = load_dataset("qywu/slimpajama_long",split="validation")

# 定义预处理函数：对句子对进行编码
def preprocess_function(examples):
    
    
    output = tokenizer(
        examples["text"],
        truncation=True,
        max_length=8192,
        padding="max_length"
    )

    return output

train_tokenized_datasets = train_raw_datasets.map(preprocess_function, batched=True, num_proc=8)
eval_tokenized_datasets = eval_raw_datasets.map(preprocess_function, batched=True, num_proc=8)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=args.output_dir,
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
    save_steps=1000,
    save_total_limit=4,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=1000,
    num_train_epochs=1,
    gradient_accumulation_steps=4
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
