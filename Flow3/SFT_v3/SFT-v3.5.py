'''
python SFT-v3.5.py
'''

import os
import wandb
import torch
import unsloth
import random
from trl import SFTTrainer
import numpy as np
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template # Keep if needed elsewhere, but apply_chat_template is used directly
from unsloth.chat_templates import standardize_sharegpt # Needed for FineTome
from unsloth.chat_templates import train_on_responses_only # Needed
from transformers import TrainingArguments, DataCollatorForSeq2Seq # Removed TextStreamer, added DataCollatorForSeq2Seq explicitly


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 3407
set_seed(SEED)

wandb.init(project="statistical-llm-sft", name="SFT_Exp1.4_Stat_GSM8K_FineTome")

### Model Settings
max_seq_length = 2048
dtype = None 
load_in_4bit = True

# Load Model and Tokenizer using the Instruct version
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct", 
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", # Standard attention targets
                    "gate_proj", "up_proj", "down_proj"], # MLP targets (often beneficial)
    use_rslora=True, # Rank Stable LoRA
    use_gradient_checkpointing="unsloth", 
    bias="none", 
)


####################################### 1. GSM8K
def apply_template_gsm8k(examples):
    questions = examples["question"]
    answers = examples["answer"] # Assumes 'answer' contains step-by-step reasoning + final answer
    messages = []
    for q, a in zip(questions, answers):
        temp = []
        temp.append({"role": "user", "content": q})
        # Answer field likely contains the full reasoning and result
        temp.append({"role": "assistant", "content": a})
        messages.append(temp)
    texts = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": texts}

dataset5 = load_dataset("gsm8k", "main", split="train")
dataset5 = dataset5.shuffle()
dataset5 = dataset5.map(apply_template_gsm8k, batched=True, remove_columns=["question", 'answer'])


final_dataset = concatenate_datasets([dataset5])
final_dataset = final_dataset.shuffle(seed=SEED)

print(f"- Final: {len(final_dataset) if final_dataset else 0}")

# ###################################### Trainer Setup

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16, 
    warmup_ratio=0.03,
    num_train_epochs=1, 
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    fp16=True,
    logging_steps=30,
    optim="adamw_8bit",
    weight_decay=0.05, 
    seed=SEED,
    output_dir="outputs", 
    report_to="wandb", 
    run_name="SFT-v3.5", 
    save_strategy="epoch", 
)

# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=final_dataset,
    dataset_text_field="text", 
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    dataset_num_proc=4,
    packing=False, 
    args=training_args,
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

trainer_stats = trainer.train()

model_location = "SFT-v3.5"
tokenizer.push_to_hub(model_location, token='')
model.push_to_hub_merged(model_location, tokenizer, save_method='merged_16bit', token='')

print("Training completed and model saved successfully!")
