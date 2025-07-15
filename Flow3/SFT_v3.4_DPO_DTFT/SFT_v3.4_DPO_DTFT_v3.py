'''
python SFT_v3.4_DPO_DTFT_v3.py
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

wandb.init(project="statistical-llm-sft", name="SFT_stat_v1_4_DPO_Final3")

### Model Settings
max_seq_length = 2048
dtype = None 
load_in_4bit = True

# Load Model and Tokenizer using the Instruct version
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="SFT_v3.4_DPO", 
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token = "",
)

# --- LoRA Configuration ---
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16, 
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"], 
    use_rslora=True, # Rank Stable LoRA
    use_gradient_checkpointing="unsloth",
    bias="none", 
)


# ##################################### Read data


def apply_template(examples):
    users = examples["0"]
    assistants = examples["1"]
    messages = []
    for u, a in zip(users, assistants):
        temps = []
        temps.append(u)
        temps.append(a)
        messages.append(temps)
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text,}


dataset = load_dataset("json", data_files = "mix_data_CV_KG.json", split = "train")
dataset = dataset.shuffle()
dataset = dataset.map(apply_template, batched=True, remove_columns=["0", "1"])


print(f"- Cross Validated data: 9071")
print(f"- Knowledge graph data: 8414")
print(f"- Mix data: {len(dataset) if dataset else 0}")

# ###################################### Trainer Setup

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    warmup_steps=50,
    max_steps=300,
    learning_rate=5e-7,
    lr_scheduler_type="linear",
    fp16=True,
    logging_steps=10,
    optim="adamw_8bit", 
    weight_decay=0.01, 
    seed=SEED,
    output_dir="SFT_v3.4_DPO_DTFT_v3",
    report_to="wandb", 
    run_name="SFT_v3.4_DPO_DTFT_v3",
    save_strategy="epoch",
)

# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
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

model_location = "SFT_v3.4_DPO_DTFT_v3"
tokenizer.push_to_hub(model_location, token='')
model.push_to_hub_merged(model_location, tokenizer, save_method='merged_16bit', token='')

print("Training completed and model saved successfully!")
