'''
python SFT_v3.4_DPO_DTFT_v4.py
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

wandb.init(project="statistical-llm-sft", name="SFT_v3.4_DPO_DTFT_v4")

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


###################################### Read CV and KG data


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


dataset1 = load_dataset("json", data_files = "mix_data_CV_KG.json", split = "train")
dataset1 = dataset1.shuffle()
dataset1 = dataset1.map(apply_template, batched=True, remove_columns=["0", "1"])

###################################### Read SC data

def extract_and_apply_template_batched(examples):
    formatted_texts = []
    num_examples_in_batch = len(examples[next(iter(examples))]) 
    max_possible_turns = 20
    for i in range(num_examples_in_batch):
        messages = []
        for turn_index in range(max_possible_turns):
            turn_key = str(turn_index)
            turn_data = examples[turn_key][i]
            if turn_data is not None and isinstance(turn_data, dict) and "role" in turn_data and "content" in turn_data:
                 if isinstance(turn_data["role"], str) and isinstance(turn_data["content"], str):
                     messages.append(turn_data)
                 else:
                     break
            else:
                break

        if messages:
            try:
                formatted_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                formatted_texts.append(formatted_text)
            except Exception as e:
                print(f"Error applying chat template for example index {i}: {e}")
                formatted_texts.append(None)
        else:
             formatted_texts.append(None)
    return {"text": formatted_texts}

dataset2 = load_dataset("json", data_files = "stat_conversation_data.json", split = "train")
dataset2 = dataset2.shuffle()
dataset2 = dataset2.map(extract_and_apply_template_batched, batched=True, remove_columns=dataset2.column_names)


dataset = concatenate_datasets([dataset1, dataset2])
dataset = dataset.shuffle()

print(f"- Cross Validated data: 9071")
print(f"- Knowledge graph data: 8414")
print(f"- Stat Conversation data: {len(dataset2) if dataset2 else 0}")
print(f"- Mix data: {len(dataset) if dataset else 0}")

####################################### Trainer Setup

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    warmup_steps=50,
    max_steps=180,
    learning_rate=5e-7,
    lr_scheduler_type="linear",
    fp16=True,
    logging_steps=30,
    optim="adamw_8bit", 
    weight_decay=0.01, 
    seed=SEED,
    output_dir="SFT_v3.4_DPO_DTFT_v4",
    report_to="wandb", 
    run_name="SFT_v3.4_DPO_DTFT_v4",
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

model_location = "SFT_v3.4_DPO_DTFT_v4"
tokenizer.push_to_hub(model_location, token='')
model.push_to_hub_merged(model_location, tokenizer, save_method='merged_16bit', token='')

print("Training completed and model saved successfully!")
