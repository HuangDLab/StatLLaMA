'''
python Instruct.py
'''

import os
import wandb
import torch
import unsloth
import random
import copy
from trl import SFTTrainer
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template 
from unsloth.chat_templates import standardize_sharegpt
from unsloth.chat_templates import train_on_responses_only
from transformers import TrainingArguments, DataCollatorForSeq2Seq


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 3407
set_seed(SEED)

# Initialize WandB (consider a more descriptive name)
wandb.init(project="statistical-llm-sft", name="Flow2_Instruct")

### Model Settings
max_seq_length = 2048
dtype = None 
load_in_4bit = True

# Load Model and Tokenizer using the Instruct version
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Flow1_CoP",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"], 
    use_rslora=True, 
    use_gradient_checkpointing="unsloth", 
    bias="none", 
)

llama3_2_template = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}"
    "{% set content = '<|begin_of_text|>' + content %}"
    "{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)

if tokenizer.chat_template is None:
    print("Tokenizer chat_template is None. Setting Llama 3.2 template.")
    tokenizer.chat_template = llama3_2_template
else:
    print(f"Tokenizer chat_template is already set.")

####################################### 1. OpenHermes data
def remove_weight(conversations):
    if isinstance(conversations, list):
        return [{k: v for k, v in msg.items() if k != 'weight'} for msg in conversations]
    return conversations 
def apply_template(examples):
    messages = examples["conversations"]
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}

dataset0 = load_dataset("teknium/OpenHermes-2.5", split="train[:100000]")
new_data = []
for example in dataset0:
    example['conversations'] = remove_weight(example['conversations'])  # 移除每筆資料中的 weight
    new_data.append(example)

dataset1 = Dataset.from_dict({k: [d[k] for d in new_data] for k in new_data[0]})
dataset1 = standardize_sharegpt(dataset1)
dataset1 = dataset1.shuffle(seed = SEED)
dataset1 = dataset1.map(apply_template, batched=True, remove_columns=['custom_instruction', 'topic', 'model_name', 'model', 'skip_prompt_formatting', 'category', 'views', 'language', 'id', 'title', 'idx', 'hash', 'avatarUrl', 'system_prompt', 'source',
 'conversations'])


####################################### 2. Dolly data
def format_dolly_for_llama3_template(example):
    instruction = example.get('instruction', '').strip()
    context = example.get('context', '').strip()
    response = example.get('response', '').strip()
    output_text = None 
    if not instruction or not response:
        print(f"Warning: Skipping example due to missing instruction or response. Instruction: '{instruction}', Response: '{response}'")
        return {"text": None}
    if context:
        user_content = f"{instruction}\n\n{context}"
    else:
        user_content = instruction
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response}
    ]
    try:
        output_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        output_text = output_text.strip()
    except Exception as e:
        print(f"ERROR: Failed to apply chat template. Messages: {messages}. Error: {e}")
    return {"text": output_text}


dataset2 = load_dataset("databricks/databricks-dolly-15k", split="train")
original_columns = list(dataset2.features)
dataset2 = dataset2.map(
    format_dolly_for_llama3_template,
    batched=False,
    remove_columns=original_columns
)
dataset2 = dataset2.filter(lambda example: example["text"] is not None and len(example["text"]) > 0)
dataset2 = dataset2.shuffle(seed = SEED)


print(f"Combining datasets. Individual sizes before combining:")
print(f"  - OpenHermes: {len(dataset1) if dataset1 else 0}")
print(f"  - Dolly-15k: {len(dataset2) if dataset2 else 0}")

dataset = concatenate_datasets([dataset1, dataset2])
dataset = dataset.shuffle(seed=SEED)

print(f"- Final: {len(dataset) if dataset else 0}")



# ###################################### Trainer Setup

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=16, 
    warmup_ratio=0.03,
    num_train_epochs=1, 
    learning_rate=2e-5, 
    lr_scheduler_type="cosine",
    fp16=True,
    logging_steps=100, 
    optim="adamw_8bit", 
    weight_decay=0.05, 
    seed=SEED,
    output_dir="Flow2_Instruct",
    report_to="wandb", 
    run_name="Flow2_Instruct",
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


trainer_stats = trainer.train()

model_location = "Flow2_Instruct"
tokenizer.push_to_hub(model_location, token='')
model.push_to_hub_merged(model_location, tokenizer, save_method='merged_16bit', token='')

print("Training completed and model saved successfully!")
