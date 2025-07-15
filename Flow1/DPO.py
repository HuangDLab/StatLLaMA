'''
python DPO.py
'''

import os
import wandb
import torch
import unsloth
import random
import numpy as np
from datasets import load_dataset, concatenate_datasets

from unsloth import PatchDPOTrainer
PatchDPOTrainer()

from trl import SFTTrainer, DPOTrainer, DPOConfig
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

wandb.init(project="statistical-llm-sft", name="Flow1_DPO")

### Model Settings
max_seq_length = 2048
dtype = None
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Flow_SFT", 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none", 
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False, 
    loftq_config = None,
)

### load data
def convert_dpo_example(example, tokenizer):
    user_prompt = example["prompt"]
    chosen_response = example["chosen"]
    rejected_response = example["rejected"]
    chosen_messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": chosen_response}
    ]
    rejected_messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": rejected_response}
    ]
    chs = tokenizer.apply_chat_template(
        chosen_messages, tokenize=False, add_generation_prompt=True
    )
    rejs = tokenizer.apply_chat_template(
        rejected_messages, tokenize=False, add_generation_prompt=True
    )
    return {
        "chosen": chs,
        "rejected": rejs
    }

dataset1 = load_dataset("json", data_files = "stat_DPO_data.json", split = "train")
dataset1 = dataset1.map(lambda x: convert_dpo_example(x, tokenizer), remove_columns=["prompt", "model_note", "tags"])

dataset2 = load_dataset("kira/math-dpo", split = "train")
dataset2 = dataset2.map(lambda x: convert_dpo_example(x, tokenizer), remove_columns=["metadata", "chosen_rating", "rejected_rating", "prompt"])

dataset = concatenate_datasets([dataset1, dataset2])
dataset = dataset.shuffle(seed=SEED)

print(f"- Statistic DPO: {len(dataset1) if dataset1 else 0}")
print(f"- Math DPO: {len(dataset2) if dataset2 else 0}")

print(f"- Final DPO: {len(dataset) if dataset else 0}")


### Train

dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = DPOConfig(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 2,
        learning_rate = 5e-6,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 20,
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "Flow1_DPO",
        report_to = "wandb", 
    ),
    beta = 0.1,
    train_dataset = dataset,
    # eval_dataset = datasets["test"],
    tokenizer = tokenizer,
    max_length = 1024,
    max_prompt_length = 512,
)

trainer_stats = dpo_trainer.train()

model_location = "Flow1_DPO"
tokenizer.push_to_hub(model_location, token='')
model.push_to_hub_merged(model_location, tokenizer, save_method='merged_16bit', token='')

print("Training completed and model saved successfully!")
