'''
python SFT4-1.py
'''

import os
import wandb
import torch
import unsloth
import random
import numpy as np
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel, is_bfloat16_supported  
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
from unsloth.chat_templates import train_on_responses_only
from transformers import TrainingArguments, TextStreamer, DataCollatorForSeq2Seq, EarlyStoppingCallback
from trl import SFTTrainer 
from datasets import load_dataset, concatenate_datasets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(3407)

wandb.init(project="statistical-llm-sft", name="SFT4-1")

### model set
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="SFT3",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=12,
    lora_alpha=24,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "gate_proj"], 
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
)


tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.2",
)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    mask = labels != -100
    loss = torch.nn.functional.cross_entropy(
        torch.tensor(logits).view(-1, tokenizer.vocab_size), 
        torch.tensor(labels).view(-1),
        reduction="none"
    )
    loss = loss.view(labels.shape)
    loss = loss[mask].mean().item()
    perplexity = torch.exp(torch.tensor(loss)).item()
    return {"perplexity": perplexity}


def apply_template(examples):
    messages = examples["conversations"]
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}

dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = standardize_sharegpt(dataset)
dataset = dataset.shuffle().select(range(50000))
dataset = dataset.map(apply_template, batched=True, remove_columns=["conversations"])


trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=4,
    packing=False,
    compute_metrics=compute_metrics,
    args=TrainingArguments(
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=200,
        optim="adamw_8bit",
        weight_decay=0.05,
        warmup_ratio=0.03,
        output_dir="output",
        seed=3407,
        run_name="SFT_statistics_chat1",
        report_to="wandb",
        greater_is_better=False,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

trainer_stats = trainer.train()

model_location = "SFT4-1"
save_path = "SFT4-1-adapter"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
tokenizer.push_to_hub(model_location, token='')
model.push_to_hub_merged(model_location, tokenizer, save_method='merged_16bit', token='')


print("Training completed and model saved successfully!")
