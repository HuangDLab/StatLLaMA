'''
python SFT1.py
'''

import torch
import unsloth
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
from unsloth.chat_templates import train_on_responses_only
from unsloth import FastLanguageModel, is_bfloat16_supported  
from transformers import TrainingArguments, TextStreamer, DataCollatorForSeq2Seq
from trl import SFTTrainer 
from datasets import load_dataset, concatenate_datasets



### model set
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"], 
    use_rslora=False,
    use_gradient_checkpointing="unsloth",
)

prompt = """Below are instructions that describe the task, along with inputs to provide more context.
Please write a response that appropriately completes this request.
Before answering, think about the question carefully and create a step-by-step chain of thought to ensure your response is logical and accurate.

### Instruction:
You are a mathematician and a statistician with professional knowledge in mathematics, data analysis, machine learning, model design, and predictive methods.
Please think carefully and answer the following statistical and mathematical questions step by step.

### Question:
{}

### Response:
{}
"""


EOS_TOKEN = tokenizer.eos_token 

def formatting_prompts_func(examples):
    problems = examples["problem"]
    steps = examples["steps"]
    texts = []
    for problem, step in zip(problems, steps):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = prompt.format(problem, step) + EOS_TOKEN
        texts.append(text)
    return {"text" : texts,}

dataset1 = load_dataset("json", data_files = "stat_cot_data.json", split = "train")
dataset1 = dataset1.shuffle(seed=42)
dataset1 = dataset1.map(formatting_prompts_func, batched = True,)

def formatting_prompts_func2(examples):
    problems = examples["problem"]
    solutions = examples["solution"]
    texts = []
    for problem, solution in zip(problems, solutions):
        text = prompt.format(problem, solution) + EOS_TOKEN
        texts.append(text)
    return {"text": texts,}

dataset2 = load_dataset("json", data_files = "math_QA_100000.json", split = "train")
dataset2 = dataset2.shuffle(seed=42).select(range(50000))
dataset2 = dataset2.map(formatting_prompts_func2, batched=True,)


dataset1 = concatenate_datasets([dataset1] * 10)
dataset = concatenate_datasets([dataset1, dataset2])
dataset = dataset.shuffle(seed=42)


trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=20,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=5,
        output_dir="output",
        seed=3407,
        run_name="SFT_noun",
        report_to="wandb",
    ),
)

trainer.train()

model_location = "SFT1"
model.save_pretrained_merged("model_save", tokenizer, save_method = "merged_16bit")
tokenizer.push_to_hub(model_location, token = '')
model.push_to_hub_merged(model_location, tokenizer, save_method = 'merged_16bit', token = '')







