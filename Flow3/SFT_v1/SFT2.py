'''
python SFT2.py
'''

import torch
import unsloth
import random
import numpy as np
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported  
from transformers import TrainingArguments, TextStreamer, DataCollatorForSeq2Seq
from trl import SFTTrainer 
from datasets import load_dataset, concatenate_datasets
import wandb
import os
from transformers import EarlyStoppingCallback

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(3407)

wandb.init(project="statistical-llm-sft", name="SFT2")

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="SFT1",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  
    lora_alpha=32,
    lora_dropout=0.05,  
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"], 
    use_rslora=True,  
    use_gradient_checkpointing="unsloth",
)

prompt_templates = [
    """Below are instructions that describe a task in statistics and mathematics, followed by a specific problem to solve.
Please provide a comprehensive solution that demonstrates expert knowledge in statistical concepts and mathematical reasoning.

### Instruction:
You are a professor of statistics and mathematics with expertise in data analysis, probability theory, hypothesis testing, 
regression analysis, statistical inference, machine learning algorithms, and research methodology. Your goal is to provide 
clear, rigorous, and thorough solutions to complex statistical and mathematical problems.

When solving problems, ensure you:
- Identify the key statistical concepts involved
- State any assumptions that underpin your approach
- Provide step-by-step mathematical derivations
- Interpret results in context of the problem
- Discuss implications and potential applications

### Problem:
{}

### Solution:
{}
""",
    
    """# Statistical Problem Solving

## Problem Statement:
{}

## Detailed Solution:
As a statistician approaching this problem, I'll walk through my thought process and solution step-by-step.

{}

## Key Concepts Applied:
- Statistical inference methods
- Probability distributions
- Hypothesis testing framework
- Statistical modeling principles
- Interpretation of statistical findings
""",
    
    """Statistical Analysis Request
===========================

PROBLEM DESCRIPTION:
{}

STATISTICAL SOLUTION:
{}

PRACTICAL APPLICATIONS:
The statistical methods used in this solution can be applied to similar problems in data science, research, quality control, and decision-making processes. Understanding these statistical concepts is valuable across multiple disciplines including economics, healthcare, engineering, and business analytics.
""",
    
    """Data Analysis Task
-----------------

Question:
{}

Analysis Process and Results:
```
{}
```

This analysis follows best practices in statistical methodology and data science workflows. The solution provided is based on established statistical theory and computational approaches commonly used in professional data analysis.
""",
    
    """Statistical Research Problem

Abstract: This document presents a solution to a statistical problem using rigorous mathematical reasoning and established statistical methods.

1. PROBLEM FORMULATION
----------------------
{}

2. METHODOLOGY AND SOLUTION
---------------------------
{}

3. CONCLUSION
-------------

The above solution demonstrates the application of fundamental statistical principles to solve the given problem. The methodology can be extended to more complex scenarios and provides insights into the underlying statistical phenomena.

Keywords: statistical analysis, mathematical reasoning, probability theory, statistical inference
""",
]

EOS_TOKEN = tokenizer.eos_token 

difficulty_levels = ["Introductory", "Intermediate", "Advanced", "Research-Level"]

def formatting_prompts_func(examples):
    problems = examples["problem"]
    steps = examples["steps"]
    texts = []
    
    for problem, step in zip(problems, steps):
        template = random.choice(prompt_templates)
        
        intros = [
            "Let's solve this statistical problem: ",
            "Consider the following statistical challenge: ",
            "Here's a problem for analysis: ",
            "Examine this statistical question: ",
            ""  
        ]
        
        problem = random.choice(intros) + problem
        
        if random.random() < 0.15:
            contexts = [
                "This problem appears in a data analysis context.",
                "This question comes from a research study.",
                "This is a common scenario in statistical practice.",
                "This is an example we might encounter in real-world data.",
                "This represents a typical statistical challenge."
            ]
            problem = problem + "\n\n" + random.choice(contexts)
            
        text = template.format(problem, step) + EOS_TOKEN
        texts.append(text)
    return {"text" : texts,}

def formatting_prompts_func2(examples):
    problems = examples["problem"]
    solutions = examples["solution"]
    texts = []
    
    for problem, solution in zip(problems, solutions):
        template = random.choice(prompt_templates)
        
        if random.random() < 0.1:
            math_domains = ["Algebra", "Calculus", "Linear Algebra", "Discrete Mathematics", "Number Theory"]
            domain = random.choice(math_domains)
            difficulty = random.choice(difficulty_levels)
            problem = f"[Domain: {domain}] [Difficulty: {difficulty}]\n{problem}"
            
        text = template.format(problem, solution) + EOS_TOKEN
        texts.append(text)
    return {"text": texts,}

dataset1 = load_dataset("json", data_files = "stat_cot_data.json", split = "train")
dataset1 = dataset1.shuffle(seed=42)

train_size = int(0.9 * len(dataset1))
eval_dataset = dataset1.select(range(train_size, len(dataset1)))
dataset1 = dataset1.select(range(train_size))
dataset1 = dataset1.map(formatting_prompts_func, batched=True, remove_columns=["problem", "steps"])

dataset2 = load_dataset("json", data_files = "SFT_data/math_QA_100000.json", split = "train")
dataset2 = dataset2.shuffle(seed=42).select(range(20000))  
dataset2 = dataset2.map(formatting_prompts_func2, batched=True, remove_columns=["problem", "solution"])

eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True, remove_columns=["problem", "steps"])

dataset1 = concatenate_datasets([dataset1] * 15) 
dataset = concatenate_datasets([dataset1, dataset2])
dataset = dataset.shuffle(seed=42)

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

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=eval_dataset,  
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,
    compute_metrics=compute_metrics, 
    args=TrainingArguments(
        learning_rate=2e-5,  
        lr_scheduler_type="cosine", 
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  
        per_device_eval_batch_size=2,
        num_train_epochs=3,  
        eval_strategy="steps", 
        eval_steps=10,
        save_strategy="steps",
        save_steps=200,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=200,
        optim="adamw_8bit",
        weight_decay=0.05,  
        warmup_ratio=0.03,
        output_dir="output",
        seed=3407,
        run_name="SFT2",
        report_to="wandb",
        load_best_model_at_end=True,  
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)], 
)


trainer.train()

model_location = "SFT2"
tokenizer.push_to_hub(model_location, token='')
model.push_to_hub_merged(model_location, tokenizer, save_method='merged_16bit', token='')

print("Training completed and model saved successfully!")
