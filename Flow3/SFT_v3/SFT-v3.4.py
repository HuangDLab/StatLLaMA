'''
python SFT-v3.4.py
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

# Initialize WandB (consider a more descriptive name)
wandb.init(project="statistical-llm-sft", name="SFT_Exp1.4_Stat_GSM8K_FineTome")

### Model Settings
max_seq_length = 2048
dtype = None # Autodetect
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
    use_gradient_checkpointing="unsloth", # Recommended by Unsloth
    bias="none", 
)

####################################### 1. stat. defs/nouns
definition = [
    """Can you briefly define {}?""",
    """What is {}?""",
    """Could you explain {} in simple terms?""",
    """I’d like to understand {}. Could you define it?""",
    """Please provide a concise explanation of {}.""",
    """How would you describe {} in a way that's easy to understand?""",
    """What is the meaning of {} and how is it used?""",
    """Give me a short and clear definition of {}.""",
    """Explain {} as if I were new to this topic.""",
    """What is {} and why is it important?""",
    """Can you briefly define {} in simple terms that anyone can understand?""",
    """What is {} and how does it relate to other concepts in this field?""",
    """Could you explain {} in an easy-to-understand way, as if you were teaching a beginner?""",
    """I’d like to understand {} better. Can you provide a concise definition with some context?""",
    """Please provide a clear and simple explanation of {} along with its key features.""",
    """How would you describe {} in a way that highlights its importance and practical applications?""",
    """What does {} mean in this domain, and why is it a crucial concept?""",
    """Give me a short and precise definition of {}, ideally with an example if possible.""",
    """Explain {} in layman’s terms while maintaining technical accuracy.""",
    """What is {} and why is it significant in its respective field? Please elaborate with an example."""
]
definition1 = [
    """{} And how is it mathematically expressed? {}""",
    """{} Its mathematical representation is given by {}.""",
    """{} Here is the corresponding formula: {}.""",
    """The definition is as follows: {}. Additionally, its formula is {}.""",
    """{} can be defined as follows, and its equation is given by {}.""",
    """{} is an important concept, but how is it mathematically expressed? {}""",
    """{} has a formal definition, but its mathematical representation is given by {}.""",
    """{} is often described in theoretical terms. Here is its corresponding formula: {}.""",
    """The definition can be understood as follows: {}. Additionally, its mathematical formula is {}.""",
    """{} is widely used in different applications. It can be defined as follows, and its equation is {}.""",
    """When working with {}, one key aspect to consider is its mathematical formulation, which is given by {}.""",
    """To understand {}, we need to look at both its definition and its mathematical representation, which is {}.""",
    """{} is fundamental in this area of study. Along with its theoretical definition, the following formula is commonly used to describe it: {}.""",
    """Mathematically, {} can be expressed using the formula {}. This allows us to analyze its properties more rigorously.""",
    """A complete understanding of {} requires both a definition and a mathematical perspective. The standard formula associated with it is {}."""

]
definition2 = [
    """{} is an important concept, and it plays a fundamental role in various contexts.""",
    """{} is a key concept, and its significance can be fully understood through its definition.""",
    """{} is widely recognized in theory, and its importance stems from its clear and precise definition.""",
    """The definition is as follows: {}. This helps us grasp its core concept.""",
    """{} is used in various fields, and its value is rooted in the clarity of its definition.""",
    """The key to understanding {} is not only its definition but also its broader implications.""",
    """To understand {}, we must first define it clearly and explore its relevance in different areas.""",
    """{} is central to this topic, and its definition is foundational to the understanding of the field.""",
    """Understanding {} requires a clear definition, which provides the basis for further exploration.""",
    """A thorough understanding of {} involves grasping its definition and recognizing its impact."""
]


def apply_template1(examples):
    terms = examples["term"]
    descriptions = examples["description"]
    formulas = examples["formula"]
    codes = examples["code"]
    messages = []
    for term, description, formula, code in zip(terms, descriptions, formulas, codes):
        definitions = random.sample(definition, 3)
        definitions1 = random.sample(definition1, 3)
        definitions2 = random.sample(definition2, 3)
        for d, d1, d2 in zip(definitions, definitions1, definitions2):
            temp = []
            if term == 'R Code Example':
                continue
            elif formula == "":
                temp.append({"role": "user", "content": d.format(term)})
                temp.append({"role": "assistant", "content": d2.format(description)})
            else:
                temp.append({"role": "user", "content": d.format(term)})
                temp.append({"role": "assistant", "content": d1.format(description, formula)})

            messages.append(temp)
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text,}

dataset1 = load_dataset("json", data_files = "stat_noun_data.json", split = "train")
dataset1 = dataset1.shuffle()
dataset1 = dataset1.map(apply_template1, batched=True, remove_columns=["term", "description", "formula", "code"])


####################################### 2. Statistical CoT
def apply_template_stats_cot(examples):
    problems = examples["problem"]
    steps = examples["steps"]
    messages = []
    for problem, step in zip(problems, steps):
        temp = []
        temp.append({"role": "user", "content": problem})
        # Removed <reasoning> tags for cleaner SFT
        temp.append({"role": "assistant", "content": step})
        messages.append(temp)

    texts = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": texts}

dataset2 = load_dataset("json", data_files = "stat_cot_data.json", split = "train")
dataset2 = dataset2.shuffle(seed=SEED)
dataset2 = dataset2.map(apply_template_stats_cot, batched=True, num_proc=4, remove_columns=["problem", "steps"])
# Doubling the dataset as in your original code
dataset2 = concatenate_datasets([dataset2] * 2)
print(f"Dataset 2 (Stats CoT) size after processing (and doubling): {len(dataset2)}")


####################################### 3. Statistical CoT + QA 
def apply_template_stats_cot_qa(examples):
    questions = examples["question"]
    reasonings = examples["reasoning"]
    answers = examples["answer"]
    messages = []
    for question, reasoning, answer in zip(questions, reasonings, answers):
        temp = []
        temp.append({"role": "user", "content": question})
        # Removed <reasoning> and <answer> tags, combined content
        assistant_content = f"Reasoning:\n{reasoning}\n\nAnswer:\n{answer}"
        temp.append({"role": "assistant", "content": assistant_content})
        messages.append(temp)
    texts = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": texts}

dataset3 = load_dataset("json", data_files = "stat_GRPO_data.json", split = "train")
dataset3 = dataset3.shuffle(seed=SEED)
dataset3 = dataset3.map(apply_template_stats_cot_qa, batched=True, num_proc=4, remove_columns=["question", "reasoning", "answer"])
# Doubling the dataset as in your original code
dataset3 = concatenate_datasets([dataset3] * 2)
print(f"Dataset 3 (Stats CoT+QA) size after processing (and doubling): {len(dataset3)}")

# ###################################### 4. FineTome-100k (General Instructions)
# Using unsloth's utility for ShareGPT format
# from unsloth.chat_templates import standardize_sharegpt # Ensure imported

# def apply_template4(examples):
#     messages = examples["conversations"]
#     text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
#     return {"text": text}

# dataset4 = load_dataset("mlabonne/FineTome-100k", split="train")
# dataset4 = standardize_sharegpt(dataset4)
# dataset4 = dataset4.shuffle().select(range(20000))
# dataset4 = dataset4.map(apply_template4, batched=True, remove_columns=["conversations", 'source', 'score'])

####################################### 5. GSM8K
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


print(f"Combining datasets. Individual sizes before combining:")
print(f"  - Stats Nouns: {len(dataset1) if dataset1 else 0}")
print(f"  - Stats CoT: {len(dataset2) if dataset2 else 0}")
print(f"  - Stats CoT+QA: {len(dataset3) if dataset3 else 0}")
# print(f"  - FineTome: {len(dataset4) if dataset4 else 0}")
print(f"  - GSM8K: {len(dataset5) if dataset5 else 0}")

final_dataset = concatenate_datasets([dataset1, dataset2, dataset3, dataset5])
final_dataset = final_dataset.shuffle(seed=SEED)

print(f"- Final: {len(final_dataset) if final_dataset else 0}")

# ###################################### Trainer Setup

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16, 
    warmup_ratio=0.03,
    num_train_epochs=3,
    learning_rate=2e-5, 
    lr_scheduler_type="cosine",
    fp16=True,
    logging_steps=30, 
    optim="adamw_8bit", 
    weight_decay=0.05,
    seed=SEED,
    output_dir="outputs", 
    report_to="wandb",
    run_name="SFT-v3.4",
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

model_location = "SFT-v3.4"
tokenizer.push_to_hub(model_location, token='')
model.push_to_hub_merged(model_location, tokenizer, save_method='merged_16bit', token='')

print("Training completed and model saved successfully!")
