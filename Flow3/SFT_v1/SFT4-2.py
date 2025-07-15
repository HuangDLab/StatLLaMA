'''
python SFT4-2.py
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


wandb.init(project="statistical-llm-sft", name="SFT4-2")

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
    target_modules=["q_proj", "v_proj", "o_proj"], 
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
code_request = [
    """Given an {}, with the question being {}, generate the required code.""",
    """I need a code implementation for an {} where the question is {}.""",
    """Can you write a program for an {} based on the question: {}?""",
    """Generate a solution in code for an {} that addresses the following question: {}.""",
    """Write a script to solve an {} problem where the given question is: {}.""",
    """Given an {}, with the question being {}, please generate the required code in a suitable programming language.""",
    """I need a code implementation for an {} where the question is {}. Please ensure clarity and efficiency.""",
    """Can you write a well-structured program for an {} based on the question: {}? Try to make the solution scalable.""",
    """Generate a solution in code for an {} that addresses the following question: {}. It should be optimized for performance.""",
    """Write a script to solve an {} problem where the given question is: {}. Make sure the approach is logically sound.""",
    """Given an {} scenario, where the problem statement is {}, provide a code snippet with appropriate comments explaining its logic.""",
    """An {} is presented with the following problem: {}. Please write an efficient and well-documented code solution.""",
    """For an {} scenario, considering the problem statement {}, generate a clear and concise code solution in R (or another suitable language).""",
    """Given an {} and a problem description of {}, create a working code solution while ensuring best practices.""",
    """I need a code snippet for an {} where the problem statement is {}. Ensure that the code is structured for easy readability and maintainability."""
]
code_response = [
    """Here is the code you requested: {}.""",
    """The following is the required implementation: {}.""",
    """The solution in code is given below: {}.""",
    """I have written the code as per your request: {}.""",
    """You can use the following code snippet to solve the problem: {}.""",
    """Here is the code you requested to solve the problem: {}. Let me know if you need any modifications.""",
    """The following is the required implementation for the given problem: {}. It should work efficiently for the provided input.""",
    """The solution in code is given below: {}. Feel free to ask for improvements or explanations.""",
    """I have written the requested code based on the problem statement. Here it is: {}. Let me know if any refinements are needed.""",
    """You can use the following code snippet to solve the problem: {}. It follows best practices and should be easy to understand.""",
    """Based on the problem requirements, here is a well-structured code solution: {}. Hope this helps!""",
    """I’ve generated the necessary code for your request: {}. Please review it and let me know if you need additional modifications.""",
    """The code snippet provided here {} should correctly address the given problem. Let me know if further optimization is needed.""",
    """Here is a working solution for the given problem statement: {}. If you need an explanation, I’d be happy to provide one.""",
    """The implementation you need is as follows: {}. This should handle the given case efficiently."""
]


def apply_template(examples):
    terms = examples["term"]
    descriptions = examples["description"]
    formulas = examples["formula"]
    codes = examples["code"]
    messages = []
    for term, description, formula, code in zip(terms, descriptions, formulas, codes):
        definitions = random.sample(definition, 5)
        definitions1 = random.sample(definition1, 5)
        definitions2 = random.sample(definition2, 5)
        code_requests = random.sample(code_request, 5)
        code_responses = random.sample(code_response, 5)
        for d, d1, d2, c1, c2 in zip(definitions, definitions1, definitions2, code_requests, code_responses):
            temp = []
            if term == 'R Code Example':
                temp.append({"role": "user", "content": c1.format(term, description)})
                temp.append({"role": "assistant", "content": c2.format(code)})
            elif formula == "":
                temp.append({"role": "user", "content": d.format(term)})
                temp.append({"role": "assistant", "content": d2.format(description)})
            else:
                temp.append({"role": "user", "content": d.format(term)})
                temp.append({"role": "assistant", "content": d1.format(description, formula)})

            messages.append(temp)
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text,}

dataset = load_dataset("json", data_files = "SFT_data/noun_data.json", split = "train")
dataset = dataset.shuffle()
dataset = dataset.map(apply_template, batched=True, remove_columns=["term", "description", "formula", "code"])


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
        num_train_epochs=3,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=200,
        optim="adamw_8bit",
        weight_decay=0.05,
        warmup_ratio=0.03,
        output_dir="output",
        seed=3407,
        run_name="SFT_statistics_chat2",
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

model_location = "SFT4-2"
save_path = "SFT4-2-adapter"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
tokenizer.push_to_hub(model_location, token='')
model.push_to_hub_merged(model_location, tokenizer, save_method='merged_16bit', token='')


print("Training completed and model saved successfully!")
