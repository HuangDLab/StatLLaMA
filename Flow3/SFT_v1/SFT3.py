'''
python SFT3.py
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

wandb.init(project="statistical-llm-sft", name="SFT3")

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="SFT2",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=12,  
    lora_alpha=16,
    lora_dropout=0.05, 
    target_modules=["k_proj", "up_proj", "down_proj"], 
    use_rslora=True,  
    use_gradient_checkpointing="unsloth",
)

prompt_templates = [
"""Below are the instructions for explaining and summarizing the content of a paper. This task requires you to extract the main research question, objective, and motivation from the abstract of the paper, and demonstrate your understanding of the study.

### Instruction:
Imagine you are a seasoned academic reviewing a new submission. Based solely on the provided abstract, articulate a concise yet comprehensive overview of this research. Your summary should clearly identify the primary research question or problem being investigated, the main objective of the study, and the key motivation or significance of this work within its field.

### Abstract:
{abstract}

### Response:
{method}
""",
"""Below are the guidelines for summarizing a paper. In this task, you will need to identify the main research question, purpose, and motivation based on the abstract, and demonstrate your understanding of the study.

### Instruction:
Assume you are preparing a brief presentation about this paper for colleagues in a different sub-discipline. Using only the abstract, pinpoint and list the most significant findings or outcomes of this research. Ensure you highlight the core results that would be of interest to a general audience in your field.

### Abstract:
{abstract}

### Response
{result}
""",#3
"""Below are the instructions for interpreting and summarizing a research paper. Your task is to extract the central research problem, goals, and motivations from the abstract and showcase your comprehension of the study.

### Instruction:
Envision yourself as a reader trying to grasp the final takeaways of this research after only reading the abstract. What are the definitive conclusions or key insights that the authors have reached based on their findings? Explain the overall interpretation and implications of these results as stated in the abstract.

### Abstract:
{abstract}

### Response
{conclusion}
""",
"""Below are the instructions for summarizing the content of a research paper. The task involves extracting the key research question, objectives, and motivations from the abstract, while showing your understanding of the paper.

### Instruction:
You are tasked with explaining the research approach to someone unfamiliar with the specific methods in this field. Based on the abstract, describe in detail the primary methodology or approach employed in this study. Furthermore, elaborate on how this particular method is likely suited to address the research question outlined in the abstract, highlighting its key advantages or relevance.

### Abstract:
{abstract}

### Response
{method}
""",#5
"""Below are the directions for reviewing and summarizing a research paper. You are required to pull out the main research question, aim, and motivation from the abstract, and present a clear understanding of the study.

### Instruction:
Consider yourself an analyst examining the abstract. How do the principal findings or results that are reported in this research directly and effectively answer the initial research question or problem that the study set out to explore? Explain the logical connection between the findings and the research objective.

### Abstract:
{abstract}

### Response
{result}
""",
"""Below are the instructions for reviewing and summarizing the content of a paper. You need to extract the primary research question, objectives, and motivation from the abstract and present a clear understanding of the study.

### Instruction:
Imagine you are writing a commentary on this research based solely on the abstract. What are the potential broader implications, real-world applications, or theoretical significance of the research findings? Discuss how these results might contribute to advancements or new understandings within the larger academic community or relevant industries.

### Abstract:
{abstract}

### Response
{conclusion}
""",#7
"""Below are the guidelines for summarizing a research paper. In this task, you will focus on identifying the main research problem, goal, and motivation based on the abstract, showcasing your understanding of the work.

### Instruction:
As a critical reader of the abstract, identify the methodology employed in this research, and highlight any potential limitations, constraints, or areas of uncertainty that might arise from the chosen approach. Consider aspects such as the scope of the study, sample size (if mentioned or inferable), any biases in data collection, or how well the method is suited to address the research problem.

### Abstract:
{abstract}

### Response
{method}
""",
"""Below are the instructions for synthesizing a paper’s content. The task involves extracting the central research question, objectives, and motivations from the abstract and demonstrating your comprehension of the study.

### Instruction:
Assume the role of a librarian cataloging this research paper based on its abstract. Provide a concise yet informative synthesis of the study. Your summary must include the core research goal, the primary methodology utilized, and the most significant findings that were achieved, all extracted from the abstract.

### Abstract:
{abstract}

### Response
Method: {method}
Result: {result}
""",#9
"""Below are the instructions for reviewing a research paper. Your job is to identify the key research question, aim, and significance from the abstract and convey a clear understanding of the research.

### Instruction:
Drawing upon your existing knowledge of related research in this domain, and focusing on the results mentioned in the abstract, analyze how the outcomes of this specific study might either align with, contradict, or offer a novel perspective compared to the findings of previously published work on similar topics or research questions.

### Abstract:
{abstract}

### Response
{result}
""",
"""Below are the steps for summarizing the content of a research paper. You should extract the main research issue, goal, and motivation from the abstract and explain your understanding of the study.

### Instruction:
Acting as a forward-thinking researcher, and considering the research presented in the abstract along with the conclusions reached, propose several logical and promising directions for future research or new research questions that could build upon or extend the findings of this current study. Justify your suggestions based on the information provided.

### Abstract:
{abstract}

### Response
{conclusion}
""",#11
"""Below are the instructions for analyzing the content of a research paper. The task is to identify the primary research question, objectives, and motivation from the abstract and show a deep understanding of the research.

### Instruction:
The abstract provides a glimpse into the research method. Elaborate on what you can infer about the specific steps or procedures likely involved in this study based on the abstract's description of the methodology.

### Abstract:
{abstract}

### Response
{method}
"""
]

EOS_TOKEN = tokenizer.eos_token 


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


def formatting_prompts_func(examples):
    abstracts = examples["Abstract"]
    methods = examples["Method"]
    results = examples["Result"]
    conclusions = examples["Conclusion"]
    texts = []
    
    for abstract, method, result, conclusion in zip(abstracts, methods, results, conclusions):
        templates = random.sample(prompt_templates, 5)
        # template = random.choice(prompt_templates)
        for template in templates:
            text = template.format(abstract=abstract, method=method, result=result, conclusion=conclusion) + EOS_TOKEN
            texts.append(text)
    return {"text" : texts,}

dataset = load_dataset("json", data_files = "stat_S2ORC_data.json", split = "train")
dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=["Abstract", "Method", "Result", "Conclusion"])
dataset = dataset.shuffle(seed=42)




trainer = SFTTrainer(
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
        run_name="SFT3",
        report_to="wandb",
        greater_is_better=False,
    ),
)

trainer.train()

model_location = "SFT3"
model.save_pretrained(model_location)
tokenizer.save_pretrained(model_location)
tokenizer.push_to_hub(model_location, token='')
model.push_to_hub_merged(model_location, tokenizer, save_method='merged_16bit', token='')


print("Training completed and model saved successfully!")
