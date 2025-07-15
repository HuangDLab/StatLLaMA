"""
python GRPO-v2.py
"""


import os
import re
import random
import copy
import math
import wandb
import unsloth
import torch
import torch.nn as nn
from fuzzywuzzy import fuzz 
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import Union, List, Optional, Dict, Any
# from rapidfuzz import fuzz
from datasets import load_dataset, Dataset, concatenate_datasets
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

def set_random_seed(seed: int = 3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_random_seed(3407)

### unsloth
max_seq_length = 1024
lora_rank = 8

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "SFT5",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.8,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, 
    lora_dropout=0.05,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], 
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

### Data prepare & Reward function

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_stat_questions():
    data = load_dataset("json", data_files = "stat_GRPO_data.json", split = "train")
    data = data.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["question"]}
        ],
        "answer": x["answer"]
    })
    return data

# def get_gsm8k_questions(split = "train"):
#     data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
#     data = data.map(lambda x: { 
#         'prompt': [
#             {'role': 'system', 'content': SYSTEM_PROMPT},
#             {'role': 'user', 'content': x['question']}
#         ],
#         'answer': extract_hash_answer(x['answer'])
#     }) 
#     return data

dataset = get_stat_questions()
dataset = dataset.shuffle()


def parse_number(text: str) -> Union[float, None]:
    if not isinstance(text, str): return None
    text = text.strip().replace(" ", "").replace(",", "")
    if not text: return None
    try:
        return float(text)
    except ValueError: pass
    if "/" in text:
        parts = text.split("/")
        if len(parts) == 2:
            try:
                numerator = float(parts[0])
                denominator = float(parts[1])
                if denominator == 0: return None
                return numerator / denominator
            except ValueError: pass
    return None

def is_number_match(ground_truth: str, prediction: str, tolerance: float = 1e-2, abs_tolerance: float = 1e-4) -> bool:
    ground_num = parse_number(ground_truth)
    pred_num = parse_number(prediction)
    if ground_num is None or pred_num is None: return False
    return math.isclose(ground_num, pred_num, rel_tol=tolerance, abs_tol=abs_tolerance)

def normalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower()
    s = re.sub(r"[^\w\s/:]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def is_text_match(ground_truth: str, prediction: str, similarity_threshold: float = 90) -> bool: # 稍微提高閾值，因為 partial_ratio 更容易高分
    """Checks text similarity using normalization and partial ratio matching."""
    if not isinstance(ground_truth, str) or not isinstance(prediction, str): return False

    ground_truth_norm = normalize_text(ground_truth)
    prediction_norm = normalize_text(prediction)

    if not ground_truth_norm or not prediction_norm: return False


    similarity_score = fuzz.partial_ratio(ground_truth_norm, prediction_norm)


    return similarity_score >= similarity_threshold

def advanced_answer_matching(ground_truth: str, prediction: str) -> bool:
    """More robust matching: prioritize numbers if possible, always fallback to text."""
    ground_truth_str = str(ground_truth).strip()
    prediction_str = str(prediction).strip()
    if not prediction_str:
        return False

    # Attempt 1: Number Matching
    # Try parsing both first
    ground_num = parse_number(ground_truth_str)
    pred_num = parse_number(prediction_str)

    # If both can be parsed as numbers, perform numeric comparison
    if ground_num is not None and pred_num is not None:
        if is_number_match(ground_truth_str, prediction_str): # Pass original strings as is_number_match reparses
            return True


    # Attempt 2: Text Matching (Fallback for ALL cases where number match didn't return True)
    # This covers: Text vs Text, Num vs Text, Text vs Num, and Num vs Num (where is_close failed)
    if is_text_match(ground_truth_str, prediction_str):
        return True

    # If both number and text matching fail
    return False

def extract_answer_adaptively(text: str, ground_truth_is_numeric: bool) -> str:
    """
    Extracts answer. For numeric GT, prioritizes the globally last number.
    For text GT, uses XML/Label structures and last line heuristics.
    """
    if not isinstance(text, str): return ""

    number_pattern = r"[-+]?\d*\.\d+|[-+]?\d+/\d+|[-+]?\d+"

    # --- Strategy for NUMERIC Ground Truth ---
    if ground_truth_is_numeric:
        # **Prioritize finding the LAST number in the ENTIRE cleaned text**
        text_cleaned = text.replace(",", "").replace(" ", "")
        all_numbers = re.findall(number_pattern, text_cleaned)
        if all_numbers:
            # print(f"Debug Extract (Numeric GT): Found global numbers {all_numbers}, returning last.")
            return all_numbers[-1].strip()
        else:
             # **Numeric GT Fallback (If NO numbers found globally):**
             # Try to extract text from standard blocks, maybe it's "None" or similar.
             extracted_content = ""
             match_xml = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
             if match_xml:
                 extracted_content = match_xml.group(1).strip()
             if not extracted_content:
                 match_tag_colon = re.search(r"<(answer|final\s*answer)>\s*:\s*(.*)", text, re.IGNORECASE)
                 if match_tag_colon:
                     potential_answer = match_tag_colon.group(2).strip()
                     extracted_content = potential_answer.splitlines()[0].strip() if potential_answer else ""
             if extracted_content:
                 return extracted_content # Return extracted text block content
             return "" # Ultimate fallback if no numbers and no text block

    # --- Strategy for TEXT Ground Truth ---
    else: # ground_truth_is_numeric is False
        extracted_content = ""
        # Strategy 1: Standard <answer>...</answer> block
        match_xml = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
        if match_xml:
            extracted_content = match_xml.group(1).strip()
            return extracted_content # Return full content for text GT

        # Strategy 1.5: Handle <tag>: content format
        match_tag_colon = re.search(r"<(answer|final\s*answer)>\s*:\s*(.*)", text, re.IGNORECASE)
        if match_tag_colon:
            potential_answer = match_tag_colon.group(2).strip()
            # For text, take the first line after the colon
            extracted_content = potential_answer.splitlines()[0].strip() if potential_answer else ""
            return extracted_content

        # Strategy 2: Label: content
        match_label = re.search(r"(?:Answer:|Final Answer:|The answer is)\s*([\s\S]*)", text, re.IGNORECASE)
        if match_label:
            potential_answer_block = match_label.group(1).strip()
            # Use the LAST non-empty line as the heuristic for text answers after a label
            lines = [line.strip() for line in potential_answer_block.splitlines() if line.strip()]
            if lines:
                extracted_content = lines[-1]
                return extracted_content

        # Final Text Fallback: If no structure found, maybe return last line of entire response? Risky.
        return "" # Return empty if no structure found for text

# Reward functions
def correctness_reward_func(prompts: List[List[Dict]],
                            completions: List[List[Dict]],
                            answer: List[str],
                            penalty_for_wrong_answer: float = -1.0, 
                            **kwargs) -> List[float]:
    num_generations = len(completions)
    if not answer: return [0.0] * num_generations
    ground_truth = str(answer[0]).strip() if answer else ""
    if not ground_truth: return [0.0] * num_generations

    ground_truth_num = parse_number(ground_truth)
    ground_truth_is_numeric = ground_truth_num is not None

    q = prompts[0][-1]['content'] if prompts and prompts[0] else "Unknown Question"

    rewards = []
    extracted_responses = []
    responses_content = []

    for i, completion in enumerate(completions):
        response_content = ""
        try:
            if isinstance(completion, list) and completion and isinstance(completion[0], dict):
                response_content = completion[0].get("content", "")
            elif isinstance(completion, dict):
                 response_content = completion.get("content", "")
            elif isinstance(completion, str):
                response_content = completion
        except Exception as e:
             print(f"Error processing completion {i} for correctness: {e}")

        responses_content.append(response_content)

        # *** Call the UPDATED adaptive extraction function ***
        extracted_pred = extract_answer_adaptively(response_content, ground_truth_is_numeric)
        extracted_responses.append(extracted_pred)

        current_reward = 0.0
        if extracted_pred:
            is_match = advanced_answer_matching(ground_truth, extracted_pred)
            if is_match:
                current_reward = 2.0
            else:
                current_reward = penalty_for_wrong_answer
        rewards.append(current_reward)

    if responses_content:
      print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{ground_truth} (Numeric: {ground_truth_is_numeric})", f"\nResponse:\n{responses_content[0]}", f"\nExtracted:\n{extracted_responses[0]}", f" --> Reward: {rewards[0]}")

    return rewards


def soft_format_reward_func(completions: List[List[Dict]],
                            reward_for_match: float = 0.2,
                            penalty_for_mismatch: float = -0.2, 
                            **kwargs) -> List[float]:
    """Rewards the presence of <reasoning>...</reasoning> followed by <answer>...</answer> blocks, allows flexibility."""
    # Pattern looks for the tags appearing anywhere in order, with content in between
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    rewards = []
    for completion in completions:
        response_content = ""
        try:
            if isinstance(completion, list) and completion and isinstance(completion[0], dict):
                response_content = completion[0].get("content", "")
            elif isinstance(completion, dict): response_content = completion.get("content", "")
            elif isinstance(completion, str): response_content = completion
        except Exception as e: print(f"Error processing completion for soft format: {e}")

        if re.search(pattern, response_content, re.DOTALL | re.IGNORECASE):
            rewards.append(reward_for_match)
        else:
            rewards.append(penalty_for_mismatch) # Apply penalty
    return rewards

REASONING_PATTERN = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)      
TRAILING_TEXT_THRESHOLD = 5

def calculate_robust_format_score(text: str) -> float:
    """Calculates a format score with bonuses and penalties. Range approx [-0.5, 1.25]."""
    if not isinstance(text, str): return 0.0

    score = 0.0
    text = text.strip() # Check stripped text

    reasoning_match = REASONING_PATTERN.search(text)
    answer_match = ANSWER_PATTERN.search(text)

    has_reasoning_tag = bool(reasoning_match)
    has_answer_tag = bool(answer_match)
    has_non_empty_reasoning = bool(reasoning_match and reasoning_match.group(1).strip())
    has_non_empty_answer = bool(answer_match and answer_match.group(1).strip())

    # Basic presence bonus (reduced slightly to make room for non-empty bonus)
    if has_reasoning_tag: score += 0.3
    if has_answer_tag: score += 0.3

    # Bonus for non-empty content within tags
    if has_non_empty_reasoning: score += 0.2
    if has_non_empty_answer: score += 0.2 # Max potential score for tags + content = 1.0

    # Order bonus/penalty
    if has_reasoning_tag and has_answer_tag:
        if reasoning_match.end() <= answer_match.start():
            score += 0.25 # Correct order bonus (Max potential = 1.25)
        else:
            score -= 0.25 # Incorrect order penalty

    # Penalty for significant text after </answer>
    if answer_match:
        end_pos = answer_match.end()
        trailing_text = text[end_pos:]
        stripped_trailing = trailing_text.strip()
        if len(stripped_trailing) > TRAILING_TEXT_THRESHOLD:
            score -= 0.5 # Significant trailing text penalty
        elif len(trailing_text) > 0 and not stripped_trailing:
             # Penalize if there's only whitespace after </answer> (less severe)
             score -= 0.1

    # Penalty if tags exist but content is empty/whitespace
    if has_reasoning_tag and not has_non_empty_reasoning:
        score -= 0.15
    if has_answer_tag and not has_non_empty_answer:
        score -= 0.15

    # Ensure score doesn't go below a certain floor
    return max(-0.75, score)


def robust_format_reward_func(completions: List[List[Dict]], **kwargs) -> List[float]:
    """Applies the detailed robust format scoring."""
    rewards = []
    for completion_list in completions:
        response_content = ""
        try:
            if isinstance(completion_list, list) and completion_list and isinstance(completion_list[0], dict):
                response_content = completion_list[0].get("content", "")
            elif isinstance(completion_list, dict): response_content = completion_list.get("content", "")
            elif isinstance(completion_list, str): response_content = completion_list
        except Exception as e: print(f"Error extracting content for robust format reward: {e}")

        rewards.append(calculate_robust_format_score(response_content))
    return rewards


### Train the model

max_prompt_length = 512 # Increased slightly, adjust based on typical question length
# Ensure max_completion_length is appropriately set based on max_seq_length and max_prompt_length
max_completion_length = max_seq_length - max_prompt_length
if max_completion_length <= 0:
    raise ValueError("max_seq_length must be greater than max_prompt_length")


training_args = GRPOConfig(
    # --- Core Training Params ---
    output_dir = "outputs",
    num_train_epochs = 1, 
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 3, 
    gradient_checkpointing = True, 

    # --- Optimizer Params ---
    optim = "paged_adamw_8bit", 
    learning_rate = 5e-5, 
    lr_scheduler_type = "cosine",
    warmup_ratio = 0.3,
    weight_decay = 0.01, 
    max_grad_norm = 1.0,

    # --- GRPO Specific Params ---
    num_generations = 6, 
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length, 

    # --- Logging & Saving ---
    logging_steps = 10, 
    report_to = "none", 

    # --- Other ---
    fp16 = True, 
    remove_unused_columns = False, 
    seed = 3407,
)

# Ensure the tokenizer has a padding token, add if missing (Llama usually needs it)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Warning: Added EOS token as PAD token.")

trainer = GRPOTrainer(
    model = model,
    tokenizer = tokenizer,
    reward_funcs = [
        correctness_reward_func,
        soft_format_reward_func,
        robust_format_reward_func
    ],
    args = training_args,
    train_dataset = dataset,
)

print("Starting GRPO training...")
trainer.train()
print("Training finished.")

model_location = "GRPO-v2"

tokenizer.push_to_hub(model_location, token = '')
model.push_to_hub_merged(model_location, tokenizer, save_method = "merged_16bit", token = '')
