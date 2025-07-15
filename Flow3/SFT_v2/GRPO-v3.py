"""
python GRPO-v3.py
"""

import os
import re
import random
import copy
import math
# import wandb # Uncomment if using wandb
import unsloth
import torch
import torch.nn as nn
from fuzzywuzzy import fuzz # Using fuzzywuzzy as in the original code
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import Union, List, Optional, Dict, Any
from datasets import load_dataset, Dataset, concatenate_datasets
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from transformers import TextStreamer # Potentially used elsewhere, keeping import

# --- Configuration ---
SEED = 3407
MODEL_NAME = "SFT-v2" 
OUTPUT_DIR = "outputs"
HUB_MODEL_LOCATION = "GRPO-v3" # Change if needed
HF_TOKEN = '' # Your Hugging Face token

# Model & Training Params
MAX_SEQ_LENGTH = 1024
LORA_RANK = 16
EPOCHS = 1
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 3
LEARNING_RATE = 5e-5 
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
OPTIMIZER = "paged_adamw_8bit"
LR_SCHEDULER = "cosine"
MAX_GRAD_NORM = 1.0
NUM_GENERATIONS = 6 

# Reward Function Weights (Implicit via magnitude)
REWARD_CORRECT = 2.0
PENALTY_WRONG = -1.5
PENALTY_MISSING_ANSWER = -1.0
# Robust Format Function rewards/penalties are defined within its calculation

# --- Seed Setting ---
def set_random_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_random_seed()
print(f"Random seed set to {SEED}")

# --- Model Loading (Unsloth) ---
print(f"Loading model: {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True,
    max_lora_rank = LORA_RANK, 
)

# Add LoRA adapters
print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK,
    lora_dropout=0.05,
    target_modules = [ 
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = LORA_RANK, 
    use_gradient_checkpointing = "unsloth", 
    random_state = SEED,

)
print("LoRA adapters added.")

# --- Data Preparation ---
SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
Detailed step-by-step thinking process to solve the problem.
</reasoning>
<answer>
Final answer derived from the reasoning.
</answer>"""


def extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_stat_questions(data_file="GRPO_data.json"):
    """Loads statistical questions and formats them for GRPO."""
    print(f"Loading dataset from {data_file}...")
    try:
        data = load_dataset("json", data_files=data_file, split="train")
        data = data.map(lambda x: {
            "prompt": [ # Format as list of dictionaries for chat template
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]}
            ],
            "answer": str(x["answer"]) # Ensure answer is string for consistency
        }, num_proc=os.cpu_count()) # Use multiprocessing for mapping
        print(f"Loaded and formatted {len(data)} examples.")
        return data
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        raise


def get_gsm8k_questions(split = "train"):
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { 
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': str(extract_hash_answer(x['answer']))
    }, num_proc=os.cpu_count()) 
    return data

dataset1 = get_stat_questions()
dataset2 = get_gsm8k_questions()
dataset2 = dataset2.select(range(1000))
dataset = concatenate_datasets([dataset1, dataset2])
dataset = dataset.shuffle(seed=SEED)
print("Dataset shuffled.")

# --- Utility Functions for Rewards ---

def parse_number(text: str) -> Union[float, None]:
    """Parses potential numbers, including fractions."""
    if not isinstance(text, str): return None
    # Handle potential percentages
    if text.endswith('%'):
        try:
            return float(text[:-1]) / 100.0
        except ValueError:
            pass # Fall through if conversion fails
    # Handle commas, spaces
    text = text.strip().replace(" ", "").replace(",", "")
    if not text: return None
    try:
        return float(text)
    except ValueError: pass
    # Handle fractions
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
    """Checks if two strings represent close numbers."""
    ground_num = parse_number(ground_truth)
    pred_num = parse_number(prediction)
    if ground_num is None or pred_num is None: return False
    return math.isclose(ground_num, pred_num, rel_tol=tolerance, abs_tol=abs_tolerance)

def normalize_text(s: str) -> str:
    """Lowercase, remove punctuation (except /:), and normalize whitespace."""
    if not isinstance(s, str): return ""
    s = s.lower()
    # Keep / and : for fractions and potential time/ratio formats, remove others
    s = re.sub(r"[^a-z0-9\s/:\.\-]", "", s) # Keep letters, numbers, whitespace, /, :, ., -
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def is_text_match(ground_truth: str, prediction: str, similarity_threshold: float = 85) -> bool: # Adjusted threshold
    """Checks text similarity using normalization and fuzzy partial ratio."""
    if not isinstance(ground_truth, str) or not isinstance(prediction, str): return False

    ground_truth_norm = normalize_text(ground_truth)
    prediction_norm = normalize_text(prediction)

    if not ground_truth_norm or not prediction_norm: return False

    # Use partial_ratio for flexibility in finding answer within generated text
    similarity_score = fuzz.partial_ratio(ground_truth_norm, prediction_norm)

    return similarity_score >= similarity_threshold

def advanced_answer_matching(ground_truth: str, prediction: str) -> bool:
    """Prioritize number match, fallback to text match."""
    ground_truth_str = str(ground_truth).strip()
    prediction_str = str(prediction).strip()
    if not prediction_str: return False

    # Attempt 1: Number Matching
    ground_num = parse_number(ground_truth_str)
    pred_num = parse_number(prediction_str)
    if ground_num is not None and pred_num is not None:
        if is_number_match(ground_truth_str, prediction_str):
            return True
        # If number parsing works but match fails, still proceed to text matching

    # Attempt 2: Text Matching (Fallback)
    if is_text_match(ground_truth_str, prediction_str):
        return True

    return False

def extract_answer_adaptively(text: str, ground_truth_is_numeric: bool) -> str:
    """Extracts answer based on ground truth type and common formats."""
    if not isinstance(text, str): return ""
    text = text.strip()
    extracted_content = ""

    # Priority 1: <answer>...</answer> block
    match_xml = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if match_xml:
        extracted_content = match_xml.group(1).strip()
        # If GT is numeric, try to extract the last number *within* this block
        if ground_truth_is_numeric:
            numbers_in_block = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+/\d+|[-+]?\d+", extracted_content.replace(",", ""))
            if numbers_in_block:
                return numbers_in_block[-1].strip()
            else:
                 return extracted_content # Return text if no number found inside block
        else:
             return extracted_content # Return full content for text GT

    # Priority 2: <answer>: content format
    match_tag_colon = re.search(r"<(answer|final\s*answer)>\s*:\s*(.*)", text, re.IGNORECASE)
    if match_tag_colon:
        potential_answer = match_tag_colon.group(2).strip()
        # Take the first line after the colon
        first_line = potential_answer.splitlines()[0].strip() if potential_answer else ""
        if ground_truth_is_numeric:
             numbers_in_line = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+/\d+|[-+]?\d+", first_line.replace(",", ""))
             if numbers_in_line:
                 return numbers_in_line[-1].strip()
             else:
                  return first_line # Return text if no number found
        else:
            return first_line

    # Priority 3: Label: content (e.g., "Final Answer: ...")
    match_label = re.search(r"(?:Answer:|Final Answer:|The answer is)\s*([\s\S]*)", text, re.IGNORECASE)
    if match_label:
        potential_answer_block = match_label.group(1).strip()
        lines = [line.strip() for line in potential_answer_block.splitlines() if line.strip()]
        if lines:
            last_line = lines[-1]
            if ground_truth_is_numeric:
                numbers_in_line = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+/\d+|[-+]?\d+", last_line.replace(",", ""))
                if numbers_in_line:
                    return numbers_in_line[-1].strip()
                else:
                     return last_line # Return text if no number
            else:
                return last_line # Return last non-empty line for text

    # Fallback (if numeric GT and no structures matched): Find last number globally
    if ground_truth_is_numeric:
        all_numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+/\d+|[-+]?\d+", text.replace(",", ""))
        if all_numbers:
            return all_numbers[-1].strip()

    # Ultimate fallback: return empty string if nothing matched
    return ""


# --- Reward Function Definitions ---

# 1. Correctness Reward
def correctness_reward_func(prompts: List[List[Dict]],
                            completions: List[List[Dict]],
                            answer: List[str],
                            reward_for_correct: float = REWARD_CORRECT,
                            penalty_for_wrong: float = PENALTY_WRONG,
                            penalty_for_missing: float = PENALTY_MISSING_ANSWER,
                            penalty_for_wrong_answer: float = PENALTY_WRONG, 
                            **kwargs) -> List[float]:
    """
    Rewards correct answers based on robust matching, penalizes wrong/missing.
    Prints summary info for the *first* completion at the end of the batch processing.
    """
    # Check if answer list is provided and non-empty
    if not answer or not answer[0]:
        # print("Warning: Missing ground truth answer for this batch.")
        return [0.0] * len(completions) # Return neutral reward if GT is missing
    ground_truth = str(answer[0]).strip()

    ground_truth_num = parse_number(ground_truth)
    ground_truth_is_numeric = ground_truth_num is not None

    # --- Extract question once for the batch ---
    try:
        q = prompts[0][-1]['content'] if prompts and prompts[0] and prompts[0][-1]['role'] == 'user' else "Unknown Question"
    except (IndexError, TypeError, KeyError):
        q = "Error extracting question"

    rewards = []
    extracted_preds_all = [] # Store all extracted preds for potential later use if needed
    response_contents_all = [] # Store all response contents for potential later use if needed

    # --- Loop through completions to calculate rewards ---
    for i, completion_list in enumerate(completions):
        response_content = ""
        extracted_pred = ""
        current_reward = 0.0
        is_match = False # Initialize is_match

        try:
            # Standard content extraction
            if isinstance(completion_list, list) and completion_list and isinstance(completion_list[0], dict):
                response_content = completion_list[0].get("content", "")
            elif isinstance(completion_list, dict): response_content = completion_list.get("content", "")
            elif isinstance(completion_list, str): response_content = completion_list
            response_contents_all.append(response_content) # Store for potential later use
        except Exception as e:
            print(f"Error extracting content for completion {i}: {e}")
            response_contents_all.append("[Error Extracting Content]")
            extracted_preds_all.append("[Extraction Skipped]")
            rewards.append(penalty_for_missing) # Penalize if content extraction fails
            continue # Skip to next completion

        try:
            extracted_pred = extract_answer_adaptively(response_content, ground_truth_is_numeric)
            extracted_preds_all.append(extracted_pred) # Store extracted pred
        except Exception as e:
             print(f"Error extracting answer for completion {i}: {e}")
             extracted_preds_all.append("[Error Extracting Answer]")
             rewards.append(penalty_for_missing) # Penalize if answer extraction fails
             continue # Skip to next completion

        if not extracted_pred or extracted_pred == "[Error Extracting Answer]":
            current_reward = penalty_for_missing
            is_match = False
        else:
            try:
                is_match = advanced_answer_matching(ground_truth, extracted_pred)
                if is_match:
                    current_reward = reward_for_correct
                else:
                    # Use the consistent penalty variable defined at the start
                    current_reward = penalty_for_wrong
            except Exception as e:
                 print(f"Error matching answer for completion {i}: {e}")
                 current_reward = penalty_for_wrong # Assume wrong if matching fails
                 is_match = False

        rewards.append(current_reward)

    # --- Print summary for the FIRST completion AFTER processing all completions ---
    # Check if there are any results to print
    if response_contents_all and extracted_preds_all and rewards:
        print('-'*20,
              f"Question:\n{q[:200]}...", # Truncated question
              f"\nAnswer:\n'{ground_truth}' (Numeric: {ground_truth_is_numeric})",
              f"\nResponse [0]:\n{response_contents_all[0][:250]}...", # Truncated response 0
              f"\nExtracted [0]:\n'{extracted_preds_all[0]}'", # Extracted prediction 0
              f"\n --> Reward [0]: {rewards[0]:.2f}") # Reward 0
    else:
        print('-'*20, "Correctness Check: No completions processed or error occurred.")
    print(f"--- End Correctness Check Batch ---")

    return rewards


# 2. Robust Format Reward
REASONING_PATTERN = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
TRAILING_TEXT_THRESHOLD = 10 # Allows a few trailing spaces/newlines

def calculate_robust_format_score(text: str) -> float:
    """Calculates a format score focusing on structure, content, order, and trailing text."""
    if not isinstance(text, str): return -0.5
    score = 0.0
    text_stripped = text.strip()
    if not text_stripped: return -0.5

    reasoning_match = REASONING_PATTERN.search(text)
    answer_match = ANSWER_PATTERN.search(text)

    has_reasoning_tag = bool(reasoning_match)
    has_answer_tag = bool(answer_match)
    has_non_empty_reasoning = bool(reasoning_match and reasoning_match.group(1) and reasoning_match.group(1).strip())
    has_non_empty_answer = bool(answer_match and answer_match.group(1) and answer_match.group(1).strip())

    # Base score for tags
    if has_reasoning_tag: score += 0.2
    if has_answer_tag: score += 0.2

    # Bonus for non-empty content
    if has_non_empty_reasoning: score += 0.3
    if has_non_empty_answer: score += 0.3 # Max potential = 1.0

    # Order bonus/penalty
    if has_reasoning_tag and has_answer_tag:
        if reasoning_match.end() <= answer_match.start():
            score += 0.25 # Correct order bonus (Max potential = 1.25)
        else:
            score -= 0.25 # Incorrect order penalty

    # Trailing text penalty (only for non-whitespace)
    if answer_match:
        end_pos = answer_match.end()
        trailing_text = text[end_pos:]
        stripped_trailing = trailing_text.strip()
        if len(stripped_trailing) > TRAILING_TEXT_THRESHOLD:
            score -= 0.5
        elif len(trailing_text) > 0 and not stripped_trailing and len(trailing_text) > TRAILING_TEXT_THRESHOLD // 2: # Penalize excessive whitespace
             score -= 0.1

    # Empty content penalty
    if has_reasoning_tag and not has_non_empty_reasoning: score -= 0.2
    if has_answer_tag and not has_non_empty_answer: score -= 0.2

    # Single tag penalty
    if has_reasoning_tag ^ has_answer_tag: score -= 0.3

    return max(-0.75, score) # Floor the score

def robust_format_reward_func(completions: List[List[Dict]], **kwargs) -> List[float]:
    """Applies the refined robust format scoring."""
    rewards = []
    for i, completion_list in enumerate(completions):
        response_content = ""
        try:
            if isinstance(completion_list, list) and completion_list and isinstance(completion_list[0], dict):
                response_content = completion_list[0].get("content", "")
            elif isinstance(completion_list, dict): response_content = completion_list.get("content", "")
            elif isinstance(completion_list, str): response_content = completion_list
        except Exception as e: print(f"Error extracting content for robust format reward (idx {i}): {e}")

        score = calculate_robust_format_score(response_content)
        rewards.append(score)
    return rewards


# 3. Reasoning Quality Reward (Heuristic)
POSITIVE_REASONING_INDICATORS = [
    r"step \d", r"firstly", r"secondly", r"thirdly",
    r"therefore", r"hence", r"thus", r"because", r"since",
    r"let['’]?s define", r"consider", r"apply", r"formula for",
    r"calculate", r"substitute", r"derive", r"we get", r"result is",
    r"based on", r"according to", r"definition of",
    r"probability", r"mean", r"variance", r"standard deviation", r"hypothesis",
    r"p-value", r"confidence interval", r"regression", r"test statistic" # Domain specific
]
NEGATIVE_REASONING_INDICATORS = [
    r"\b(it seems|maybe|perhaps|i guess|i think)\b",
    r"\b(the answer is just|obviously|clearly)\b", # Can indicate lack of rigor
    r"\b(let me check|checking)\b",
]

def calculate_reasoning_heuristics(reasoning_text: str) -> float:
    """Calculates a heuristic score based on length, keywords, and structure."""
    if not reasoning_text: return -0.5
    score = 0.0
    text_lower = reasoning_text.lower()
    words = reasoning_text.split()
    text_len = len(words)

    # Length Bonus (capped)
    length_bonus = min(text_len / 100.0, 0.5)
    score += length_bonus

    # Positive Indicator Bonus (capped)
    positive_count = sum(1 for pattern in POSITIVE_REASONING_INDICATORS if re.search(pattern, text_lower))
    positive_bonus = min(positive_count * 0.05, 0.5)
    score += positive_bonus

    # Negative Indicator Penalty (capped)
    negative_count = sum(1 for pattern in NEGATIVE_REASONING_INDICATORS if re.search(pattern, text_lower))
    score -= min(negative_count * 0.1, 0.3)

    # Structure Bonus (simple check for multiple lines)
    if '\n' in reasoning_text.strip() and text_len > 10: # Avoid rewarding single short lines with newline
        score += 0.1

    # Bonus for using numbers/equations (simple check for digits)
    if any(char.isdigit() for char in reasoning_text):
        score += 0.05

    return max(-0.5, min(score, 1.1)) # Cap score range

def reasoning_quality_reward_func(completions: List[List[Dict]], **kwargs) -> List[float]:
    """Provides a heuristic reward based on the content of the <reasoning> block."""
    rewards = []
    for i, completion_list in enumerate(completions):
        response_content = ""
        try:
            if isinstance(completion_list, list) and completion_list and isinstance(completion_list[0], dict):
                response_content = completion_list[0].get("content", "")
            elif isinstance(completion_list, dict): response_content = completion_list.get("content", "")
            elif isinstance(completion_list, str): response_content = completion_list
        except Exception as e: print(f"Error extracting content for reasoning quality reward (idx {i}): {e}")

        reasoning_match = REASONING_PATTERN.search(response_content)
        reasoning_text = reasoning_match.group(1).strip() if reasoning_match else ""

        score = calculate_reasoning_heuristics(reasoning_text)
        rewards.append(score)

    return rewards


# --- Training Setup ---

max_prompt_length = 512 # Max tokens for System + User Question
max_completion_length = MAX_SEQ_LENGTH - max_prompt_length
if max_completion_length <= 50: # Need sufficient length for reasoning + answer
    raise ValueError(f"max_completion_length ({max_completion_length}) is too short. "
                     f"Increase max_seq_length or decrease max_prompt_length.")
print(f"Max Prompt Length: {max_prompt_length}, Max Completion Length: {max_completion_length}")


training_args = GRPOConfig(
    # --- Core Training Params ---
    output_dir = OUTPUT_DIR,
    num_train_epochs = EPOCHS,
    # max_steps = MAX_STEPS, # Use either epochs or max_steps
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM_STEPS,
    gradient_checkpointing = True, # Let Unsloth handle the backend details

    # --- Optimizer Params ---
    optim = OPTIMIZER,
    learning_rate = LEARNING_RATE,
    lr_scheduler_type = LR_SCHEDULER,
    warmup_ratio = WARMUP_RATIO,
    weight_decay = WEIGHT_DECAY,
    max_grad_norm = MAX_GRAD_NORM,

    # --- GRPO Specific Params ---
    beta = 0.1, # Default DPO/GRPO beta
    num_generations = NUM_GENERATIONS,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    # generation_kwargs = {"temperature": 0.7, "do_sample": True}, # Optional: Control generation during training

    # --- Logging & Saving ---
    logging_steps = 10,
    save_strategy = "epoch",
    report_to = "none", 

    # --- Other ---
    bf16 = False, 
    fp16 = True,
    remove_unused_columns = False, # IMPORTANT for passing 'answer' to reward funcs
    seed = SEED,
)

# Ensure tokenizer has a padding token (essential for batching)
if tokenizer.pad_token is None:
    print("Setting pad_token to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token

# --- Trainer Initialization ---
trainer = GRPOTrainer(
    model = model,
    tokenizer = tokenizer,
    reward_funcs = [ # The list of reward functions to use
        correctness_reward_func,
        robust_format_reward_func,
        reasoning_quality_reward_func
    ],
    args = training_args,
    train_dataset = dataset,
)
print("GRPOTrainer initialized.")

# --- Start Training ---
print("Starting GRPO training...")
try:
    trainer.train()
    print("Training finished successfully.")
except Exception as e:
    print(f"An error occurred during training: {e}")

# --- Save & Push Final Model ---
print("Saving final LoRA adapters...")

print(f"Pushing merged model to Hub: {HUB_MODEL_LOCATION}")
try:
    # Push merged model (16-bit for wider compatibility)
    model.push_to_hub_merged(HUB_MODEL_LOCATION, tokenizer, save_method="merged_16bit", token=HF_TOKEN)
    print("Model pushed successfully.")
except Exception as e:
    print(f"Error pushing model to Hub: {e}")
    print("Consider saving locally using:")
    print(f"  model.save_pretrained_merged('{OUTPUT_DIR}/final_merged_16bit', tokenizer, save_method='merged_16bit')")


print("Script finished.")