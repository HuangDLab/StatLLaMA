# python Data2Token.py

import torch
import transformers
from typing import Sequence, Dict, List
import re
import os
import json
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlamaForCausalLM,
    LlamaConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from tqdm import tqdm
from huggingface_hub import login
import multiprocessing

# --- Configuration ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_token = ""

try:
    login(token=hf_token) 
except Exception as e:
    print(f"Could not log in to Hugging Face Hub: {e}. Please check your token. Script will continue, but may not be able to download private models.")

# Model and path settings
model_name_or_path = 'unsloth/Llama-3.2-3B-Instruct' 
input_json_path = 'pretrain_data.json' 
output_json_path = 'pretrain_data_token.json' 
tokenizer_max_length = 8192 # Max length for the Tokenizer (recommended to be the model's max supported length)
chunk_max_length = 2048     # Max length for the token chunks after splitting (for training)

# --- Tokenizer Loading ---
print(f"Loading tokenizer from {model_name_or_path}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        model_max_length=tokenizer_max_length, 
        padding_side="left", 
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad_token, will use eos_token as pad_token.")
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded successfully. Pad token ID: {tokenizer.pad_token_id}")

except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

# --- Helper Function (for very long texts, if needed) ---
def split_text_into_chunks(text, max_text_chars=130000):
    """Splits very long text by character count before tokenization."""
    # This function might not be necessary unless individual text items are extremely large
    print(f"Warning: Text length exceeds {max_text_chars}, splitting by characters...")
    return [text[i:i+max_text_chars] for i in range(0, len(text), max_text_chars)]

# --- Dataset Class (Includes Label Padding Fix) ---
class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: transformers.PreTrainedTokenizer, max_chunk_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_chunk_length = max_chunk_length 
        self.input_ids: List[List[int]] = []
        self.labels: List[List[int]] = []

        if not isinstance(texts, list):
             raise TypeError(f"Input 'texts' must be a list, but received {type(texts)}")

        print(f"Starting to process {len(texts)} texts...")
        for text in tqdm(texts, desc="Tokenizing Texts"):
            if not isinstance(text, str):
                 print(f"Warning: Input list contains non-string element (type: {type(text)}), skipping.")
                 continue
            if not text.strip(): # Skip empty strings
                continue

            try:
                result = self._tokenize_and_chunk_text(text)
                self.input_ids.extend(result['input_ids'])
                self.labels.extend(result['labels'])
            except Exception as e:
                print(f"Error processing text: {e}. Skipping this text. Text start: {text[:100]}...")


        print(f"Processing complete. Generated {len(self.input_ids)} tokenized data chunks.")
        if not self.input_ids:
            print("Warning: No valid data chunks were generated after processing! Please check input data and processing logic.")


    def _tokenize_and_chunk_text(self, text: str) -> Dict[str, List[List[int]]]:
        """Tokenizes a single text and splits it into fixed-length chunks."""
        input_ids_chunks: List[List[int]] = []
        labels_chunks: List[List[int]] = []

        # --- Tokenization ---
        tokens_dict = self.tokenizer(text, padding=False, truncation=True, return_tensors=None, add_special_tokens=False) # Do not automatically add bos/eos
        all_tokens = tokens_dict['input_ids']

        if not all_tokens: # If tokenization results in empty list
            return {"input_ids": [], "labels": []}

        # --- Chunking ---
        stride = max(1, self.max_chunk_length // 2) 
        for i in range(0, len(all_tokens), stride):
            chunk = [self.tokenizer.bos_token_id] + all_tokens[i : i + self.max_chunk_length - 2] # -2 for bos and eos
            chunk = chunk + [self.tokenizer.eos_token_id]

            current_length = len(chunk)

            # --- Padding ---
            if current_length < self.max_chunk_length:
                pad_length = self.max_chunk_length - current_length
                ids_padded = [self.tokenizer.pad_token_id] * pad_length + chunk
                labels_padded = [-100] * pad_length + chunk
            elif current_length == self.max_chunk_length:
                ids_padded = chunk
                labels_padded = chunk 
            else:
                print(f"Warning: Chunk length ({current_length}) exceeds expected ({self.max_chunk_length}), skipping this chunk.")
                continue

            input_ids_chunks.append(ids_padded)
            labels_chunks.append(labels_padded)

        return {"input_ids": input_ids_chunks, "labels": labels_chunks}

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx) -> Dict[str, List[int]]:
        """Returns the tokenized chunk (input_ids and labels) for the given index."""
        if idx >= len(self.input_ids):
            raise IndexError("Index out of range")
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx]
        }

# --- Main Processing Logic ---
if __name__ == "__main__":
    print(f"Reading raw data from {input_json_path}...")
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            # Assume the JSON file contains a list of strings
            raw_text_data = json.load(f)
            if not isinstance(raw_text_data, list):
                raise TypeError(f"Input JSON file should contain a list, but found {type(raw_text_data)}")
            print(f"Successfully read {len(raw_text_data)} raw text entries.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_json_path}")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not parse input file {input_json_path}, ensure it is valid JSON format.")
        exit()
    except Exception as e:
        print(f"Unknown error while reading input file: {e}")
        exit()

    print(f"Creating TextDataset and processing data (target chunk length: {chunk_max_length})...")
    dataset = TextDataset(raw_text_data, tokenizer, max_chunk_length=chunk_max_length)

    output_data = {"input_ids": [], "labels": []}

    if len(dataset) > 0:
        print(f"Organizing {len(dataset)} tokenized data chunks into output dictionary...")
        for i in tqdm(range(len(dataset)), desc="Collecting Tokenized Data"):
            try:
                sample = dataset[i]
                # Re-verify sample structure is correct
                if "input_ids" in sample and "labels" in sample and \
                   isinstance(sample["input_ids"], list) and isinstance(sample["labels"], list) and \
                   len(sample["input_ids"]) == chunk_max_length and len(sample["labels"]) == chunk_max_length:

                    output_data["input_ids"].append(sample["input_ids"])
                    output_data["labels"].append(sample["labels"])
                else:
                     print(f"Warning: Sample at index {i} has incorrect format or length, skipping. Sample keys: {sample.keys()}, input_ids length: {len(sample.get('input_ids', []))}, labels length: {len(sample.get('labels', []))}")

            except Exception as e:
                 print(f"Error collecting data at index {i}: {e}")

        print(f"Saving processed data to {output_json_path}...")
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f)
            print("Data saved successfully!")
        except IOError as e:
            print(f"Error saving file {output_json_path}: {e}")
        except Exception as e:
            print(f"Unknown error while saving file: {e}")
    else:
        print("No valid data chunks generated, not saving file.")