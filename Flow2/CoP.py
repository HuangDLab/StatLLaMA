# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --num_gpus=8 CoP.py

import os
import json
import logging
import torch
import copy
from tqdm.notebook import tqdm
from typing import Dict, Optional, Sequence
from generate_pretrain_data import RawPretrainDataset
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType, 
    prepare_model_for_kbit_training
)
from huggingface_hub import login
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
from datasets import Dataset
from dataclasses import dataclass, field
import transformers
import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed import init_distributed

import torch.distributed as dist


os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
login(token="")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='unsloth/Llama-3.2-3B')

@dataclass
class DataArguments:
    data_path: str = field(default='pretrain_data_token.json')

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = './results'
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = 2048
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 4
    save_steps: int = 5000
    logging_dir: Optional[str] = './logs'
    logging_steps: int = 100
    learning_rate: float = 2e-5
    fp16: bool = True
    deepspeed: Optional[str] = "ds_config.json" 
    gradient_accumulation_steps: int = 3
    lr_scheduler_type: Optional[str] = "cosine"
    remove_unused_columns: bool = False
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.5
    weight_decay: float = 0.01
    dataloader_num_workers: int = 12
    gradient_checkpointing: bool = True
    local_rank: int = int(os.getenv("LOCAL_RANK", -1))


def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict, tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class PretrainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        with open(data_path, "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        if isinstance(index, int):  
            return {
                "input_ids": torch.tensor(self.input_ids[index], dtype=torch.long),
                "labels": torch.tensor(self.labels[index], dtype=torch.long)
            }
        elif isinstance(index, list) or isinstance(index, torch.Tensor): 
            indices = index.tolist() if isinstance(index, torch.Tensor) else index
            return {
                "input_ids": torch.stack([torch.tensor(self.input_ids[i], dtype=torch.long) for i in indices]),
                "labels": torch.stack([torch.tensor(self.labels[i], dtype=torch.long) for i in indices])
            }


@dataclass
class DataCollatorForPretrainDataset(object):
    """Collator for pretraining dataset, ensuring labels are padded with -100."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = [instance["input_ids"] for instance in instances]
        labels_list = [instance["labels"] for instance in instances]

        # Use pad_token_id to pad input_ids.
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100  
        )

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

def make_pretrain_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    train_dataset = PretrainDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForPretrainDataset(tokenizer=tokenizer)
    return {"train_dataset": train_dataset, "data_collator": data_collator}


def setup_device(training_args):
    if training_args.local_rank != -1:
        init_distributed()
        device = torch.device(f"cuda:{training_args.local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=[])

    device = setup_device(training_args)

    config = LlamaConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    # If using gradient checkpointing, make sure use_cache is set to False.
    if training_args.gradient_checkpointing:
        config.use_cache = False 

    base_model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=config,
        # attn_implementation="sdpa" 
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=True,
    )
    # Ensure that a pad token is set.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if isinstance(base_model.config, dict):
         base_model.config = LlamaConfig.from_dict(base_model.config)
    if hasattr(base_model.config, "rope_scaling"):
         base_model.config.rope_scaling = {"type": "linear", "factor": 2.0}

    if training_args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        base_model.config.use_cache = False

    lora_config = LoraConfig(
        r=48,
        lora_alpha=72,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        inference_mode=False,
        fan_in_fan_out=False,
        use_dora=True, 
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False

    if training_args.local_rank != -1:
        dist.barrier(device_ids=[training_args.local_rank])

    data_module = make_pretrain_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        data_collator=data_module["data_collator"],
    )

    torch.cuda.empty_cache()

    trainer.train()

    if training_args.local_rank != -1:
        dist.destroy_process_group()

    save_path = "Flow2_CoP"

    if training_args.local_rank == 0:

        print(f"Saving model to {save_path}...")
        
        trainer.save_state()
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        lora_config.save_pretrained(save_path)
        
        with open(os.path.join(save_path, "training_args.json"), "w") as f:
            json.dump(training_args.to_dict(), f, indent=4)
        
        with open(os.path.join(save_path, "training_metrics.json"), "w") as f:
            json.dump(trainer.state.log_history, f)
    
        with open(os.path.join(save_path, "training_log.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=4)
        
        print("Training complete, model saved successfully.")


if __name__ == "__main__":
    train()
