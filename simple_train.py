from types import SimpleNamespace

import wandb

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

from utils import parse_args, create_alpaca_prompt_with_response, debug_trainer_data

WANDB_PROJECT = "mixtral"
WANDB_ENTITY = "capecape"
WANDB_TAGS = None

config = SimpleNamespace(
    dataset_id = "c-s-ale/alpaca-gpt4-data",
    split = "train",
    model_id = "mistralai/Mixtral-8x7B-v0.1",
    batch_size = 1, # what my GPU can handle, depends on how many layers are we training
    effective_batch_size = 8, # batch size for gradient accumulation
    gradient_checkpointing = False,
    load_in_4bit=True,
    load_in_8bit=False,
    max_seq_length = 512,
    num_train_epochs = 3, # we do 3 pasess over the dataset.
    lr = 2e-5,
    log_model=False,
    # for debug purposes
    max_steps=-1, 
    debug_data=False,
)


def get_train_args(config, output_dir = "./output/"):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=max(config.batch_size//2, 1),
        bf16=True,
        learning_rate=config.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_steps=config.max_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        evaluation_strategy="no",
        # logging strategies
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
    )
    return training_args

def main(config):
    accelerator = Accelerator()

    if accelerator.is_main_process:
        wandb.init(
            project=WANDB_PROJECT, 
            entity=WANDB_ENTITY, 
            job_type="train",
            tags=WANDB_TAGS,
            config=config)

    # some sane defaults computations
    config.gradient_accumulation_steps = (1024 // config.max_seq_length) * config.effective_batch_size // config.batch_size
    config.tokens_per_step = 8 * config.max_seq_length * config.batch_size * config.gradient_accumulation_steps
    print(f"\nWe are training for {config.max_steps} steps with an effective batch size of {con