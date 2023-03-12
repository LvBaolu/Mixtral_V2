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
    model_id = "mistralai/Mixtra