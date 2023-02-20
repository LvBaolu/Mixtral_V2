from types import SimpleNamespace

import wandb

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

from utils import parse_args, create_alpaca_prompt_with_response, debug_trainer