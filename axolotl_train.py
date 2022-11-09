
"""Prepare and train a model on a dataset. Can also infer from a model or merge lora"""

import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import transformers.modelcard
from accelerate.logging import get_logger
from accelerate import Accelerator
from datasets import Dataset
from optimum.bettertransformer import BetterTransformer
from transformers.deepspeed import is_deepspeed_zero3_enabled

from axolotl.common.cli import TrainerCliArgs
from axolotl.logging_config import configure_logging
from axolotl.utils.dict import DictDefault
from axolotl.utils.freeze import freeze_parameters_except
from axolotl.utils.models import load_model, load_tokenizer
from axolotl.utils.trainer import setup_trainer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

configure_logging()
LOG = get_logger("axolotl.train")


@dataclass
class TrainDatasetMeta:
    """
    dataclass to capture the dataset specific options for training
    """

    train_dataset: Dataset
    eval_dataset: Optional[Dataset] = None
    total_num_steps: Optional[int] = None


def train(
    *, cfg: DictDefault, cli_args: TrainerCliArgs, dataset_meta: TrainDatasetMeta
):
    accelerator = Accelerator()
    # load the tokenizer first
    LOG.debug(
        f"loading tokenizer... {cfg.tokenizer_config or cfg.base_model_config}",
        main_process_only=True,
    )

    if accelerator.is_main_process:
        import wandb
        wandb.init(project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                config=cfg)

    tokenizer = load_tokenizer(cfg)

    train_dataset = dataset_meta.train_dataset
    eval_dataset = dataset_meta.eval_dataset
    total_num_steps = dataset_meta.total_num_steps

    # Load the model and tokenizer
    msg = "loading model"
    if cfg.adapter:
        msg += " and peft_config..."
    LOG.debug(msg)
    model, peft_config = load_model(cfg, tokenizer, inference=cli_args.inference)

    safe_serialization = cfg.save_safetensors is True
