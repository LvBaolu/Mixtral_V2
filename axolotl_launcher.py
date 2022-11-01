
"""
CLI to run training on a model
"""

## TODO
## - Overwrite the output dir with the run-id (we are overwriting the output dir everytime)
## - Do something with Axolotl params so they can be logged and injected somehow, right now we
## log HF Trainer params that get created inside axolotl, we may have conflicting params.
## One solution is to create a nested dict with the params and log that, but then we need to
## make sure that the params are not conflicting with HF Trainer params.
## Something like: wandb.init(..., config = {axolotl_params: parsed_config})
## - Pull latest axolotl and rebuild the docker image.

import pickle
import torch
import logging, os, yaml
from pathlib import Path
import wandb
import fire
import transformers


from axolotl.cli import (
    check_accelerate_default_config,
    check_user_token,
    load_datasets,
    print_axolotl_text_art,
    validate_config,
    prepare_optim_env,
    normalize_config,
    setup_wandb_env_vars
)
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils.dict import DictDefault

LOG = logging.getLogger("axolotl.cli.train")


def load_cfg(config: Path = Path("examples/"), **kwargs):
    # load the config from the yaml file