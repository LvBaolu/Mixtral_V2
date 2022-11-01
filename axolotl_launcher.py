
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
    with open(config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    cfg["axolotl_config_path"] = config
    # if there are any options passed in the cli, if it is something that seems valid from the yaml,
    # then overwrite the value
    cfg_keys = cfg.keys()
    for k, _ in kwargs.items():
        # if not strict, allow writing to cfg even if it's not in the yml already
        if k in cfg_keys or not cfg.strict:
            # handle booleans
            if isinstance(cfg[k], bool):
                cfg[k] = bool(kwargs[k])
            else:
                cfg[k] = kwargs[k]
    return cfg


def dump_config(parsed_cfg):
    # dump config to yaml file
    with open("dumped_config.yaml", "w") as file:
        print(f"Dumping config to dumped_config.yaml: {type(parsed_cfg)}")
        yaml_config = yaml.dump(parsed_cfg, file)

def do_cli(config_fname: Path = Path("examples/"), **kwargs):
    # pylint: disable=duplicate-code
    parsed_cfg = load_cfg(config_fname, **kwargs)

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    
    if ddp:
        from torch.distributed import init_process_group
        init_process_group(backend="nccl")
        if int(os.environ["RANK"]) == 0:
            print(f"We are in rank {os.environ['RANK']}, initializing wandb")
            wandb.init(project=parsed_cfg["wandb_project"], 
                       entity=parsed_cfg["wandb_entity"], 