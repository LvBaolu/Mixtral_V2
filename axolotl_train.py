
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
