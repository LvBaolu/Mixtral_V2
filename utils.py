import argparse
from ast import literal_eval

from transformers import Trainer

def str2bool(v):
    "Fix Argparse to process bools"
    if isinstance(v, bool):
        return v
    if v.lower() == 'true'