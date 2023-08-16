import argparse
from ast import literal_eval

from transformers import Trainer

def str2bool(v):
    "Fix Argparse to process bools"
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(config):
    print("Running with the following config")
    parser = argparse.Argum