# Mixtral_V2

Project Mixtral_V2 fine-tunes the Mixtral model.

## Requirements

An optimized Docker image is available to run the code. You can build it or directly pull it [from this repo](https://github.com/LvBaolu/Mixtral_V2/pkgs/container/Mixtral_V2).

`flash_attn` may be hard to install on some configurations. This Docker image is specifically built to work with H100 GPUs.

## Run

- You can execute the `simple_inference.py` script to first test the model. It has proved to run efficiently on systems with equipped with an A100 and 40GB of memory!

## Train

You can test your system setup by running the `simple_train.py` script. This will train a mod