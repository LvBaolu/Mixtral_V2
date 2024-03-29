# Mixtral_V2

Project Mixtral_V2 fine-tunes the Mixtral model.

## Requirements

An optimized Docker image is available to run the code. You can build it or directly pull it [from this repo](https://github.com/LvBaolu/Mixtral_V2/pkgs/container/Mixtral_V2).

`flash_attn` may be hard to install on some configurations. This Docker image is specifically built to work with H100 GPUs.

## Run

- You can execute the `simple_inference.py` script to first test the model. It has proved to run efficiently on systems with equipped with an A100 and 40GB of memory!

## Train

You can test your system setup by running the `simple_train.py` script. This will train a model on a small dataset, typically completing in about an hour on an 8xH100 machine.

## Axolotl

Our experiments are conducted using the Axolotl library. A dedicated Docker image can be pulled from this repo for this purpose. It provides an execution environment for the `axolotl_launcher.py` script as a replacement for the standard axolotl.cli.train script. One can inject parameters directly from the W&B UI to launch a job.
