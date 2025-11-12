import os
import torch
import logging
from transformers import HfArgumentParser
from .configs import GRPOConfig
from .utils import train_grpo

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    parser = HfArgumentParser(GRPOConfig)
    configs = parser.parse_args_into_dataclasses()[0]
    train_grpo(configs)