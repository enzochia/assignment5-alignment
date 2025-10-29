import os
import torch
import logging
from transformers import HfArgumentParser
from .configs import SFTConfig
from .utils import run_sft

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    parser = HfArgumentParser(SFTConfig)
    configs = parser.parse_args_into_dataclasses()[0]
    run_sft(configs)