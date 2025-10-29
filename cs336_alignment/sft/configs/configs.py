import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class SFTConfig:
    # Model and data parameters
    model_path: Optional[str] = field(default="models/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2/")
    train_dtype: torch.dtype = field(default=torch.bfloat16)
    data_train_path: Optional[str] = field(default="data/MATH/train.jsonl")
    data_sft_path: Optional[str] = field(default="data/MATH/sft.jsonl")
    data_eval_path: Optional[str] = field(default="data/MATH/validation.jsonl")
    prompt: Optional[str] = field(default="r1_zero")
    
    # Device parameters
    device: str = field(default = torch.device("cuda") if torch.cuda.is_available() else 
                        (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")))
    train_device: str = field(default = "cuda:0")
    eval_device: str = field(default = "cuda:1")

    # Training parameters
    checkpoint_dir: str = field(default = "outputs/ckpt/")
    log_dir: str = field(default = "outputs/logs/")
    seed: int = field(default = 2048)
    lr_scheduler: str = field(default = "cosine_with_min_lr")
    lr: float = field(default = 4e-5)
    lr_scheduler_kwargs: Dict[str, float] = field(default_factory = lambda : {"min_lr_rate": 0.1})
    batch_size: int = field(default = 16)
    gradient_accumulation_steps: int = field(default = 4)
    num_epochs: int = field(default = 4)

    # Logging parameters
    wandb_entity: Optional[str] = field(default=None)
    wandb_project: Optional[str] = field(default=None)
    wandb_run_name: Optional[str] = field(default=None)
    log_every: Optional[int] = field(default=None)
    eval_every: Optional[int] = field(default=None)
    eval_iters: Optional[int] = field(default=100)

    # Eval parameter
    do_eval: bool = False
    temperature: float = field(default = 1.0)
    top_p: float = field(default = 1)
    max_tokens: int = field(default = 1024)


    def __post_init__(self):
        prompt_template_path_dict = {"r1_zero": "cs336_alignment/prompts/r1_zero.prompt",
                            "question_only": "cs336_alignment/prompts/question_only.prompt"}
        assert self.prompt in prompt_template_path_dict
        self.prompt_template_path = prompt_template_path_dict[self.prompt]