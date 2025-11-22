import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Literal, Tuple


@dataclass
class GRPOConfig:
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
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = field(default="reinforce_with_baseline")
    train_batch_size: int = field(default = 16)
    gradient_accumulation_steps: int = field(default = 4)
    rollout_batch_size: int = field(default = 16)
    group_size: int = field(default = 4)
    cliprange: float = field(default = 0.2)
    n_grpo_steps: int = field(default = 200)
    n_train_steps_per_rollout_batch: int = field(default = 4)
    advantage_eps: float = field(default = 1e-6)
    normalize_by_std: bool = False
    grad_clip: float = field(default = 1)
    num_epochs: int = field(default = 4)

    # Optim parameters
    lr_scheduler: str = field(default = "cosine_with_min_lr")
    lr: float = field(default = 4e-5)
    lr_scheduler_kwargs: Dict[str, float] = field(default_factory = lambda : {"min_lr_rate": 0.1})
    weight_decay: float = field(default = 0)
    betas: Tuple[float] = field(default_factory = lambda : (0.9, 0.95))

    # Logging parameters
    wandb_entity: Optional[str] = field(default=None)
    wandb_project: Optional[str] = field(default=None)
    wandb_run_name: Optional[str] = field(default=None)
    log_every: Optional[int] = field(default=None)
    eval_every: Optional[int] = field(default=10)
    eval_iters: Optional[int] = field(default=100)

    # Eval parameter
    do_eval: bool = False
    do_eval_before_train: bool = False
    temperature: float = field(default = 1.0)
    top_p: float = field(default = 1)
    min_tokens: int = field(default = 4)
    max_tokens: int = field(default = 1024)
    stop: str = field(default = "</answer>")
    gpu_memory_utilization: float = field(default=0.275)
    eval_batch_size: int = field(default = 2)


    def __post_init__(self):
        prompt_template_path_dict = {"r1_zero": "cs336_alignment/prompts/r1_zero.prompt",
                            "question_only": "cs336_alignment/prompts/question_only.prompt"}
        assert self.prompt in prompt_template_path_dict
        self.prompt_template_path = prompt_template_path_dict[self.prompt]

        assert self.train_batch_size % self.gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
        )
        self.micro_train_batch_size = self.train_batch_size // self.gradient_accumulation_steps
        assert self.rollout_batch_size % self.group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
        )
        self.n_prompts_per_rollout_batch = self.rollout_batch_size // self.group_size
        assert self.train_batch_size >= self.group_size, (
        "train_batch_size must be greater than or equal to group_size"
        )
        self.n_microbatches_per_rollout_batch = self.rollout_batch_size // self.micro_train_batch_size

        # self.off_policy = self.n_grpo_iterations > 1
        self.off_policy = self.n_train_steps_per_rollout_batch > 1

        assert ((self.loss_type != "grpo_clip") or self.off_policy)

        self.grpo_start_from = 0 if self.do_eval_before_train else 1