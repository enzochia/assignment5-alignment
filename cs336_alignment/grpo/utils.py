import os
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    get_scheduler
)
from torch.utils.data import DataLoader
from vllm import SamplingParams
from typing import List, Dict, Literal
from collections.abc import Callable
from cs336_alignment.utils import (
    set_random_seed,
    load_prompt_template,
    load_eval_data,
    init_vllm,
    load_policy_into_vllm_instance,
    log_generations,
    tokenize_prompt_and_output,
    get_response_log_probs
)
from cs336_alignment.drgrpo_grader import (
    r1_zero_reward_fn,
    question_only_reward_fn
)
from cs336_alignment.data_util import (
    MATH_SFT_Dataset,
    collate_fn_train
)
from .configs import GRPOConfig
from itertools import cycle, islice
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], Dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float = 1e-6,
    normalize_by_std: bool = True,
    device: str | torch.device | None = "cuda",
):
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            advantages: torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            raw_rewards: torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            metadata: dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    rewards_return_list: List[float] = []
    for response, truth in zip(rollout_responses, repeated_ground_truths):
        rewards_return_list.append(reward_fn(response, truth))
    rewards = [r["reward"] for r in rewards_return_list]
    raw_rewards = torch.tensor(rewards, dtype=torch.float32, device=device).reshape(-1, group_size)
    rewards_mean = raw_rewards.mean(dim=-1, keepdim=True)
    rewards_std = raw_rewards.std(dim=-1, keepdim=True) if normalize_by_std else 1
    advantages = (raw_rewards - rewards_mean) / (rewards_std + advantage_eps)
    advantages = advantages.reshape(-1)
    raw_rewards = raw_rewards.reshape(-1)
    return advantages, raw_rewards, {}


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    return -policy_log_probs * raw_rewards_or_advantages


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            metadata: dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    g_clip = (1 + cliprange) * advantages * (advantages >= 0) + \
             (1 - cliprange) * advantages * (advantages < 0)
    advantages_importance_sampled = torch.exp(policy_log_probs - old_log_probs) * advantages
    loss = -torch.min(advantages_importance_sampled, g_clip)
    return loss, {}


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    assert loss_type in {"no_baseline", "reinforce_with_baseline", "grpo_clip"}
    assert loss_type != "no_baseline" or ((raw_rewards is not None) and 
                                          (len(raw_rewards.size()) == 2) and 
                                          (raw_rewards.size()[1] == 1))
    assert advantages not in {"reinforce_with_baseline", "grpo_clip"} or \
           ((advantages is not None) and 
            (len(advantages.size()) == 2) and 
            (advantages.size()[1] == 1))
    assert old_log_probs != "grpo_clip" or ((advantages is not None) and 
                                            (len(advantages.size()) == 2))
    assert old_log_probs != "grpo_clip" or (cliprange is not None) 

    loss = None
    metadata = {}
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs
        ) 
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs
        ) 
    elif loss_type == "grpo_clip":
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange
        )
    return loss, metadata


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None= None,
) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    return tensor.masked_fill(mask.logical_not(), 0).sum(dim=dim) / mask.sum(dim=dim)



def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange
    )
    loss = masked_mean(
        tensor=loss,
        mask=response_mask
    )
    loss /= gradient_accumulation_steps
    loss.backward()
    return loss, metadata

def get_sampling_params(configs):
    sampling_params = SamplingParams(
        temperature=configs.temperature,
        top_p=configs.top_p,
        min_tokens=configs.min_tokens,
        max_tokens=configs.max_tokens,
        stop=[configs.stop],
        include_stop_str_in_output=True,
        n=configs.group_size,
        seed=configs.seed
    )

    sampling_params_eval = SamplingParams(
        temperature=configs.temperature,
        top_p=configs.top_p,
        min_tokens=configs.min_tokens,
        max_tokens=configs.max_tokens,
        stop=[configs.stop],
        include_stop_str_in_output=True
    )
    return sampling_params, sampling_params_eval

def get_inference_model(configs):
    logging.info(f"Initializing inference model.")
    model_inf = init_vllm(
        model=configs.model_path,
        device=configs.eval_device,
        gpu_memory_utilization=configs.gpu_memory_utilization if configs.train_device == configs.eval_device else 0.9,
        dtype=configs.train_dtype,
        seed=configs.seed
    )
    return model_inf


def infinite_dataloader(loader):
    while True:
        for batch in loader:
            yield batch

# TODO: 1) record rewards, 2) wandb
def train_grpo(
    configs: GRPOConfig
):
    set_random_seed(configs.seed)
    logging.info(f"Loading model and tokenizer to {torch.device(configs.train_device)}")
    model = AutoModelForCausalLM.from_pretrained(
        configs.model_path,
        torch_dtype = configs.train_dtype,
        attn_implementation = "flash_attention_2"
    ).to(torch.device(configs.train_device))
    tokenizer = AutoTokenizer.from_pretrained(configs.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info(f"Loading and processing train dataset.")
    train_data = MATH_SFT_Dataset(configs.data_train_path)
    train_dataloader = DataLoader(
        train_data,
        batch_size=configs.n_prompts_per_rollout_batch,
        shuffle=True,
        collate_fn=collate_fn_train
    )
    micro_steps_per_epoch = len(train_dataloader)
    microbatch_size = configs.train_batch_size // configs.gradient_accumulation_steps
    logging.info(f"length of train_data: {len(train_data)}, length of train_dataloader: {len(train_dataloader)}.")
 
    model_inf = get_inference_model(configs)

    logging.info(f"Loading and processing eval dataset.")
    eval_questions, eval_answers = load_eval_data("MATH", configs.data_eval_path)
    prompt_template = load_prompt_template(configs.prompt_template_path)
    eval_prompts = [prompt_template.format(question=question) for question in eval_questions]

    logging.info(f"Initializing optimizer and lr scheduler.")
    optimizer = AdamW(model.parameters(), lr=configs.lr)
    total_steps = configs.n_grpo_steps * configs.n_train_steps_per_rollout_batch * configs.rollout_batch_size // configs.train_batch_size
    lr_scheduler = get_scheduler(
        name=configs.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0.05 * total_steps,
        num_training_steps=total_steps,
        scheduler_specific_kwargs=configs.lr_scheduler_kwargs
    )

    sampling_params, sampling_params_eval = get_sampling_params(configs)

    total_step_count = -1
    model.train()
    pbar = tqdm(total=configs.n_grpo_steps, desc="GRPO", dynamic_ncols=True)
    for step, (problems, _, _, _, answers) in enumerate(islice(infinite_dataloader(train_dataloader), configs.n_grpo_steps), start=0):
        prompts = [prompt_template.format(question=problem) for problem in problems]
        outputs = model_inf.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        batch_data = {"prompts": [], "outputs": [], "answers": []}
        for p, a, o in zip(prompts, answers, outputs):
            batch_data["prompts"].extend([p for _ in range(configs.group_size)])
            batch_data["outputs"].extend([o.outputs[idx].text for idx in range(configs.group_size)])
            batch_data["answers"].extend([a for _ in range(configs.group_size)])

        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn if configs.prompt == "r1_zero" else question_only_reward_fn,
            rollout_responses=batch_data["outputs"],
            repeated_ground_truths=batch_data["answers"],
            group_size=configs.group_size,
            advantage_eps=configs.advantage_eps,
            normalize_by_std=configs.normalize_by_std,
            device=configs.train_device
        )
        # a dict of 3 tensors, each has [configs.n_prompts_per_rollout_batch * configs.group_size, seq_len]
        tokenized_results = tokenize_prompt_and_output(
            prompt_strs=batch_data["prompts"], 
            output_strs=batch_data["outputs"], 
            tokenizer=tokenizer,
            device=configs.train_device,
        )

        if configs.off_policy:
            old_log_probs_list = []
            with torch.no_grad():
                for micro_step in range(configs.n_microbatches_per_rollout_batch + 1):
                    idx_start = micro_step * configs.micro_train_batch_size
                    if idx_start < configs.rollout_batch_size:
                        idx_end = min(idx_start + configs.micro_train_batch_size,
                                      configs.rollout_batch_size)
                        log_probs_dict = get_response_log_probs(
                            model=model,
                            input_ids=tokenized_results["input_ids"][idx_start:idx_end],
                            labels=tokenized_results["labels"][idx_start:idx_end],
                            return_token_entropy=False
                        )
                        old_log_probs_list.append(log_probs_dict["log_probs"])
                old_log_probs_tensor = torch.cat(old_log_probs_list)
        
        model.train()
        micro_step_count = 0
        step_loss = 0
        for grpo_step in range(configs.n_train_steps_per_rollout_batch):
            for micro_step in range(configs.n_microbatches_per_rollout_batch + 1):
                idx_start = micro_step * configs.micro_train_batch_size
                if idx_start < configs.rollout_batch_size:
                    idx_end = min(idx_start + configs.micro_train_batch_size,
                                  configs.rollout_batch_size)
                    log_probs_dict = get_response_log_probs(
                        model=model,
                        input_ids=tokenized_results["input_ids"][idx_start:idx_end],
                        labels=tokenized_results["labels"][idx_start:idx_end],
                        return_token_entropy=False
                    )
                    old_log_probs_tensor = old_log_probs_tensor if configs.off_policy else \
                                           log_probs_dict["log_probs"].detach()
                    loss, metadata = grpo_microbatch_train_step(
                        policy_log_probs=log_probs_dict["log_probs"],
                        response_mask=tokenized_results["response_mask"][idx_start:idx_end],
                        gradient_accumulation_steps=configs.gradient_accumulation_steps,
                        loss_type=configs.loss_type,
                        raw_rewards=raw_rewards[idx_start:idx_end],
                        advantages=advantages[idx_start:idx_end].unsqueeze(-1),
                        old_log_probs=old_log_probs_tensor,
                        cliprange=configs.cliprange,
                    ) 
                    micro_step_count += 1
                    step_loss += loss
                    if ((micro_step_count % configs.gradient_accumulation_steps == 0) or
                        (micro_step_count == configs.rollout_batch_size)):
                        clip_grad_norm_(model.parameters(), max_norm=configs.grad_clip)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                        total_step_count += 1
                        pbar.set_postfix(
                            step=step,
                            step_loss=f"{step_loss:.4f}",
                            step_reward=f"place_holder"
                        )
                        step_loss = 0

        if (configs.do_eval and
            step % configs.eval_every == 0):
            model.eval()
            load_policy_into_vllm_instance(model, model_inf)

            eval_results = log_generations(
                tokenizer=tokenizer,
                model_vllm=model_inf,
                model=model,
                prompts=eval_prompts,
                answers=eval_answers,
                step=step,
                sampling_params=sampling_params_eval,
                log_to=configs.log_dir,
                device=configs.eval_device,
                reward=configs.prompt,
                temperature=configs.temperature,
                top_p=configs.top_p,
                max_tokens=configs.max_tokens,
                eval_batch_size=configs.eval_batch_size
            )
            logging.info(eval_results)
        pbar.update(1)

    if configs.checkpoint_dir is not None:
        ckpt_path = os.path.join(configs.checkpoint_dir, f"ckpt_{step}rollouts_{total_step_count}steps")
        logging.info(f"Saving trained checkpoint to {ckpt_path}")
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)

"""
note: start from outputs/ckpt/ckpt_2epoch_220steps/ model, 
 {'step': 0, 'eval_sample_size': 5000, 'count_correct_format': 3912.0, 'count_correct_answer': 1909.0, 'total_reward': 1909.0, 'avg_token_entropy': 0.400390625, 
 'avg_response_len': 139.8553924560547, 'avg_correct_response_len': 110.40387725830078, 'avg_incorrect_response_len': 158.04464721679688}
 {'step': 40, 'eval_sample_size': 5000, 'count_correct_format': 4368.0, 'count_correct_answer': 1030.0, 'total_reward': 1030.0, 'avg_token_entropy': 0.5703125, 
 'avg_response_len': 160.14520263671875, 'avg_correct_response_len': 64.53981018066406, 'avg_incorrect_response_len': 184.9496307373047}
"""
