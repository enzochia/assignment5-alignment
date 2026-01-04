import os
import torch
import logging
import wandb
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler
)
from vllm import SamplingParams
from .configs import SFTConfig
from torch.utils.data import DataLoader
from cs336_alignment.data_util import Train_Dataset
from cs336_alignment.utils import (
    set_random_seed,
    load_prompt_template,
    load_eval_data,
    init_vllm,
    sft_microbatch_train_step,
    load_policy_into_vllm_instance,
    tokenize_prompt_and_output,
    get_response_log_probs,
    log_generations,
    get_grad_norm
)
from cs336_alignment.grpo import (
    evaluate_model,
    get_eval_data
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_sft(
    configs: SFTConfig
) -> None:
    set_random_seed(configs.seed)
    wandb.init(
        project=configs.wandb_project,
        entity=configs.wandb_entity,
        name=configs.wandb_run_name,
        config=vars(configs)
    )

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
    train_data = Train_Dataset(configs=configs, train="sft")
    train_dataloader = DataLoader(
        train_data,
        batch_size=configs.batch_size // configs.gradient_accumulation_steps,
        shuffle=True,
        collate_fn=train_data.collate_fn_sft
    )
    micro_steps_per_epoch = len(train_dataloader)
    microbatch_size = configs.batch_size // configs.gradient_accumulation_steps
    print(f"length of train_data: {len(train_data)}, length of train_dataloader: {len(train_dataloader)}.")

    eval_prompts, eval_answers = get_eval_data(configs)

    logging.info(f"Initializing optimizer and lr scheduler.")
    optimizer = AdamW(model.parameters(), lr=configs.lr)
    lr_scheduler = get_scheduler(
        name=configs.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0.05 * configs.num_epochs * len(train_dataloader) // configs.gradient_accumulation_steps,
        num_training_steps=configs.num_epochs * len(train_dataloader) // configs.gradient_accumulation_steps,
        scheduler_specific_kwargs=configs.lr_scheduler_kwargs
    )

    model_eval = init_vllm(
        model=configs.model_path,
        device=configs.eval_device,
        gpu_memory_utilization=configs.gpu_memory_utilization if configs.train_device == configs.eval_device else 0.9,
        dtype=configs.train_dtype,
        seed=configs.seed
    )
    sampling_params_eval = SamplingParams(
        temperature=configs.temperature,
        top_p=configs.top_p,
        max_tokens=configs.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    if configs.do_eval:
        evaluate_model(
            configs=configs,
            tokenizer=tokenizer,
            model=model,
            model_inf=model_eval,
            eval_prompts=eval_prompts,
            eval_answers=eval_answers,
            step_count=0,
            sampling_params_eval=sampling_params_eval
        )

    model.train()
    total_microstep_count = -1



    for epoch in range(configs.num_epochs):
        loss_batch = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        # for prompts, outputs, answers in tqdm(train_dataloader, desc=f"Epoch #{epoch}"):
        for micro_step, (prompts, outputs, answers) in enumerate(pbar, start=1):
            total_microstep_count += 1
            batch_tokenized = tokenize_prompt_and_output(
                prompt_strs=prompts, 
                output_strs=outputs,
                tokenizer=tokenizer,
                device=configs.train_device
            )
            input_ids = batch_tokenized["input_ids"]
            labels = batch_tokenized["labels"]
            response_mask = batch_tokenized["response_mask"]

            policy_log_probs = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels
            )["log_probs"]

            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=configs.gradient_accumulation_steps,
                normalize_constant=response_mask.sum(dim=-1).float().mean()
            )
            loss_batch += loss.item()

            if (total_microstep_count - epoch * micro_steps_per_epoch + 1) % configs.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=configs.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                lr_show = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    step=total_microstep_count // configs.gradient_accumulation_steps,
                    loss=f"{loss_batch:.4f}",
                    lr=f"{lr_show:.2e}"
                )
                wandb.log({
                    "train/lr": lr_show,
                    "train/grad_norm": grad_norm,
                    "train/step_loss": loss_batch,
                }, step=total_microstep_count)
                optimizer.zero_grad()
                loss_batch = 0
                
        if (total_microstep_count - epoch * micro_steps_per_epoch + 1) % configs.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=configs.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            lr_show = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                step=total_microstep_count // configs.gradient_accumulation_steps,
                loss=f"{loss_batch:.4f}",
                lr=f"{lr_show:.2e}"
            )
            loss_batch = 0

        if configs.do_eval:
            load_policy_into_vllm_instance(model, model_eval)
            evaluate_model(
                configs=configs,
                tokenizer=tokenizer,
                model=model,
                model_inf=model_eval,
                eval_prompts=eval_prompts,
                eval_answers=eval_answers,
                step_count=total_microstep_count,
                sampling_params_eval=sampling_params_eval
            )
    
        if ((configs.checkpoint_dir is not None) and
            (epoch < configs.keep_ckpt_until_epoch)):
            step = (total_microstep_count + 1) // configs.gradient_accumulation_steps
            ckpt_path = os.path.join(configs.checkpoint_dir, f"{configs.wandb_run_name}/ckpt_epoch_{epoch}_{step}steps")
            logging.info(f"Saving trained checkpoint to {ckpt_path}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)