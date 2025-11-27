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
from cs336_alignment.data_util import (
    MATH_SFT_Dataset,
    collate_fn_sft
)
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
    train_data = MATH_SFT_Dataset(configs.data_sft_path)
    train_dataloader = DataLoader(
        train_data,
        batch_size=configs.batch_size // configs.gradient_accumulation_steps,
        shuffle=True,
        collate_fn=collate_fn_sft
    )
    micro_steps_per_epoch = len(train_dataloader)
    microbatch_size = configs.batch_size // configs.gradient_accumulation_steps
    print(f"length of train_data: {len(train_data)}, length of train_dataloader: {len(train_dataloader)}.")

    logging.info(f"Loading and processing eval dataset.")
    eval_questions, eval_answers = load_eval_data("MATH", configs.data_eval_path)
    prompt_template = load_prompt_template(configs.prompt_template_path)
    eval_prompts = [prompt_template.format(question=question) for question in eval_questions]

    logging.info(f"Initializing optimizer and lr scheduler.")
    optimizer = AdamW(model.parameters(), lr=configs.lr)
    lr_scheduler = get_scheduler(
        name=configs.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0.05 * configs.num_epochs * len(train_dataloader) // configs.gradient_accumulation_steps,
        num_training_steps=configs.num_epochs * len(train_dataloader) // configs.gradient_accumulation_steps,
        scheduler_specific_kwargs=configs.lr_scheduler_kwargs
    )

    if configs.do_eval:
        logging.info(f"Initializing eval model.")
        model_eval = init_vllm(
            model=configs.model_path,
            device=configs.eval_device,
            gpu_memory_utilization=configs.gpu_memory_utilization if configs.train_device == configs.eval_device else 0.9,
            dtype=configs.train_dtype,
            seed=configs.seed
        )
        eval_sampling_params = SamplingParams(
            temperature=configs.temperature,
            top_p=configs.top_p,
            max_tokens=configs.max_tokens,
            stop=["</answer>"],
            include_stop_str_in_output=True
        )

        eval_results = log_generations(
            tokenizer=tokenizer,
            model_vllm=model_eval,
            model=model,
            prompts=eval_prompts,
            answers=eval_answers,
            step=0,
            sampling_params=eval_sampling_params,
            log_to=configs.log_dir,
            device=configs.train_device, # this may be wrong
            reward=configs.prompt,
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            eval_batch_size= configs.eval_batch_size # lower it if OOM
        )
        logging.info(f"Eval metric results:")
        logging.info(eval_results)
        wandb.log({
            "eval/format_reward": eval_results["count_correct_format"] / eval_results["eval_sample_size"],
            "eval/answer_reward": eval_results["count_correct_answer"] / eval_results["eval_sample_size"],
            "eval/total_reward": eval_results["total_reward"] / eval_results["eval_sample_size"],
            "eval/avg_token_entropy": eval_results["avg_token_entropy"],
            "eval/avg_response_len": eval_results["avg_response_len"],
            "eval/avg_correct_response_len": eval_results["avg_correct_response_len"],
            "eval/avg_incorrect_response_len": eval_results["avg_incorrect_response_len"]
        }, step=0)
    else:
        model_eval = None

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
                wandb.log({
                    "train/lr": lr_show,
                    "train/grad_norm": get_grad_norm(model.parameters()),
                    "train/step_loss": loss_batch,
                }, step=total_microstep_count // configs.gradient_accumulation_steps)
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
            logging.info(f"Evaluating...")
            load_policy_into_vllm_instance(model, model_eval)
            eval_results = log_generations(
                tokenizer=tokenizer,
                model_vllm=model_eval,
                model=model,
                prompts=eval_prompts,
                answers=eval_answers,
                step=total_microstep_count // configs.gradient_accumulation_steps,
                sampling_params=eval_sampling_params,
                log_to=configs.log_dir,
                device=configs.train_device, # this may be wrong
                reward=configs.prompt,
                temperature=1.0,
                top_p=1.0,
                max_tokens=1024,
                eval_batch_size= configs.eval_batch_size # lower it if OOM
            )
            logging.info(f"Eval metric results:")
            logging.info(eval_results)
            wandb.log({
                "eval/format_reward": eval_results["count_correct_format"] / eval_results["eval_sample_size"],
                "eval/answer_reward": eval_results["count_correct_answer"] / eval_results["eval_sample_size"],
                "eval/total_reward": eval_results["total_reward"] / eval_results["eval_sample_size"],
                "eval/avg_token_entropy": eval_results["avg_token_entropy"],
                "eval/avg_response_len": eval_results["avg_response_len"],
                "eval/avg_correct_response_len": eval_results["avg_correct_response_len"],
                "eval/avg_incorrect_response_len": eval_results["avg_incorrect_response_len"]
            }, step=total_microstep_count // configs.gradient_accumulation_steps)
    
        if configs.checkpoint_dir is not None:
            step = (total_microstep_count + 1) // configs.gradient_accumulation_steps
            ckpt_path = os.path.join(configs.checkpoint_dir, f"ckpt_epoch_{epoch}_{step}steps")
            logging.info(f"Saving trained checkpoint to {ckpt_path}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

"""
original Qeuen2.5-Math-1.5B eval results:
{'step': 10, 'eval_sample_size': 5000, 'count_correct_format': 856.0, 'count_correct_answer': 142.0, 'total_reward': 142.0, 'avg_token_entropy': 0.7421875, 
'avg_response_len': 347.2590026855469, 'avg_correct_response_len': 136.34506225585938, 'avg_incorrect_response_len': 353.4240417480469}
after 1 epoch of SFT:
{'step': 441, 'eval_sample_size': 5000, 'count_correct_format': 3245.0, 'count_correct_answer': 1493.0, 'total_reward': 1493.0, 'avg_token_entropy': 0.5, 
'avg_response_len': 145.38539123535156, 'avg_correct_response_len': 105.8305435180664, 'avg_incorrect_response_len': 162.2246856689453}
after 2 epochs of SFT:
{'step': 883, 'eval_sample_size': 5000, 'count_correct_format': 4050.0, 'count_correct_answer': 1982.0, 'total_reward': 1982.0, 'avg_token_entropy': 0.396484375, 
'avg_response_len': 146.68759155273438, 'avg_correct_response_len': 114.66548919677734, 'avg_incorrect_response_len': 167.71737670898438}
"""





    