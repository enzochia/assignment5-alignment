import os
import torch
import json
import logging
import datetime
from tqdm import tqdm
from vllm import LLM, SamplingParams, model_executor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from unittest.mock import patch
from typing import Tuple, Any, List, Dict
from collections.abc import Callable
from .drgrpo_grader import (
    r1_zero_reward_fn,
    question_only_reward_fn
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def init_vllm(model: str,
              device: str | torch.device | None = "cuda:0",
              gpu_memory_utilization: float = 0.9,
              dtype: torch.dtype | None = torch.bfloat16,
            #   dtype: torch.dtype | None = torch.float16,
              seed: int = 4096) -> LLM:
    model_executor.set_random_seed(seed)
    # world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    # profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
    # with world_size_patch, profiling_patch:
    llm = LLM(
        model=model,
        device=device,
        dtype=dtype,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization
    )
    return llm


def load_eval_data(
        data_name: str,
        path: str | os.PathLike
    ) -> Tuple[Any]:
    questions: List[str] = []
    answers: List[str] = [] 
    question_label: Dict[str, str] = {"MATH": "problem", "gsm8k": "question"}
    answer_label: Dict[str, str] = {"MATH": "answer", "gsm8k": "ground_truth"}
    with open(path, "r", encoding="utf-8") as f:
        logging.info(f"Loading eval {data_name} data from {path}.")
        for line in f:
            try:
                data = json.loads(line)
                questions.append(data[question_label[data_name]])
                answers.append(data[answer_label[data_name]])
            except Exception as e:
                logging.error(e)
    logging.info(f"Loading completed.")
    return questions, answers


def load_prompt_template(path: str | os.PathLike) -> str:
    with open(path, "r") as f:
        prompt_template = f.read()
    return prompt_template


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    answers: List[str],
    save_to: str | os.PathLike
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Evaluate a language model on a list of prompts, compute evaluation metrics, and serialize results to disk.
    """
    results = []
    logging.info(f"Generating...")
    outputs = vllm_model.generate(prompts=prompts, sampling_params=eval_sampling_params)
    logging.info(f"Evaluating...")
    for output, answer in zip(outputs, answers):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        reward = reward_fn(generated_text, answer)
        results.append(
            {
                "prompt": prompt,
                "answer": answer,
                "generated_text": generated_text,
                "format_reward": reward["format_reward"],
                "answer_reward": reward["answer_reward"],
                "reward": reward["reward"]
            }
        )
    
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    if save_to is not None:
        logging.info(f"Saving evaluation results...")
        save_to_path = os.path.join(save_to, f"{timestamp_str}.json")
        try:
            with open(save_to_path, "w") as f:
                json.dump(results, f)
        except Exception as e:
            logging.error(e)
    return timestamp_str, results



def calculate_metrics(
    results: List[Dict[str, str]],
    save_to: str | os.PathLike,
    timestamp_str: str
) -> Dict:
    count_dict: Dict[str, int] = {
        "correct_format_correct_answer": 0,
        "incorrect_format_correct_answer": 0,
        "correct_format_incorrect_answer": 0,
        "incorrect_format_incorrect_answer": 0
    }
    example_dict: Dict[str, List[str]] = {
        "correct_format_correct_answer": [],
        "incorrect_format_correct_answer": [],
        "correct_format_incorrect_answer": [],
        "incorrect_format_incorrect_answer": []
    }
    for r in results:
        if ((r["format_reward"] == 1) and 
            (r["answer_reward"] == 1)):
            count_dict["correct_format_correct_answer"] += 1
            if len(example_dict["correct_format_correct_answer"]) < 10:
                example_dict["correct_format_correct_answer"].append(
                    f"Prompt: {r['prompt']}\n answer: {r['answer']}\n generated text: {r['generated_text']}\n")
        elif ((r["format_reward"] == 1) and 
              (r["answer_reward"] == 0)):
            count_dict["correct_format_incorrect_answer"] += 1
            if len(example_dict["correct_format_incorrect_answer"]) < 10:
                example_dict["correct_format_incorrect_answer"].append(
                    f"Prompt: {r['prompt']}\n answer: {r['answer']}\n generated text: {r['generated_text']}\n")
        elif r["answer_reward"] == 1:
            count_dict["incorrect_format_correct_answer"] += 1
            if len(example_dict["incorrect_format_correct_answer"]) < 10:
                example_dict["incorrect_format_correct_answer"].append(
                    f"Prompt: {r['prompt']}\n answer: {r['answer']}\n generated text: {r['generated_text']}\n")
        else:
            count_dict["incorrect_format_incorrect_answer"] += 1
            if len(example_dict["incorrect_format_incorrect_answer"]) < 10:
                example_dict["incorrect_format_incorrect_answer"].append(
                    f"Prompt: {r['prompt']}\n answer: {r['answer']}\n generated text: {r['generated_text']}\n")
    
    total_count = len(results)
    metric_dict: Dict[str, float] = {
        "accuracy": count_dict["correct_format_correct_answer"] / total_count if total_count else 0,
        "total_count": total_count
    }
    metric_dict.update(count_dict)
    save_to_path_metrics = os.path.join(save_to, f"{timestamp_str}_metrics.json")
    with open(save_to_path_metrics, "w") as f:
        json.dump(metric_dict, f)

    example_list: List[str] = ["\n\n######### correct_format_correct_answer #########"] + \
                              example_dict["correct_format_correct_answer"] + \
                              ["\n\n######### correct_format_incorrect_answer #########"] + \
                              example_dict["correct_format_incorrect_answer"] + \
                              ["\n\n######### incorrect_format_correct_answer #########"] + \
                              example_dict["incorrect_format_correct_answer"] + \
                              ["\n\n######### incorrect_format_incorrect_answer #########"] + \
                              example_dict["incorrect_format_incorrect_answer"]
    save_to_path_examples = os.path.join(save_to, f"{timestamp_str}_examples.json")
    with open(save_to_path_examples, "w") as f:
        for line in example_list:
            f.write(line + "\n\n")
    return metric_dict


def load_model(
    model_id: str = "Qwen/Qwen2.5-Math-1.5B",
    cache_dir: str | os.PathLike = "models/bf16/" 
) -> None:
    logging.info(f"Loading tokenizer for {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir 
        )

        logging.info(f"Loading model {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            cache_dir=cache_dir
        )
        print("Model loaded.")

    except Exception as e:
        logging.error(e)
    return tokenizer, model


def tokenize_prompt_and_output(
    prompt_strs: List[str], 
    output_strs: List[str], 
    tokenizer: PreTrainedTokenizer,
    device: str | torch.device | None = "cuda",
):
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    bs = len(prompt_strs)
    prompts_tokenized = [tokenizer.encode(p) for p in prompt_strs]
    outputs_tokenized = [tokenizer.encode(o) for o in output_strs]
    max_seq_len = max(len(p) + len(o) for p, o in zip(prompts_tokenized, outputs_tokenized))

    input_ids = torch.full((bs, max_seq_len), tokenizer.pad_token_id, device=device)
    labels = torch.full((bs, max_seq_len - 1), tokenizer.pad_token_id, device=device)
    response_mask = torch.zeros((bs, max_seq_len - 1), dtype=torch.bool, device=device)

    for idx_seq in range(bs):
        seq = prompts_tokenized[idx_seq] + outputs_tokenized[idx_seq]
        input_ids[idx_seq, :len(seq)] = torch.tensor(seq, device=device, dtype=torch.long)
        labels[idx_seq, :(len(seq) - 1)] = torch.tensor(seq[1:], device=device, dtype=torch.long)
        response_mask[idx_seq, (len(prompts_tokenized[idx_seq]) - 1):(len(seq) - 1)] = True
    input_ids = input_ids[:, :-1]

    data_dict = {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }
    return data_dict


def _logsumexp(logits: torch.Tensor) -> torch.Tensor:

    max_logit = torch.max(logits, dim=-1, keepdim=True)[0]
    logits = logits - max_logit
    return max_logit + torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
        containing unnormalized logits.
    Returns:
        torch.Tensor: Shape (batch_size, sequence_length). The entropy for each next-token
        prediction.
    """
    log_prob = logits - _logsumexp(logits)
    return -(log_prob * torch.exp(log_prob)).sum(dim=-1, keepdim=False)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False
) -> dict[str, torch.Tensor]:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    # [..., seq_len, vocab_size]
    logits = model(input_ids).logits
    # [..., seq_len, 1]
    log_probs = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)) - \
                _logsumexp(logits)
    result_dict = {
            "log_probs": log_probs.squeeze(-1)
        }
    if return_token_entropy:
        result_dict["token_entropy"] = compute_entropy(logits)
    return result_dict


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None= None
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    tensor_masked = tensor.masked_fill(mask.logical_not(), 0)
    tensor_sum = tensor_masked.sum(dim=dim)
    return tensor_sum / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
            SFT policy being trained.
        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
            prompt/padding.
        gradient_accumulation_steps Number of microbatches per optimizer step.
        normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
                this so we can log it.
            metadata Dict with metadata from the underlying loss call, and any other statistics you
                might want to log.
    """
    loss = -masked_normalize(
        tensor=policy_log_probs, 
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=-1
    )
    loss = loss.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {"metadata": None}


def log_generations(
    tokenizer: PreTrainedTokenizer,
    model_vllm: LLM,
    model: PreTrainedModel,
    prompts: List[str],
    answers: List[str],
    step: int,
    sampling_params: SamplingParams = None,
    log_to: str | os.PathLike | None = None,
    device: str | torch.device | None = "cuda",
    reward: str = "r1_zero",
    temperature: float = 1.0,
    top_k: float = 1.0,
    max_tokens: int = 1024,
    eval_batch_size: int = 4
) -> Dict[str, Any]:
    if sampling_params is None:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=["</answer>"],
            include_stop_str_in_output=True
        )
    reward_fn = r1_zero_reward_fn if reward == "r1_zero" else question_only_reward_fn
    timestamp_str, results = evaluate_vllm(
        vllm_model=model_vllm,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_sampling_params=sampling_params,
        answers=answers,
        save_to=None
    )

    outputs = [r["generated_text"] for r in results]
    rewards = [r["reward"] for r in results]
    # rewards = torch.tensor(rewards, device=device)
    count_correct_format = sum(r["format_reward"] for r in results)
    count_correct_answer = sum(r["answer_reward"] for r in results)
    count_reward = sum(r["reward"] for r in results)
    token_dict = tokenize_prompt_and_output(prompts, outputs, tokenizer, device)

    log_prob_list = []
    token_entropy_list = []
    for idx in tqdm(range(0, len(prompts), eval_batch_size)):
        input_ids = token_dict["input_ids"][idx:(idx + eval_batch_size)]
        labels = token_dict["labels"][idx:(idx + eval_batch_size)]
        with torch.no_grad():
            log_prob_dict = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True
            )
        log_prob_list.append(log_prob_dict["log_probs"])
        token_entropy_list.append(log_prob_dict["token_entropy"])
    log_probs = torch.cat(log_prob_list, dim=0).to(device)
    token_entropies = torch.cat(token_entropy_list, dim=0).to(device)

    avg_token_entropy = masked_normalize(
        tensor=token_entropies,
        mask=token_dict["response_mask"],
        normalize_constant=token_dict["response_mask"].sum()
    )
    response_len = token_dict["response_mask"].sum(dim=-1)
    avg_response_len = response_len.mean()
    avg_correct_response_len = response_len[rewards > 0.5].mean()
    avg_incorrect_response_len = response_len[rewardss < 0.5].mean()

    metric_dict = {
        "step": step,
        "eval_sample_size": len(prompts),
        "count_correct_format": count_correct_format,
        "count_correct_answer": count_correct_answer,
        "total_reward": count_reward,
        "avg_token_entropy": avg_token_entropy,
        "avg_response_len": avg_response_len,
        "avg_correct_response_len": avg_correct_response_len,
        "avg_incorrect_response_len": avg_incorrect_response_len
    }
    if log_to is not None:
        metric_path = os.path.join(log_to, f"eval_metrics_step_{step}.json")
        if not os.path.exists(log_to):
            os.makedirs(log_to)
        with open(metric_path, "w") as f:
            json.dump(metric_dict, f)

        text_path = os.path.join(log_to, f"eval_texts_step_{step}.json")
        with open(text_path, "w") as f:
            for line in results:
                f.write(line + "\n\n")

    return metric_dict
