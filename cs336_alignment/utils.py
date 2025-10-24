import os
import torch
import json
import logging
import datetime
from vllm import LLM, SamplingParams, model_executor
from transformers import AutoModelForCausalLM, AutoTokenizer
from unittest.mock import patch
from typing import Tuple, Any, List, Dict
from collections.abc import Callable
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
    cache_dir: str | os.PathLike = "models/ii/" 
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
            device_map="auto",
            cache_dir=cache_dir
        )
        print("Model loaded.")

    except Exception as e:
        logging.error(e)