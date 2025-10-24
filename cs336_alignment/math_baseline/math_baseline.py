import os
import torch
import argparse
import logging
from vllm import LLM, SamplingParams
from cs336_alignment.utils import (
    init_vllm,
    evaluate_vllm,
    load_eval_data,
    calculate_metrics,
    load_prompt_template
)
from cs336_alignment.drgrpo_grader import (
    r1_zero_reward_fn,
    question_only_reward_fn
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/Qwen2.5-Math-1.5B")
    parser.add_argument("--data", type=str, default="MATH", choices=["MATH", "gsm8k"])
    parser.add_argument("--data_path", type=str, default="data/MATH/validation.jsonl", 
                        choices=["data/MATH/validation.jsonl", "data/gsm8k/test.jsonl"])
    parser.add_argument("--prompt_template_path", type=str, default="cs336_alignment/prompts/r1_zero.prompt",
                        choices=["cs336_alignment/prompts/r1_zero.prompt", "cs336_alignment/prompts/question_only.prompt"])
    parser.add_argument("--reward", type=str, default="r1_zero", choices=["r1_zero", "question_only"])
    parser.add_argument("--save_to", type=str, default="outputs/math_baseline/")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_k", type=float, default=1)
    parser.add_argument("--max_tokens", type=float, default=1024)
    parser.add_argument("--stop", type=str, default="</answer>")
    parser.add_argument("--do_not_include_str_in_output", action="store_false")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else 
                        ("mps" if torch.backends.mps.is_available() else "cpu")))
    args = parser.parse_args()

    model = init_vllm(model=args.model, device=args.device)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop=["</answer>"]
    )
    
    questions, answers = load_eval_data(data_name=args.data, path=args.data_path)
    reward_fn = r1_zero_reward_fn if args.reward == "r1_zero" else question_only_reward_fn
    prompt_template = load_prompt_template(args.prompt_template_path)
    prompts = [prompt_template.format(question=q) for q in questions]
    timestamp_str, results = evaluate_vllm(
        vllm_model=model,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_sampling_params=sampling_params,
        answers=answers,
        save_to=args.save_to
    )

    metric_dict = calculate_metrics(    
        results=results,
        save_to=args.save_to,
        timestamp_str=timestamp_str
    )
    logging.info(metric_dict)