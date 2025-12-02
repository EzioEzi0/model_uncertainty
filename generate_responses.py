
"""
Uses GPU 0 (vLLM) + GPU 1 (HF)

"""

import argparse
import json
import pickle
import logging
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch
from vllm import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.generation_utils_hybrid import (
    generate_with_vllm_and_extract_hidden_state,
    generate_batch_with_vllm,
    build_chat_prompt
)

# Set CUDA devices for this script
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def setup_logging(save_dir: Path, suffix: str):
    """Setup logging to both file and console"""
    log_file = save_dir / f"generation_part1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses for APPS dataset")
    parser.add_argument("--data_type", type=str, default="test", choices=["test", "train"],
                        help="Data type: test or train")
    parser.add_argument("--run_id", type=int, default=1,
                        help="Run ID for multiple train runs (1, 2, 3, ...)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Config
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    script_dir = Path(__file__).parent
    save_dir = script_dir / "output"
    save_dir.mkdir(exist_ok=True)

    # Select data file based on data_type
    if args.data_type == "test":
        data_path = script_dir / "data" / "apps_test_1.pkl"  # 测试用，改回 apps_test_split.pkl 跑全量
        output_suffix = ""
    else:  # train
        data_path = script_dir / "data" / "apps_train.pkl"  # 2670 questions
        output_suffix = f"_run{args.run_id}"

    num_samples = 10
    max_new_tokens = 4096 * 2
    hidden_state_layer = -1
    save_every = 100  # Save checkpoint every N questions

    # Greedy params
    greedy_temperature = 0.05
    greedy_top_p = 1.0
    greedy_top_k = 50

    # Sampling params
    sampling_temperature = 1.0
    sampling_top_p = 1.0
    sampling_top_k = 50

    system_prompt = "You are an expert programmer. Read the problem and write a Python solution. Keep your reasoning **EXTREMELY BRIEF** (1-2 sentences max). Output format: Reasoning: [1-2 sentences] Code: ```python\n...\n```"

    gpu_memory_utilization = 0.85
    tensor_parallel_size = 1  # Single GPU for vLLM
    hf_device = "cuda:1"  # HF on second GPU
    torch_dtype = torch.float16

    logger = setup_logging(save_dir, f"{args.data_type}{output_suffix}")

    logger.info("="*80)
    logger.info(f"APPS Generation - {args.data_type.upper()}{output_suffix}")
    logger.info("="*80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Data type: {args.data_type}, Run ID: {args.run_id}")
    logger.info(f"vLLM GPU: cuda:0 (physical GPU 0), HF GPU: cuda:1 (physical GPU 1)")
    logger.info(f"Num samples: {num_samples}")
    logger.info("")

    # Load vLLM model
    logger.info("Loading vLLM model...")
    vllm_llm = LLM(
        model=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype="float16"
    )
    logger.info("vLLM model loaded")

    # Load HuggingFace model
    logger.info("Loading HuggingFace model for hidden state extraction...")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=hf_device,
        trust_remote_code=True
    )
    hf_model.eval()
    logger.info(f"HuggingFace model loaded on {hf_device}")

    tokenizer = vllm_llm.get_tokenizer()

    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # Data is a list of questions
    all_questions = data
    for q in all_questions:
        q['level'] = q.get('difficulty', 'unknown')

    logger.info(f"Loaded {len(all_questions)} questions")

    # Resume logic
    save_path = save_dir / f"apps_generation_results_{args.data_type}{output_suffix}.pkl"
    if save_path.exists():
        with open(save_path, 'rb') as f:
            results = pickle.load(f)
        start_idx = len(results)
        logger.info(f"Resumed from checkpoint: {start_idx} questions already processed")
    else:
        results = []
        start_idx = 0

    for idx in tqdm(range(start_idx, len(all_questions)), desc="Generating Part1"):
        item = all_questions[idx]
        question = item.get('question', '')
        ground_truth = item.get('solutions', '')  # APPS uses 'solutions'
        level = item.get('level', 'unknown')
        question_id = item.get('id', idx)

        # Build prompt
        prompt = build_chat_prompt(tokenizer, question, system_prompt)

        # 1. Greedy generation with hidden state extraction
        try:
            greedy_text, greedy_logprob, greedy_hidden, greedy_meta = generate_with_vllm_and_extract_hidden_state(
                vllm_llm=vllm_llm,
                hf_model=hf_model,
                hf_tokenizer=hf_tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=greedy_temperature,
                top_p=greedy_top_p,
                top_k=greedy_top_k,
                extract_hidden_state=True,
                hidden_state_layer=hidden_state_layer,
                device=hf_device
            )
        except Exception as e:
            logger.error(f"Question {idx} failed during greedy generation: {str(e)}")
            continue

        # 2. Sampled generations
        try:
            batch_results = generate_batch_with_vllm(
                vllm_llm=vllm_llm,
                prompt=prompt,
                num_samples=num_samples,
                max_new_tokens=max_new_tokens,
                temperature=sampling_temperature,
                top_p=sampling_top_p,
                top_k=sampling_top_k
            )

            sampled_responses = []
            for sampled_text, sampled_logprob, sampled_meta in batch_results:
                sampled_responses.append({
                    'response': sampled_text,
                    'answer': sampled_meta['answer'],
                    'cum_mean_logprob': sampled_logprob,
                    'cum_sum_logprob': sampled_meta.get('answer_logprob_sum', 0.0),
                    'answer_tokens': sampled_meta['answer_num_tokens'],
                    'token_logprobs': sampled_meta['answer_token_logprobs']
                })
        except Exception as e:
            logger.warning(f"Question {idx} batch sampling failed: {str(e)}")
            sampled_responses = []

        # Store results
        result = {
            'question_id': question_id,
            'level': level,
            'question': question,
            'ground_truth': ground_truth,
            'greedy_response': greedy_text,
            'greedy_answer': greedy_meta['answer'],
            'greedy_cum_mean_logprob': greedy_logprob,
            'greedy_cum_sum_logprob': greedy_meta.get('answer_logprob_sum', 0.0),
            'greedy_answer_tokens': greedy_meta['answer_num_tokens'],
            'greedy_token_logprobs': greedy_meta['answer_token_logprobs'],
            'greedy_hidden_state': greedy_hidden,
            'sampled_responses': sampled_responses
        }

        results.append(result)
        logger.info(f"Completed question {idx+1}/{len(all_questions)} (Level: {level}, ID: {question_id})")

        # Save checkpoint every save_every questions
        if (idx + 1) % save_every == 0:
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"Checkpoint saved: {len(results)} questions processed")

    # Final save
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Saved to: {save_path}")

    logger.info("")
    logger.info("="*80)
    logger.info("PART 1 GENERATION COMPLETE")
    logger.info(f"Total: {len(results)} questions processed")
    logger.info("="*80)


if __name__ == "__main__":
    main()
