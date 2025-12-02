
"""
Hybrid generation utilities: vLLM for speed + HuggingFace for hidden states.

"""

import re
import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from vllm import SamplingParams


def extract_reasoning_and_answer(text: str) -> Tuple[str, str, int]:
    """
    Extract reasoning (inside <think> tags) and final answer from generated text.
    """
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, text, re.DOTALL)

    if think_match:
        reasoning = think_match.group(1).strip()
        answer_start_pos = think_match.end()
        answer = text[answer_start_pos:].strip()
    else:
        reasoning = ""
        answer = text.strip()
        answer_start_pos = 0

    return reasoning, answer, answer_start_pos


def find_answer_token_range(
    tokenizer,
    generated_ids: List[int],
    generated_text: str,
    answer_start_pos: int
) -> Tuple[int, int]:
    """
    Find the token range that corresponds to the answer part.
    """
    if answer_start_pos == 0:
        return 0, len(generated_ids)

    answer_start_token_idx = None

    for i in range(len(generated_ids)):
        partial_ids = generated_ids[:i+1]
        partial_text = tokenizer.decode(partial_ids, skip_special_tokens=True)
        curr_text_len = len(partial_text)

        if answer_start_token_idx is None:
            if curr_text_len > answer_start_pos:
                answer_start_token_idx = i
                break

    if answer_start_token_idx is None:
        answer_start_token_idx = 0

    answer_end_token_idx = len(generated_ids)

    return answer_start_token_idx, answer_end_token_idx


def generate_with_vllm_and_extract_hidden_state(
    vllm_llm,
    hf_model,
    hf_tokenizer,
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    extract_hidden_state: bool = False,
    hidden_state_layer: int = -1,
    device: str = "cuda:1"
) -> Tuple[str, float, Optional[np.ndarray], Dict[str, Any]]:
    """Generate with vLLM, optionally extract hidden state with HuggingFace."""
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        logprobs=1,
    )

    outputs = vllm_llm.generate([prompt], sampling_params)
    output = outputs[0]

    generated_text = output.outputs[0].text
    token_ids = output.outputs[0].token_ids
    logprobs_data = output.outputs[0].logprobs

    # Check truncation
    is_truncated_thinking = False
    if '<think>' in generated_text and '</think>' not in generated_text:
        is_truncated_thinking = True

    # Extract reasoning and answer
    reasoning, answer, answer_start_pos = extract_reasoning_and_answer(generated_text)

    # Extract log probabilities for each token
    token_logprobs = []
    for token_logprob_dict in logprobs_data:
        if token_logprob_dict is not None and len(token_logprob_dict) > 0:
            # Get the logprob for the token that was actually generated
            # In vLLM, the dict contains logprobs for top tokens
            # We need the one for the selected token
            max_logprob = max(token_logprob_dict.values(), key=lambda x: x.logprob)
            token_logprobs.append(max_logprob.logprob)
        else:
            token_logprobs.append(0.0)

    # Find answer token range
    vllm_tokenizer = vllm_llm.get_tokenizer()
    answer_start_token_idx, answer_end_token_idx = find_answer_token_range(
        vllm_tokenizer, token_ids, generated_text, answer_start_pos
    )

    answer_token_logprobs = token_logprobs[answer_start_token_idx:answer_end_token_idx]

    # Calculate MEAN of log probabilities (length-normalized)
    if answer_token_logprobs and len(answer_token_logprobs) > 0:
        answer_logprob_mean = sum(answer_token_logprobs) / len(answer_token_logprobs)
        answer_logprob_sum = sum(answer_token_logprobs)
    else:
        answer_logprob_mean = 0.0
        answer_logprob_sum = 0.0

    # Step 2: Extract hidden state with HuggingFace (if needed)
    hidden_state = None
    if extract_hidden_state and hf_model is not None and hf_tokenizer is not None:
        # Tokenize the full sequence (prompt + generated text)
        full_text = prompt + generated_text
        inputs = hf_tokenizer(full_text, return_tensors="pt")

        # Move to HF device
        input_ids = inputs.input_ids.to(device)

        # Get prompt length
        prompt_inputs = hf_tokenizer(prompt, return_tensors="pt")
        prompt_length = prompt_inputs.input_ids.shape[1]

        # Forward pass to extract hidden states
        with torch.no_grad():
            hidden_outputs = hf_model(
                input_ids,
                output_hidden_states=True,
                return_dict=True
            )

        # Extract hidden state from ONLY the last layer (layer -1)
        # hidden_outputs.hidden_states is a tuple with length num_layers + 1
        num_layers = len(hidden_outputs.hidden_states)

        # Use the specified layer (default -1 for last layer)
        if hidden_state_layer < 0:
            target_layer_idx = num_layers + hidden_state_layer
        else:
            target_layer_idx = hidden_state_layer

        # Calculate SLT position for answer
        answer_length = answer_end_token_idx - answer_start_token_idx

        if answer_length >= 2:
            # SLT of answer = second-to-last token of answer
            # Position in full sequence = prompt_length + answer_start_token_idx + (answer_length - 2)
            slt_idx = prompt_length + answer_start_token_idx + (answer_length - 2)
        elif answer_length == 1:
            # Only one token in answer
            slt_idx = prompt_length + answer_start_token_idx
        else:
            # No answer tokens, use last token
            slt_idx = -1

        # Extract hidden state from the target layer at SLT position
        layer_hidden = hidden_outputs.hidden_states[target_layer_idx]

        # Make sure index is within bounds
        if slt_idx >= 0 and slt_idx < layer_hidden.shape[1]:
            hidden_state = layer_hidden[0, slt_idx, :].cpu().numpy()
        else:
            # Fallback: use last token
            hidden_state = layer_hidden[0, -1, :].cpu().numpy()

    # Metadata
    metadata = {
        'full_text': generated_text,
        'reasoning': reasoning,
        'answer': answer,
        'has_reasoning': bool(reasoning),
        'is_truncated_thinking': is_truncated_thinking,
        'answer_start_char_pos': answer_start_pos,
        'answer_start_token_idx': answer_start_token_idx,
        'answer_end_token_idx': answer_end_token_idx,
        'answer_num_tokens': answer_end_token_idx - answer_start_token_idx,
        'total_num_tokens': len(token_ids),
        'answer_logprob': answer_logprob_mean,
        'answer_logprob_sum': answer_logprob_sum,
        'answer_avg_logprob_per_token': answer_logprob_mean / (answer_end_token_idx - answer_start_token_idx) if (answer_end_token_idx - answer_start_token_idx) > 0 else 0.0,
        'answer_token_logprobs': answer_token_logprobs
    }

    return generated_text, answer_logprob_mean, hidden_state, metadata


def generate_batch_with_vllm(
    vllm_llm,
    prompt: str,
    num_samples: int = 10,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """Generate multiple samples in parallel using vLLM's n parameter."""
    # Use vLLM's n parameter to generate multiple samples at once
    sampling_params = SamplingParams(
        n=num_samples,  # Generate n samples in parallel
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        logprobs=1,
    )

    outputs = vllm_llm.generate([prompt], sampling_params)
    output = outputs[0]  # Single prompt, but multiple outputs

    results = []
    vllm_tokenizer = vllm_llm.get_tokenizer()

    # Process each of the n samples
    for sample_output in output.outputs:
        generated_text = sample_output.text
        token_ids = sample_output.token_ids
        logprobs_data = sample_output.logprobs

        # Check truncation
        is_truncated_thinking = False
        if '<think>' in generated_text and '</think>' not in generated_text:
            is_truncated_thinking = True

        # Extract reasoning and answer
        reasoning, answer, answer_start_pos = extract_reasoning_and_answer(generated_text)

        # Extract log probabilities
        token_logprobs = []
        for token_logprob_dict in logprobs_data:
            if token_logprob_dict is not None and len(token_logprob_dict) > 0:
                max_logprob = max(token_logprob_dict.values(), key=lambda x: x.logprob)
                token_logprobs.append(max_logprob.logprob)
            else:
                token_logprobs.append(0.0)

        # Find answer token range
        answer_start_token_idx, answer_end_token_idx = find_answer_token_range(
            vllm_tokenizer, token_ids, generated_text, answer_start_pos
        )

        # Extract answer log probs
        answer_token_logprobs = token_logprobs[answer_start_token_idx:answer_end_token_idx]

        # Calculate MEAN and SUM
        if answer_token_logprobs and len(answer_token_logprobs) > 0:
            answer_logprob_mean = sum(answer_token_logprobs) / len(answer_token_logprobs)
            answer_logprob_sum = sum(answer_token_logprobs)
        else:
            answer_logprob_mean = 0.0
            answer_logprob_sum = 0.0

        # Metadata
        metadata = {
            'full_text': generated_text,
            'reasoning': reasoning,
            'answer': answer,
            'has_reasoning': bool(reasoning),
            'is_truncated_thinking': is_truncated_thinking,
            'answer_start_char_pos': answer_start_pos,
            'answer_start_token_idx': answer_start_token_idx,
            'answer_end_token_idx': answer_end_token_idx,
            'answer_num_tokens': answer_end_token_idx - answer_start_token_idx,
            'total_num_tokens': len(token_ids),
            'answer_logprob': answer_logprob_mean,
            'answer_logprob_sum': answer_logprob_sum,
            'answer_avg_logprob_per_token': answer_logprob_mean / (answer_end_token_idx - answer_start_token_idx) if (answer_end_token_idx - answer_start_token_idx) > 0 else 0.0,
            'answer_token_logprobs': answer_token_logprobs
        }

        results.append((generated_text, answer_logprob_mean, metadata))

    return results


def build_chat_prompt(
    tokenizer,
    question: str,
    system_prompt: str = "Think step-by-step to solve the problem. Show your complete reasoning process, then provide your final answer."
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt
