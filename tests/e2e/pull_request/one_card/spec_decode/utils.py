from __future__ import annotations

import os
from typing import Any

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODELS = {
    "eagle3": {
        "main": "Qwen/Qwen3-8B",
        "spec": "RedHatAI/Qwen3-8B-speculator.eagle3",
    },
}

DRAFT_PARALLEL_MODELS = {
    "draft_parallel": {
        "main": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
        "spec": "amd/PARD-Llama-3.2-1B",
    },
}

DFLASH = {
    "dflash": {
        "main": "Qwen/Qwen3-8B",
        "spec": "z-lab/Qwen3-8B-DFlash-b16",
    }
}

DSPARK = {
    "dspark": {
        "main": "Qwen/Qwen3-8B",
        "spec": "deepseek-ai/dspark_qwen3_8b_block7",
    }
}

BASELINES = {
    "eagle": [0.74, 0.44, 0.29],
    "eagle3": [0.68, 0.40, 0.18],
    "draft_parallel": [0.83, 0.50, 0.33, 0.17, 0.17, 0.17, 0.17, 0.00],
    "dflash": [0.60, 0.50, 0.30, 0.20, 0.20, 0.10, 0.00, 0.00],
    "dspark": [1.0, 0.8, 0.6, 0.6, 0.6, 0.6, 0.6],
}


def eagle_model_name():
    return "vllm-ascend/EAGLE-LLaMA3.1-Instruct-8B"


def eagle3_model_name():
    return "vllm-ascend/EAGLE3-LLaMA3.1-Instruct-8B"


def vl_eagle3_model_name():
    return "MNN/Qwen3-VL-8B-Instruct-Eagle3"


def calculate_acceptance_per_pos(metrics: list[Any], num_speculative_tokens: int, counter_type: Any, vector_type: Any):
    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * num_speculative_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, counter_type)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, vector_type)
            for pos in range(len(metric.values)):
                num_accepted_tokens_per_pos[pos] += metric.values[pos]

    return [num_accepted_tokens / num_drafts for num_accepted_tokens in num_accepted_tokens_per_pos]
