# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
# Run `pytest tests/e2e/multicard/2-cards/spec_decode/test_spec_decode.py`.

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.tokenizers.registry import resolve_tokenizer_args
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODELS = {
    "eagle3": {
        "main": "Qwen/Qwen3-8B",
        "spec": "RedHatAI/Qwen3-8B-speculator.eagle3",
    },
}

P_EAGLE_MODELS = {
    "p-eagle": {
        "main": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "spec": "amazon/Qwen3-Coder-30B-A3B-Instruct-P-EAGLE",
    },
}

# NOTE: golden may change (eagle_proposer only runs in eager mode currently),
# thus please update it if ci fails but you have better acceptance
BASELINES_SP = {
    "eagle3": [0.68, 0.40, 0.18],
    "p-eagle": [0.83, 0.50, 0.33, 0.17, 0.17, 0.17, 0.17, 0.00],
}


@pytest.mark.skip(reason="skip test_eagle3_sp_acceptance")
@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"})
@pytest.mark.parametrize("method", ["eagle3"])
@pytest.mark.parametrize("num_speculative_tokens", [3])
@pytest.mark.parametrize("disable_padded_drafter_batch", [True, False])
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_eagle3_sp_acceptance(
    method: str,
    num_speculative_tokens: int,
    disable_padded_drafter_batch: bool,
    async_scheduling: bool,
):
    if disable_padded_drafter_batch and async_scheduling:
        pytest.skip(
            "skip disable_padded_drafter_batch=True and async_scheduling=True",
        )

    main_model_name = MODELS[method]["main"]
    spec_model_name = MODELS[method]["spec"]

    tokenizer = AutoTokenizer.from_pretrained(
        main_model_name,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        max_tokens=256,
    )

    # sp will only be enabled when query_lens > 1000
    prompts = [
        {
            "role": "user",
            "content": " " * 1000 + "Hello, my name is",
        },
        {
            "role": "user",
            "content": " " * 1000 + "The president of the United States is",
        },
        {
            "role": "user",
            "content": " " * 1000 + "The capital of France is",
        },
        {
            "role": "user",
            "content": " " * 1000 + "The future of AI is",
        },
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [prompt],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    speculative_config = {
        "enforce_eager": True,
        "method": method,
        "num_speculative_tokens": num_speculative_tokens,
        "disable_padded_drafter_batch": disable_padded_drafter_batch,
        "model": spec_model_name,
    }

    compilation_config = CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[12])

    with VllmRunner(
        main_model_name,
        enforce_eager=True,
        max_model_len=8192,
        disable_log_stats=False,
        tensor_parallel_size=2,
        max_num_seqs=256,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.7,
        speculative_config=speculative_config,
        compilation_config=compilation_config,
        async_scheduling=async_scheduling,
    ) as llm:
        _ = llm.generate(prompts, sampling_params)
        metrics = llm.model.get_metrics()

    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * num_speculative_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                num_accepted_tokens_per_pos[pos] += metric.values[pos]

    acceptance_per_pos = [num_accepted_tokens / num_drafts for num_accepted_tokens in num_accepted_tokens_per_pos]
    golden = BASELINES_SP[method]

    match = all(abs(a - b) < 0.06 for a, b in zip(acceptance_per_pos, golden))
    if not match:
        print(f"acceptance_per_pos: {acceptance_per_pos}")
        print(f"golden: {golden}")

    assert match


def test_qwen3_eagle3_pcp2_tp1():
    """
    Test Qwen3-8B with Eagle3 speculative decoding under PCP + TP1 configuration.
    This test verifies that eagle3 spec decode works correctly with:
    - PCP enabled (prefill_context_parallel_size=2)
    - Tensor Parallel size = 1
    - num_speculative_tokens = 3
    - enforce_eager = True
    """
    method = "eagle3"
    num_speculative_tokens = 3

    main_model_name = MODELS[method]["main"]
    spec_model_name = MODELS[method]["spec"]

    tokenizer = AutoTokenizer.from_pretrained(
        main_model_name,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        max_tokens=256,
    )

    prompts = [
        {
            "role": "user",
            "content": "Hello, my name is",
        },
        {
            "role": "user",
            "content": "The president of the United States is",
        },
        {
            "role": "user",
            "content": "The capital of France is",
        },
        {
            "role": "user",
            "content": "The future of AI is",
        },
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [prompt],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    speculative_config = {
        "method": method,
        "num_speculative_tokens": num_speculative_tokens,
        "model": spec_model_name,
    }

    with VllmRunner(
        main_model_name,
        enforce_eager=True,
        max_model_len=2048,
        disable_log_stats=False,
        tensor_parallel_size=1,
        prefill_context_parallel_size=2,
        max_num_seqs=256,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.7,
        speculative_config=speculative_config,
    ) as llm:
        llm.generate(prompts, sampling_params)


@pytest.mark.parametrize("method", P_EAGLE_MODELS.keys())
@pytest.mark.parametrize("num_speculative_tokens", [8])
@pytest.mark.parametrize("draft_tensor_parallel_size", [None, 2])
def test_p_eagle_acceptance(
    method: str,
    num_speculative_tokens: int,
    draft_tensor_parallel_size: None | int,
):
    """
    Test acceptance rate for parallel drafting speculative decoding
    using a smaller draft model with parallel_drafting enabled.
    """
    main_model_name = P_EAGLE_MODELS[method]["main"]
    spec_model_name = P_EAGLE_MODELS[method]["spec"]

    tokenizer_path = resolve_tokenizer_args(main_model_name)[1]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        max_tokens=256,
    )

    prompts = [
        {
            "role": "user",
            "content": "Hello, your name is",
        },
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [prompt],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    speculative_config = {
        "method": "eagle3",
        "model": spec_model_name,
        "num_speculative_tokens": num_speculative_tokens,
        "draft_tensor_parallel_size": draft_tensor_parallel_size,
        "parallel_drafting": True,
    }

    compilation_config = CompilationConfig(cudagraph_capture_sizes=[12])

    with VllmRunner(
        main_model_name,
        max_model_len=4096,
        disable_log_stats=False,
        tensor_parallel_size=2,
        max_num_seqs=256,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        speculative_config=speculative_config,
        compilation_config=compilation_config,
        enable_prefix_caching=False,
    ) as llm:
        outputs = llm.model.generate(prompts, sampling_params)
        metrics = llm.model.get_metrics()

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        output_tokens = output.outputs[0].token_ids
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        print(f"Output tokens: {output_tokens}")

    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * num_speculative_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                num_accepted_tokens_per_pos[pos] += metric.values[pos]

    acceptance_per_pos = [num_accepted_tokens / num_drafts for num_accepted_tokens in num_accepted_tokens_per_pos]

    golden = BASELINES_SP[method]

    match = all(abs(a - b) < 0.2 for a, b in zip(acceptance_per_pos, golden))
    if not match:
        print(f"acceptance_per_pos: {acceptance_per_pos}")
        print(f"golden: {golden}")

    assert match
