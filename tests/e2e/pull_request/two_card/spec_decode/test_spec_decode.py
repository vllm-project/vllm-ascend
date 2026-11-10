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
# Run `pytest tests/e2e/pull_request/two_card/spec_decode/test_spec_decode.py`.
#
# NOTE: the heavyweight spec-decode cases (dspark / MoE-MTP hang / VWN /
# sliding-window) live in test_spec_decode_heavy.py so the two file-level
# targets can run in parallel on separate 2-card jobs.

from __future__ import annotations

import os

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import CompilationConfig, CompilationMode
from vllm.inputs import TokensPrompt
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODELS = {
    "eagle3": {
        "main": "Qwen/Qwen3-8B",
        "spec": "RedHatAI/Qwen3-8B-speculator.eagle3",
    },
}

DFLASH2_MODELS = {
    "dflash2": {
        "main": "Qwen/Qwen3.8-27B",
        "spec": "z-lab/Qwen3.8-27B-DFlash2",
    },
}

# NOTE: golden may change (eagle_proposer only runs in eager mode currently),
# thus please update it if ci fails but you have better acceptance

BASELINES_SP = {
    "eagle3": [0.68, 0.40, 0.18],
    "dflash2": [0.855, 0.593, 0.359, 0.290, 0.193, 0.152, 0.076, 0.007],
}


@pytest.mark.skip(reason="skip test_eagle3_sp_acceptance")
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
    assert match, f"acceptance_per_pos {acceptance_per_pos} does not match golden {golden}"


@pytest.mark.parametrize("method", DFLASH2_MODELS.keys())
@pytest.mark.parametrize("num_speculative_tokens", [7])
def test_dflash2_v2_acceptance(
    method: str,
    num_speculative_tokens: int,
):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    main_model_name = DFLASH2_MODELS[method]["main"]
    spec_model_name = DFLASH2_MODELS[method]["spec"]

    tokenizer = AutoTokenizer.from_pretrained(
        main_model_name,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        max_tokens=512,
        seed=42,
    )

    prompts = [
        {
            "role": "user",
            "content": "What is your name, please provide a detailed introduction.",
        },
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [prompt],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]

    speculative_config = {
        "method": "dflash",
        "model": spec_model_name,
        "num_speculative_tokens": num_speculative_tokens,
        "enforce_eager": False,
    }

    # Keep decode within the FULL graph buckets; vary only the prefill length
    # below to exercise PIECEWISE and eager under the same engine configuration.
    decode_query_len = num_speculative_tokens + 1
    capture_sizes = [batch_size * decode_query_len for batch_size in (1, 2)]
    compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode="FULL_AND_PIECEWISE",
        cudagraph_capture_sizes=capture_sizes,
    )

    with VllmRunner(
        main_model_name,
        max_model_len=2048,
        disable_log_stats=False,
        tensor_parallel_size=2,
        max_num_seqs=2,
        max_num_batched_tokens=2048,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.9,
        speculative_config=speculative_config,
        enforce_eager=False,
        compilation_config=compilation_config,
        enable_prefix_caching=False,
    ) as llm:
        assert llm.model.llm_engine.vllm_config.use_v2_model_runner
        outputs = llm.model.generate(prompts, sampling_params)

        # Reuse this engine for two sequential single-request generations,
        # then submit both prompts together to exercise request-slot reuse.
        # K=7 gives a 9-token PIECEWISE prefill and a 32-token eager prefill,
        # followed by 8-token FULL decode in each single-request generation.
        # Token IDs fix the lengths, and the token budget admits each prefill
        # in one step. The joint submission also exercises batched execution.
        prefill_lengths = (decode_query_len + 1, 2 * max(capture_sizes))
        prompt_token_ids = tokenizer.encode("Continue counting: 1, 2, 3, 4, 5.", add_special_tokens=False)
        assert prompt_token_ids
        prompt_token_ids *= (max(prefill_lengths) + len(prompt_token_ids) - 1) // len(prompt_token_ids)
        graph_sampling_params = SamplingParams(temperature=0, ignore_eos=True, max_tokens=48, seed=42)
        graph_prompts: list[TokensPrompt] = [
            {"prompt_token_ids": prompt_token_ids[:prefill_len]} for prefill_len in prefill_lengths
        ]
        # generate() completes each batch before the next one is submitted.
        for graph_batch in [[prompt] for prompt in graph_prompts] + [graph_prompts]:
            graph_outputs = llm.model.generate(graph_batch, graph_sampling_params)
            assert len(graph_outputs) == len(graph_batch)
            for output in graph_outputs:
                assert output.finished
                assert len(output.outputs[0].token_ids) == graph_sampling_params.max_tokens
            outputs.extend(graph_outputs)

        # Aggregate acceptance over the original request and all graph batches.
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
    golden = BASELINES_SP[method][:num_speculative_tokens]

    match = all(abs(a - b) < 0.2 for a, b in zip(acceptance_per_pos, golden, strict=True))
    assert match, f"acceptance_per_pos {acceptance_per_pos} does not match golden {golden}"
