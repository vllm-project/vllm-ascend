#
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

from __future__ import annotations

import os
import random
from functools import lru_cache
from typing import Any

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL_PATH = os.environ.get("VLLM_QWEN35_MTP_MODEL", "/home/weights/Qwen3.5-0.8B")
PROMPT_LEN_RANGE = (128, 256)
BATCH_CASES = (
    (1, 5),
    (2, 6),
    (4, 8),
    (8, 10),
)
COMMON_SPECULATIVE_CONFIG = {
    "method": "qwen3_5_mtp",
    "num_speculative_tokens": 3,
    "enforce_eager": True,
}
GRAPH_COMPILATION_CONFIG = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [1, 2, 4, 8],
}
COMMON_RUNNER_KWARGS = {
    "max_model_len": 512,
    "max_num_seqs": 8,
    "max_num_batched_tokens": 4096,
    "gpu_memory_utilization": 0.6,
    "enable_prefix_caching": False,
    "distributed_executor_backend": "mp",
    "async_scheduling": True,
    "seed": 0,
    "speculative_config": COMMON_SPECULATIVE_CONFIG,
}


def ensure_model_available() -> None:
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"requires local model weights at {MODEL_PATH}")


@lru_cache(maxsize=1)
def get_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


@lru_cache(maxsize=1)
def get_prompt_body_ids() -> list[int]:
    tokenizer = get_tokenizer()
    prompt_body = (
        "Compatibility smoke test for Qwen3.5 MTP on Ascend. "
        "Validate async scheduling, graph capture, tensor parallel, "
        "and prefill context parallel with variable sequence lengths. "
    )
    token_ids = tokenizer.encode(prompt_body, add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("failed to build prompt body token ids")
    return token_ids


def build_variable_length_prompts(batch_size: int, seed: int) -> list[list[int]]:
    tokenizer = get_tokenizer()
    prompt_body_ids = get_prompt_body_ids()
    rng = random.Random(seed)
    prompts: list[list[int]] = []

    for prompt_idx in range(batch_size):
        target_len = rng.randint(*PROMPT_LEN_RANGE)
        prefix_ids = tokenizer.encode(
            (
                f"Compatibility sample {seed}-{prompt_idx}. "
                f"Batch size {batch_size}. "
            ),
            add_special_tokens=False,
        )
        prompt_ids = list(prefix_ids)
        while len(prompt_ids) < target_len:
            prompt_ids.extend(prompt_body_ids)
        prompts.append(prompt_ids[:target_len])

    return prompts


def assert_outputs_match_shape(
    prompts: list[list[int]],
    outputs: list[tuple[list[list[int]], list[str]]],
    output_len: int,
) -> None:
    assert len(outputs) == len(prompts)

    for prompt_ids, (sample_ids, sample_texts) in zip(prompts, outputs):
        assert len(sample_ids) == 1
        assert len(sample_texts) == 1
        assert len(sample_ids[0]) == len(prompt_ids) + output_len


def run_qwen35_mtp_smoke_test(mode_name: str, **runner_overrides: Any) -> None:
    ensure_model_available()

    runner_kwargs = dict(COMMON_RUNNER_KWARGS)
    runner_kwargs.update(runner_overrides)

    with VllmRunner(MODEL_PATH, **runner_kwargs) as runner:
        for case_idx, (batch_size, output_len) in enumerate(BATCH_CASES):
            prompts = build_variable_length_prompts(
                batch_size=batch_size,
                seed=sum(ord(ch) for ch in mode_name) + case_idx,
            )
            outputs = runner.generate(
                prompts,
                SamplingParams(
                    temperature=0.0,
                    max_tokens=output_len,
                    min_tokens=output_len,
                ),
            )
            assert_outputs_match_shape(prompts, outputs, output_len)
