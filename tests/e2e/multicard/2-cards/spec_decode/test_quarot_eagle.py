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
# Run `pytest tests/e2e/multicard/2-cards/spec_decode/test_quarot_eagle.py`.

from __future__ import annotations

import os
from typing import Any

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

K = 4  # Number of speculative tokens
TOL = 0.06  # Absolute tolerance for acceptance comparison


# Here, the two selected models correspond to two scenarios.
# The 32B draft model comes with its own embedding,
# while the 30B draft model shares the embedding of the target model.
MODELS: dict[str, dict] = {
    "32B": {
        "target": {
            "float": "Qwen/Qwen3-32B",
            "w8a8": "vllm-ascend/Qwen3-32B-W8A8-QuaRot",
        },
        "draft": "RedHatAI/Qwen3-32B-speculator.eagle3",
    },
    "30B": {
        "target": {
            "float": "Qwen/Qwen3-30B-A3B",
            "w8a8": "vllm-ascend/Qwen3-30B-A3B-W8A8-QuaRot",
        },
        "draft": "AngelSlim/Qwen3-a3B_eagle3",
    },
}


def _build_prompts(target_model: str) -> list[str]:
    # These prompts were formed by taking one from each category of mt-bench.
    # Although there are still some differences from the processing method of
    # vllm serve bench, it does not affect this test.
    # it is possible to directly take from mt-bench or further
    # call vllm bench serve for direct testing later.
    prompts = [
        {
            "role": "user",
            "content": "Compose an engaging travel blog post about a recent trip to Hawaii, "
            "highlighting cultural experiences and must-see attractions.",
        },
        {
            "role": "user",
            "content": "Pretend yourself to be Elon Musk in all the following conversations. "
            "Speak like Elon Musk as much as possible. Why do we need to go to Mars?",
        },
        {
            "role": "user",
            "content": "Imagine you are participating in a race with a group of people. "
            "If you have just overtaken the second person, what's your current position? "
            "Where is the person you just overtook?",
        },
        {
            "role": "user",
            "content": "The vertices of a triangle are at points (0, 0), (-1, 1), and (3, 3). "
            "What is the area of the triangle?",
        },
        {
            "role": "user",
            "content": "Develop a Python program that reads all the text files under a directory "
            "and returns top-5 words with the most number of occurrences.",
        },
        {
            "role": "user",
            "content": "Evaluate the following movie reviews on a scale of 1 to 5, with 1 being very negative, "
            "3 being neutral, and 5 being very positive:\n1. This movie released on Nov. 18, 2019, was phenomenal. "
            "The cinematography, the acting, the plot - everything was top-notch.\n"
            "2. Never before have I been so disappointed with a movie. The plot was predictable and the characters "
            "were one-dimensional. In my opinion, this movie is the worst one to have been released in 2022.\n"
            "3. The movie was okay. There were some parts I  enjoyed, but there were also parts that felt lackluster. "
            "This is a movie that was released in Feb 2018 and seems to be quite ordinary.\n"
            "Return the answer as a JSON array of integers.",
        },
        {
            "role": "user",
            "content": "In the field of quantum physics, what is superposition, "
            "and how does it relate to the phenomenon of quantum entanglement?",
        },
        {
            "role": "user",
            "content": "Provide insights into the correlation between economic indicators such as GDP, "
            "inflation, and unemployment rates. Explain how fiscal and monetary policies affect those indicators.",
        },
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        target_model,
        trust_remote_code=True,
    )

    prompts_with_template: list[str] = [
        tokenizer.apply_chat_template(
            [prompt],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    return prompts_with_template


def _run_model(
    llm_kwargs: dict,
    prompts: list[str],
    sampling_params: SamplingParams,
) -> list[Any]:
    with VllmRunner(**llm_kwargs) as llm:
        _ = llm.generate(prompts, sampling_params)
        metrics = llm.model.get_metrics()

    return metrics


def _compute_acceptance(metrics: list[Any]) -> list[float | int]:
    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * K

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value

        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for i, v in enumerate(metric.values):
                num_accepted_tokens_per_pos[i] += v

    acceptance_per_pos = [
        num_accepted_tokens / num_drafts if num_drafts > 0 else 0.0
        for num_accepted_tokens in num_accepted_tokens_per_pos
    ]

    return acceptance_per_pos


@pytest.mark.parametrize("model", ["32B", "30B"])
def test_quarot_eagle_acceptance_tp2(model: str):
    target_model = MODELS[model]["target"]["float"]
    draft_model = MODELS[model]["draft"]

    prompts = _build_prompts(target_model)

    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        max_tokens=512,
    )

    llm_kwargs = dict(
        model_name=target_model,
        enforce_eager=True,
        max_model_len=4096,
        disable_log_stats=False,
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.9,
        speculative_config={
            "enforce_eager": True,
            "method": "eagle3",
            "model": draft_model,
            "num_speculative_tokens": K,
        },
    )

    # Run the float model and the quarot model,
    # and then compare their acceptance rates at each position.
    ref_metrics = _run_model(llm_kwargs, prompts, sampling_params)
    ref_acceptance = _compute_acceptance(ref_metrics)

    llm_kwargs["model_name"] = MODELS[model]["target"]["w8a8"]
    llm_kwargs["quantization"] = "ascend"

    quarot_metrics = _run_model(llm_kwargs, prompts, sampling_params)
    quarot_acceptance = _compute_acceptance(quarot_metrics)

    match = all(abs(i - j) <= TOL for i, j in zip(ref_acceptance, quarot_acceptance))

    assert match, (
        f"\nref_acceptance_per_pos: {[round(_, 4) for _ in ref_acceptance]}"
        f"\nquarot_acceptance_per_pos: {[round(_, 4) for _ in quarot_acceptance]}"
    )
