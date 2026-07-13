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

import os
from pathlib import Path

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

_DEFAULT_CACHE_ROOT = Path(os.getenv("CACHE_ROOT", "/data/.cache"))
_DEFAULT_MODELSCOPE_CACHE = Path(os.getenv("MODELSCOPE_CACHE", str(_DEFAULT_CACHE_ROOT / "modelscope")))
_DEEPSEEK_MODEL_ORG = "deepseek-ai"


def _modelscope_model_path(model_name: str, env_name: str, aliases: tuple[str, ...] = ()) -> str:
    if env_model_path := os.getenv(env_name):
        return env_model_path

    candidate_names = (model_name, *aliases)
    model_path_candidates = [
        _DEFAULT_MODELSCOPE_CACHE / _DEEPSEEK_MODEL_ORG / candidate_name for candidate_name in candidate_names
    ]
    model_path_candidates += [
        _DEFAULT_MODELSCOPE_CACHE / "hub" / "models" / _DEEPSEEK_MODEL_ORG / candidate_name
        for candidate_name in candidate_names
    ]
    model_path_candidates += [
        _DEFAULT_MODELSCOPE_CACHE / "models" / _DEEPSEEK_MODEL_ORG / candidate_name
        for candidate_name in candidate_names
    ]

    for candidate_path in model_path_candidates:
        if candidate_path.exists():
            return str(candidate_path)
    return f"{_DEEPSEEK_MODEL_ORG}/{model_name}"


DISTILL_GRAPH_CASES = [
    pytest.param(
        "DeepSeek-R1-Distill-Qwen-1.5B",
        "QWEN15B_MODEL_PATH",
        ("DeepSeek-R1-Distill-Qwen-1___5B",),
        4,
        [1, 2, 4],
        0.70,
        id="qwen-1.5b",
    ),
    pytest.param(
        "DeepSeek-R1-Distill-Llama-8B",
        "LLAMA8B_MODEL_PATH",
        (),
        2,
        [1, 2],
        0.80,
        id="llama-8b",
    ),
]

PROMPTS = [
    "Reply with one short sentence about graph mode.",
    "Give one concise reason to test real model weights.",
    "Name the backend family used by this dense model.",
    "Summarize why startup-only checks are insufficient.",
]


@wait_until_npu_memory_free(0.7)
@pytest.mark.parametrize(
    ("model_name", "model_path_env", "model_path_aliases", "max_num_seqs", "capture_sizes", "gpu_memory_utilization"),
    DISTILL_GRAPH_CASES,
)
def test_deepseek_r1_distill_full_decode_only_aclgraph_310p(
    model_name: str,
    model_path_env: str,
    model_path_aliases: tuple[str, ...],
    max_num_seqs: int,
    capture_sizes: list[int],
    gpu_memory_utilization: float,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AIV")
    prompts = PROMPTS[:max_num_seqs]

    with VllmRunner(
        _modelscope_model_path(model_name, model_path_env, model_path_aliases),
        dtype="float16",
        max_model_len=4096,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=gpu_memory_utilization,
        additional_config={"ascend_compilation_config": {"fuse_norm_quant": False}},
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": capture_sizes,
        },
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, max_tokens=8)

    assert len(outputs) == len(prompts)
    for _, output_text in outputs:
        assert len(output_text) > 0
