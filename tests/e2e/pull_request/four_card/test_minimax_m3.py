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

"""Four-card MiniMax-M3 W8A8 functional smoke test.

Full GPQA and TextVQA accuracy coverage remains in the MiniMax-M3 W8A8
nightly configuration. This pull-request test only checks bounded greedy
generation with TP4 and EP.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import regex as re

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MINIMAX_M3_MODEL_PATH = os.environ.get("MINIMAX_M3_MODEL_PATH", "Eco-Tech/MiniMax-M3-w8a8-0626")
GSM8K_QUESTION = "Ali had $21. Leila gave him half of her $100. How much does Ali have now?"
GSM8K_ANSWER = "71"
MAX_TOKENS = 128

GSM8K_PROMPT_TEMPLATE = (
    'Answer the following question.The last line of the response should follow this format: "answer:$ANSWER" '
    "(without quotes), where ANSWER is a number. Let's think step by step.\n\nQuestion: {question}"
)

ANSWER_RE = re.compile(r"answer\s*:\s*\$?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

os.environ["HCCL_OP_EXPANSION_MODE"] = "AIV"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _extract_predicted_answer(text: str) -> str:
    matches = ANSWER_RE.findall(text)
    if matches:
        # The first formatted answer belongs to the requested question. Some
        # MiniMax-M3 environments continue with unrelated GSM8K examples after
        # answering it, so taking the last match can turn a correct completion
        # into a false failure.
        return _normalize_number(matches[0])

    numbers = NUMBER_RE.findall(text)
    assert numbers, f"No numeric answer found in model output: {text!r}"
    return _normalize_number(numbers[-1])


def _normalize_number(value: str) -> str:
    normalized = value.strip().replace(",", "").rstrip(".")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def _configure_jemalloc() -> None:
    jemalloc_path = "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
    if Path(jemalloc_path).exists():
        ld_preload = os.environ.get("LD_PRELOAD", "")
        os.environ["LD_PRELOAD"] = f"{jemalloc_path}:{ld_preload}" if ld_preload else jemalloc_path


def test_extract_predicted_answer_ignores_follow_up_examples() -> None:
    output = (
        "Ali has $71. answer:71\n\n"
        "---\nQuestion: How many meters did James run?\n"
        "James ran 540 meters. answer:540"
    )

    assert _extract_predicted_answer(output) == GSM8K_ANSWER


@pytest.mark.e2e_model(str(MINIMAX_M3_MODEL_PATH))
@pytest.mark.e2e_coverage(
    arch="multimodal",
    feature="aclgraph",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="full_decode_only",
)
@patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3"})
@wait_until_npu_memory_free()
def test_minimax_m3_w8a8_tp4_gsm8k_one_case() -> None:
    _configure_jemalloc()

    example_prompts = [GSM8K_PROMPT_TEMPLATE.format(question=GSM8K_QUESTION)]
    with VllmRunner(
        MINIMAX_M3_MODEL_PATH,
        max_model_len=8192,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        dtype="auto",
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.95,
        quantization="ascend",
        long_prefill_token_threshold=2048,
        limit_mm_per_prompt={"image": 1},
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
        },
        additional_config={
            "enable_cpu_binding": True,
            "ascend_compilation_config": {
                "enable_static_kernel": True,
                "fuse_norm_quant": False,
            },
            "multistream_overlap_shared_expert": False,
            "weight_nz_mode": 2,
            "enable_shared_expert_dp": True,
        },
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(example_prompts, MAX_TOKENS)

        assert len(outputs) == len(example_prompts)
        output_ids, output_str = outputs[0]
        assert len(output_str) > 0
        assert len(output_ids) > 0
        actual_answer = _extract_predicted_answer(output_str)
        assert actual_answer == GSM8K_ANSWER, (
            f"MiniMax-M3 GSM8K answer mismatch for question={GSM8K_QUESTION!r}: "
            f"expected={GSM8K_ANSWER!r}, actual={actual_answer!r}, output={output_str!r}"
        )
