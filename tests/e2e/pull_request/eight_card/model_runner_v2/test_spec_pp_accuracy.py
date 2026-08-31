# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2026 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import regex as re
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

DEEPSEEK_V4_MODEL = "UploadWeight/DeepSeek-V4-Flash-DSpark-w4a8-test"
MINIMAX_M3_MODEL = "Eco-Tech/MiniMax-M3-w8a8-0626"
MINIMAX_M3_DRAFT_MODEL = "Inferact/MiniMax-M3-EAGLE3-GQA"

GSM8K_PROMPT = (
    'Answer the following question. The last line of the response should follow this format: "answer:$ANSWER" '
    "(without quotes), where ANSWER is a number. Let's think step by step.\n\n"
    "Question: Ali had $21. Leila gave him half of her $100. How much does Ali have now?"
)
GSM8K_ANSWER = "71"
ANSWER_RE = re.compile(r"answer\s*:\s*\$?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_answer(text: str) -> str:
    matches = ANSWER_RE.findall(text) or NUMBER_RE.findall(text)
    assert matches, f"No numeric answer found in model output: {text!r}"
    normalized = matches[-1].strip().replace(",", "").rstrip(".")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def _assert_speculative_accuracy(outputs, metrics) -> None:
    assert len(outputs) == 1
    output_ids, output_text = outputs[0]
    assert output_ids and output_text
    assert _extract_answer(output_text) == GSM8K_ANSWER, output_text

    num_drafts = 0
    num_accepted = 0
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            num_accepted += sum(metric.values)

    assert num_drafts > 0, "Speculative decoding did not generate draft tokens"
    assert num_accepted > 0, "Speculative decoding did not accept any draft tokens"


def _configure_jemalloc() -> None:
    jemalloc_path = "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
    if Path(jemalloc_path).exists():
        os.environ["LD_PRELOAD"] = jemalloc_path


@pytest.mark.e2e_model(DEEPSEEK_V4_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="spec_decode,aclgraph",
    parallel="PP,TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_BUFFSIZE": "2048",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v4_dspark_pp_accuracy() -> None:
    with VllmRunner(
        DEEPSEEK_V4_MODEL,
        max_model_len=8192,
        max_num_seqs=8,
        max_num_batched_tokens=4096,
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.9,
        disable_log_stats=False,
        async_scheduling=True,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        speculative_config={
            "method": "dspark",
            "num_speculative_tokens": 5,
            "enforce_eager": True,
        },
        additional_config={"enable_dsa_cp": False},
    ) as runner:
        outputs = runner.generate_greedy([GSM8K_PROMPT], max_tokens=512)
        metrics = runner.model.get_metrics()

    _assert_speculative_accuracy(outputs, metrics)


@pytest.mark.e2e_model(MINIMAX_M3_MODEL, MINIMAX_M3_DRAFT_MODEL)
@pytest.mark.e2e_coverage(
    arch="multimodal",
    feature="eagle3,aclgraph",
    parallel="PP,TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_BUFFSIZE": "2048",
        "HCCL_OP_EXPANSION_MODE": "AIV",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_minimax_m3_eagle3_pp_accuracy() -> None:
    _configure_jemalloc()

    with VllmRunner(
        MINIMAX_M3_MODEL,
        max_model_len=8192,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.95,
        quantization="ascend",
        long_prefill_token_threshold=2048,
        limit_mm_per_prompt={"image": 1},
        disable_log_stats=False,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        speculative_config={
            "model": MINIMAX_M3_DRAFT_MODEL,
            "method": "eagle3",
            "num_speculative_tokens": 3,
            "enforce_eager": True,
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
    ) as runner:
        outputs = runner.generate_greedy([GSM8K_PROMPT], max_tokens=512)
        metrics = runner.model.get_metrics()

    _assert_speculative_accuracy(outputs, metrics)
