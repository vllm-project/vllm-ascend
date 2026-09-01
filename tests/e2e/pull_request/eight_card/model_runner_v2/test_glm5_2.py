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

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = "Eco-Tech/GLM-5.2-w4a8"


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp",
    parallel="TP,EP",
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
    },
)
@wait_until_npu_memory_free()
def test_glm5_2_mtp_full_decode_only() -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    num_speculative_tokens = 3
    sampling_params = SamplingParams(max_tokens=1024, temperature=0.0)

    with VllmRunner(
        MODEL,
        quantization="ascend",
        tensor_parallel_size=8,
        max_model_len=8192,
        max_num_seqs=16,
        enable_expert_parallel=True,
        disable_log_stats=False,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": num_speculative_tokens,
        },
    ) as runner:
        outputs = runner.model.generate(prompts, sampling_params)
        metrics = runner.model.get_metrics()

    assert len(outputs) == len(prompts)
    assert all(output.outputs[0].token_ids for output in outputs)

    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * num_speculative_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            assert len(metric.values) == num_speculative_tokens
            for pos, value in enumerate(metric.values):
                num_accepted_tokens_per_pos[pos] += value

    assert num_drafts > 0, "Speculative decoding did not generate any draft tokens"
    acceptance_per_pos = [accepted / num_drafts for accepted in num_accepted_tokens_per_pos]
    assert any(acceptance_per_pos)
    assert all(0 <= acceptance <= 1 for acceptance in acceptance_per_pos)


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="sfa_pcp",
    parallel="TP,EP,PCP",
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
    },
)
@wait_until_npu_memory_free()
def test_glm5_2_sfa_pcp_full_decode_only() -> None:
    """Exercise MRV2 SFA PCP prefill and full-decode-only graph replay without C8 SFA."""
    long_prompt = (
        "You are validating a distributed language-model runtime. Explain how "
        "prefill, KV-cache reuse, decode graph replay, and attention outputs "
        "work together when serving a request with a long context. "
    ) * 4
    prompts = [f"{long_prompt} Request identifier: {request_id}." for request_id in range(4)]
    sampling_params = SamplingParams(max_tokens=2, temperature=0.0)

    with VllmRunner(
        MODEL,
        quantization="ascend",
        tensor_parallel_size=4,
        prefill_context_parallel_size=2,
        max_model_len=8192,
        max_num_seqs=16,
        max_num_batched_tokens=1024,
        enable_expert_parallel=True,
        disable_log_stats=False,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
    ) as runner:
        outputs = runner.model.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts)
    assert all(output.outputs[0].token_ids for output in outputs)
