#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""
Execute the inference of fused_moe_allgather_ep and fused_moe_alltoall_ep.

Run 'pytest tests/multicard/test_fused_moe_allgather_ep.py'.
"""

import os
from unittest.mock import patch

import pytest
from modelscope import snapshot_download  # type: ignore
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


@pytest.mark.skipif(
    True,
    reason=
    "Current disaggregated pd implementation may cause memory pulse, which will cause this test OOM, skip this test until the ringmla is ready "
)
@patch.dict(
    os.environ, {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TASK_QUEUE_ENABLE": "1",
        "VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP": "1"
    })
def test_generate_with_allgather():
    example_prompts = ["Hello, my name is"]
    sampling_params = SamplingParams(max_tokens=100, temperature=0.0)

    with VllmRunner(snapshot_download("vllm-ascend/DeepSeek-V3-Pruning"),
                    tensor_parallel_size=2,
                    max_model_len=1024,
                    dtype="auto",
                    enable_expert_parallel=True,
                    additional_config={
                        "ascend_scheduler_config": {
                            "enabled": True,
                            "chunked_prefill_enabled": False,
                        },
                    }) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)


@pytest.mark.skipif(
    True,
    reason=
    "Current disaggregated pd implementation may cause memory pulse, which will cause this test OOM, skip this test until the ringmla is ready "
)
@patch.dict(os.environ, {
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "TASK_QUEUE_ENABLE": "1"
})
def test_generate_with_alltoall():
    example_prompts = ["Hello, my name is"]
    sampling_params = SamplingParams(max_tokens=100, temperature=0.0)

    with VllmRunner(snapshot_download("vllm-ascend/DeepSeek-V3-Pruning"),
                    tensor_parallel_size=2,
                    max_model_len=1024,
                    dtype="auto",
                    enable_expert_parallel=True,
                    additional_config={
                        "ascend_scheduler_config": {
                            "enabled": True,
                            "chunked_prefill_enabled": False,
                        },
                    }) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)
