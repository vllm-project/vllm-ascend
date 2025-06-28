#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

import os
from unittest.mock import patch

import pytest
from modelscope import snapshot_download  # type: ignore

from tests.conftest import VllmRunner


model_name=snapshot_download("vllm-ascend/DeepSeek-R1-w4a8-pruning")


@pytest.mark.skipif(os.getenv("VLLM_USE_V1") == "0",
                    reason="w4a8_dynamic is not supported on v0")
@patch.dict(os.environ, {"VLLM_USE_V1": "1", "VLLM_ASCEND_MLA_PA": "1"})
def test_deepseek_W4A8(model: str):
    prompts = [
        "The capital of France is",
        "The future of AI is",
    ]
    dtype = "bfloat16"
    max_tokens = 5
    with VllmRunner(
            model_name,
            dtype=dtype,
            tensor_parallel_size=2,
            quantization="ascend",
            enforce_eager=True,
            enable_expert_parallel=True,
            additional_config={
                "torchair_graph_config": {
                    "enabled": False,
                },
                "ascend_scheduler_config": {
                    "enabled": True,
                }
            },
    ) as vllm_model:
        # use greedy sampler to make sure the generated results are fix
        vllm_output = vllm_model.generate_greedy(prompts, max_tokens)

    golden_results = [
        'The capital of France is逸 Ban Corporealistically',
        'The future of AI is逸 Ban Corporealistically',
    ]
    assert len(golden_results) == len(vllm_output)
    for i in range(len(vllm_output)):
        assert golden_results[i] == vllm_output[i][1]
        print(f"Generated text: {vllm_output[i][1]!r}")
