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

import pytest

from tests.e2e.conftest import VllmRunner


@pytest.mark.e2e_features("310P", "multimodal", "eager_mode", "mtp", "mamba_ssm")
@pytest.mark.e2e_model("Qwen/Qwen3.5-4B")
def test_qwen3_5_mtp_tp1_eager():
    example_prompts = ["Hello, my name is"]
    with VllmRunner(
        "Qwen/Qwen3.5-4B",
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="float16",
        max_model_len=2048,
        mamba_ssm_cache_dtype="float16",
        speculative_config={
            "method": "qwen3_5_mtp",
            "num_speculative_tokens": 1,
        },
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens=8)
