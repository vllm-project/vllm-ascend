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
# This file is a part of the vllm-ascend project.
#

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.singlecard.utils import PROMPTS_SHORT, compare_logprobs


@wait_until_npu_memory_free()
def test_dense_piecewise_graph():
    """Verify dense generation on the piecewise graph path."""
    runner_kwargs = {
        "model_name": "Qwen/Qwen3-0.6B",
        "max_model_len": 1024,
        "cudagraph_capture_sizes": [1, 2, 4, 8],
    }
    compare_logprobs(runner_kwargs=runner_kwargs, prompts=PROMPTS_SHORT)
