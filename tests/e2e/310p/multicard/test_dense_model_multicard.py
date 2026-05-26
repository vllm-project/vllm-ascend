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

from tests.e2e.conftest import VllmRunner


def test_qwen3_dense_tp4_w8a8sc():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
        "Eco-Tech/Qwen3-14B-w8a8sc-310-vllm/TP4/Qwen3-14B-w8a8sc-310-vllm-tp4",
        tensor_parallel_size=4,
        enforce_eager=True,
        dtype="float16",
        quantization="ascend",
        load_format="sharded_state",
        max_model_len=16384,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
