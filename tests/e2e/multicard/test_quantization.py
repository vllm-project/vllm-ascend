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
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
#
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/e2e/multicard/test_quantization.py`.
"""
from modelscope import snapshot_download  # type: ignore

from tests.e2e.conftest import VllmRunner


def test_qwen2_5_w8a8_external_quantized_tp2():
    example_prompts = [
        "The president of the United States is",
    ]
    max_tokens = 5
    with VllmRunner(
            snapshot_download("neuralmagic/Qwen2.5-3B-quantized.w8a8"),
            tensor_parallel_size=2,
            max_model_len=4096,
            gpu_memory_utilization=0.8,
    ) as vllm_model:
        vllm_output = vllm_model.generate_greedy(example_prompts, max_tokens)

    golden_results = [
        'The president of the United States is the head of state and',
    ]

    for i in range(len(vllm_output)):
        assert golden_results[i] == vllm_output[i][1]
        print(f"Generated text: {vllm_output[i][1]!r}")
