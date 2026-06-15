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
"""Prefix-cache CP guard.

Run `pytest tests/e2e/pull_request/four_card/long_sequence/test_prefix_caching_cp.py`.
"""

import os
from unittest.mock import patch

from tests.e2e.conftest import _LONG_PROMPTS, VllmRunner

MODEL = "vllm-ascend/DeepSeek-V2-Lite-W8A8"

with open(_LONG_PROMPTS[0], encoding="utf-8") as file:
    LONG_PROMPT = file.read()

INPUT_PROMPTS = [
    LONG_PROMPT + "Question: what is the age of John Doe? Your answer: The age of John Doe is ",
    LONG_PROMPT + "Question: what is the age of Umar Black? Your answer: The age of Umar Black is ",
]

VLLM_OUTPUT = [INPUT_PROMPTS[0] + "29", INPUT_PROMPTS[1] + "39"]


@patch.dict(os.environ, {"HCCL_BUFFSIZE": "768"})
def test_prefix_cache_with_pcp_dcp_full_graph() -> None:
    with VllmRunner(
        MODEL,
        block_size=128,
        max_model_len=4096,
        enforce_eager=False,
        enable_prefix_caching=True,
        enable_expert_parallel=True,
        max_num_batched_tokens=4096,
        tensor_parallel_size=2,
        quantization="ascend",
        prefill_context_parallel_size=2,
        decode_context_parallel_size=2,
        compilation_config={
            "cudagraph_capture_sizes": [4, 8, 24, 48, 60],
            "cudagraph_mode": "FULL_DECODE_ONLY",
        },
    ) as vllm_model:
        prefix_cache_outputs = vllm_model.generate_greedy(INPUT_PROMPTS, 2)

    for i, (_, output_text) in enumerate(prefix_cache_outputs):
        assert output_text == VLLM_OUTPUT[i]
