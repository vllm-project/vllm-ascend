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
"""Compare the outputs of vLLM when using W4A8 quantization on qwen3 models.

Run `pytest tests/multicard/test_model_qwen3_w4a8.py`.
"""
import os

import pytest
from modelscope import snapshot_download  # type: ignore
from vllm import LLM, SamplingParams

from vllm_ascend.ascend_config import clear_ascend_config

MODELS = ["vllm-ascend/Qwen3-8B-W4A8"]
PROMPTS = [
    "Hello, my name is",
    "The future of AI is",
]


@pytest.mark.skipif(os.getenv("VLLM_USE_V1") == "0",
                    reason="w4a8_dynamic is not supported on v0")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [16])
def test_qwen3_model_with_w4a8_linear_method(model: str,
                                             max_tokens: int) -> None:
    clear_ascend_config()
    messages = [[{"role": "user", "content": prompt}] for prompt in PROMPTS]
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
    )
    llm = LLM(
        model=snapshot_download(model),
        max_model_len=1024,
        tensor_parallel_size=2,
        enforce_eager=True,
        quantization="ascend",
    )
    vllm_outputs = llm.chat(
        messages,
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},
    )
    golden_outputs = [
        "Hello! My name is Qwen, and I'm a large language model developed",
        "The future of AI is a topic of great interest and debate, with many possibilities",
    ]
    assert len(vllm_outputs) == len(golden_outputs)
    for vllm_output, golden_output in zip(vllm_outputs, golden_outputs):
        assert vllm_output.outputs[0].text == golden_output
        print(f"Generated text: {vllm_output.outputs[0].text!r}")
    clear_ascend_config()
