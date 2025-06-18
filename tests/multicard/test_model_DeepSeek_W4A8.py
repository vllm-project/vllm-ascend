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

import pytest

from vllm import LLM, SamplingParams

MODELS = ["vllm-ascend/DeepSeek-R1-w4a8-pruning"]

prompts = [
    "Hello, my name is",
    "The future of AI is",
]


@pytest.mark.skipif(os.getenv("VLLM_USE_V1") == "0",
                    reason="w4a8_dynamic is not supported on v0")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [100])
def test_model_w4a8(
    model: str,
    max_tokens: int,
) -> None:
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
    )

    llm = LLM(model=model,
              tensor_parallel_size=2,
              enforce_eager=True,
              trust_remote_code=True,
              max_model_len=1024,
              quantization="ascend",
              additional_config={
                  'expert_tensor_parallel_size': 1,
                  'enable_graph_mode': False,
                  'ascend_scheduler_config': {},
              })
    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")