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
Compare the outputs of vLLM with and without aclgraph.

Run `pytest tests/compile/test_aclgraph.py`.
"""

import pytest
import torch
from vllm import LLM, SamplingParams

MODELS = ["deepseek-ai/DeepSeek-V2-Lite"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [1])
def test_models(
    model: str,
    max_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    return

    prompts = "The president of the United States is"

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
    )

    vllm_model = LLM(model, long_prefill_token_threshold=4, enforce_eager=True)
    output_chunked = vllm_model.generate(prompts, sampling_params)
    logprobs_chunked = output_chunked.outputs[0].logprobs
    del vllm_model
    torch.npu.empty_cache()

    vllm_model = LLM(model,
                     enforce_eager=True,
                     additional_config={
                         'ascend_scheduler_config': {
                             'enabled': True
                         },
                     })
    output = vllm_model.generate(prompts, sampling_params)
    logprobs = output.outputs[0].logprobs
    del vllm_model
    torch.npu.empty_cache()

    logprobs_similarity = torch.cosine_similarity(logprobs_chunked.flatten(),
                                                  logprobs.flatten(),
                                                  dim=0)
    assert logprobs_similarity > 0.95
