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
import gc

import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [1])
def test_models(
    model: str,
    max_tokens: int,
) -> None:
    prompts = ["The president of the United States is"]

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
    )

    with VllmRunner(model, long_prefill_token_threshold=20,
                    enforce_eager=True) as vllm_model:
        output1 = vllm_model.generate(prompts, sampling_params)

    with VllmRunner(model,
                    enforce_eager=True,
                    additional_config={
                        'ascend_scheduler_config': {
                            'enabled': True
                        },
                    }) as vllm_model:
        output2 = vllm_model.generate(prompts, sampling_params)

    # Extract the generated token IDs for comparison
    token_ids1 = output1[0][0][0]
    token_ids2 = output2[0][0][0]

    print(f"Token IDs 1: {token_ids1}")
    print(f"Token IDs 2: {token_ids2}")

    # Convert token IDs to tensors and calculate cosine similarity
    # Take the length of a shorter sequence to ensure consistent dimensions
    min_len = min(len(token_ids1), len(token_ids2))

    tensor1 = torch.tensor(token_ids1[:min_len], dtype=torch.float32)
    tensor2 = torch.tensor(token_ids2[:min_len], dtype=torch.float32)

    # Calculate similarity using torch.cosine_similarity
    similarity = torch.cosine_similarity(tensor1, tensor2, dim=0)
    print(f"Token IDs cosine similarity: {similarity.item()}")

    assert similarity > 0.95

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
