# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import torch
from torch import nn

from vllm_ascend.ops.attn_res import apply_attn_res


@pytest.mark.parametrize(
    ("num_tokens", "num_block_residuals"),
    [
        pytest.param(1, 1, id="decode-min-streams"),
        pytest.param(1, 8, id="decode-max-streams"),
        pytest.param(16, 8, id="aclgraph-padded-decode"),
        pytest.param(32, 8, id="multi-token-max-streams"),
        pytest.param(129, 8, id="prefill-multiple-tokens-per-core"),
    ],
)
@torch.inference_mode()
def test_kimi_k3_attn_res(num_tokens: int, num_block_residuals: int):
    torch.manual_seed(42)
    hidden_size = 7168
    epsilon = 1e-5
    prefix_sum = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="npu")
    block_residual = torch.randn(
        num_tokens,
        num_block_residuals,
        hidden_size,
        dtype=torch.bfloat16,
        device="npu",
    )
    projection = nn.Linear(hidden_size, 1, bias=False, device="npu", dtype=torch.bfloat16)
    norm = nn.Module()
    norm.register_parameter(
        "weight",
        nn.Parameter(torch.randn(hidden_size, dtype=torch.bfloat16, device="npu")),
    )
    norm.variance_epsilon = epsilon

    actual = apply_attn_res(prefix_sum, block_residual, projection, norm)

    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1).float()
    normalized_without_gamma = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon)
    score_weight = norm.weight.float() * projection.weight.squeeze(0).float()
    scores = (normalized_without_gamma * score_weight).sum(-1)
    probabilities = scores.softmax(-1).unsqueeze(1)
    expected = torch.matmul(probabilities, values).squeeze(1).to(prefix_sum.dtype)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
