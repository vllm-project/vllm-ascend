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

"""Numerical regression coverage for Kimi K3 attention residual fusion."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm_ascend.ops.triton.kimi_k3.attention_residual import apply_attn_res


pytestmark = [
    pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available"),
    pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required"),
    pytest.mark.skip_global_cleanup,
]


@torch.inference_mode()
@pytest.mark.parametrize(
    ("num_tokens", "num_blocks", "block_capacity"),
    [
        pytest.param(7, 4, 7, id="partial-capacity"),
        pytest.param(512, 8, 8, id="profile-shape"),
    ],
)
def test_kimi_k3_attention_residual_triton_matches_reference(
    num_tokens,
    num_blocks,
    block_capacity,
):
    torch.manual_seed(1)
    hidden_size = 7168
    eps = 1e-6
    prefix_sum = torch.randn(
        (num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device="npu",
    )
    block_residual = torch.randn(
        (num_tokens, block_capacity, hidden_size),
        dtype=torch.bfloat16,
        device="npu",
    )
    projection = SimpleNamespace(
        weight=torch.randn(
            (1, hidden_size),
            dtype=torch.bfloat16,
            device="npu",
        )
    )
    norm = SimpleNamespace(
        weight=torch.randn(
            (hidden_size,),
            dtype=torch.bfloat16,
            device="npu",
        ),
        variance_epsilon=eps,
    )

    actual = apply_attn_res(
        prefix_sum,
        block_residual,
        projection,
        norm,
        num_blocks,
    )

    values = torch.cat((block_residual[:, :num_blocks, :], prefix_sum.unsqueeze(1)), dim=1).float()
    normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    score_weight = norm.weight.float() * projection.weight.squeeze(0).float()
    scores = (normalized * score_weight).sum(dim=-1)
    probabilities = scores.softmax(-1).unsqueeze(1)
    expected = torch.matmul(probabilities, values).squeeze(1).to(prefix_sum.dtype)

    torch.testing.assert_close(
        actual.cpu(),
        expected.cpu(),
        rtol=1e-2,
        atol=1e-2,
    )
