# import pytest
# import torch
# from vllm_ascend.ops.fused_moe import fused_moe


# def test_fused_moe():
#     # Since we are using native PyTorch operations in the function, the most reliable ground truth
#     # for comparison is the manually computed output. By using hardcoded data, we can ensure
#     # that the function produces the expected results and validate its correctness against a known reference.

#     # Step 1: Constructing inputs
#     hidden_states = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

#     # w1: [3, 4, 3] (num_experts=3, intermediate_size*2=4, hidden_size=3)
#     w1 = torch.tensor(
#         [
#             [[1.0, 0.0, -1.0], [2.0, 1.0, 0.0], [1.0, 1.0, -1.0], [1.0, -1.0, 1.0]],
#             [[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [2.0, -2.0, 2.0], [1.0, 0.0, -1.0]],
#             [[-2.0, -1.0, 1.0], [2.0, -1.0, 1.0], [-1.0, 2.0, 1.0], [1.0, 1.0, -1.0]],
#         ]
#     )

#     # w2: [3, 3, 2] (num_experts=3, hidden_size=3, intermediate_size=2)
#     w2 = torch.tensor(
#         [
#             [[1.0, 0.5], [2.0, -1.0], [0.0, 1.0]],
#             [[1.0, 1.0], [-1.0, 1.0], [1.0, -0.0]],
#             [[-2.0, 1.0], [1.0, -1.0], [2.0, 1.0]],
#         ]
#     )

#     # gating_output: [2, 3] (num_tokens=2, num_experts=3)
#     gating_output = torch.tensor([[0.0, 0.5, 0.5], [0.5, 0.5, 0.0]])

#     topk = 2

#     global_num_experts = 3

#     # Only has the first two experts
#     expert_map = torch.tensor([0, 1, -1])

#     renormalize = False

#     use_grouped_topk = False

#     # Step 2: Expected output calculation

#     # We use topk=2, which means we select the top 2 experts based on gating_output.
#     # For sample 1, gating_output = [0.1, 0.7, 0.2], topk_weights = [0.7, 0.2], selected experts = 1, 2
#     # For sample 2, gating_output = [0.5, 0.4, 0.1], topk_weights = [0.5, 0.4], selected experts = 0, 1

#     # 1. Calculate linear transformation of hidden_states with w1[0] -> F.linear(hidden_states, w1[0])
#     # 2. Apply gating function to get gate values -> F.silu(x[:, :intermediate_size])
#     # 3. Apply second linear transformation with w2[0] -> F.linear(x, w2[0])
#     # 4. Use the topk_weights for each sample and add the weighted outputs of experts 1 and 2

#     expected_hidden_states = torch.tensor([[4.6763, -7.3797, 6.0280], [7.1232, 0.6220, 6.1364]])

#     # Step 3: Running the fused_moe function
#     final_output = fused_moe(
#         hidden_states, w1, w2, gating_output, topk, global_num_experts, expert_map, renormalize, use_grouped_topk
#     )

#     # Step 4: Check the shape and values (this should match the expected result you computed manually)
#     assert (
#         final_output.shape == hidden_states.shape
#     ), f"Expected shape {hidden_states.shape}, but got {final_output.shape}"

#     assert torch.allclose(final_output, expected_hidden_states, atol=1e-4), "Output does not match expected result"

#


# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/kernels/test_moe.py
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
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MOE layers.

Run `pytest tests/ops/test_moe.py`.
"""
import pytest
import torch

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.moe_torch_iterative import fused_moe as iterative_moe
from vllm_ascend.ops.fused_moe import fused_experts, fused_experts_with_ep

NUM_EXPERTS = [8, 64]
EP_SIZE = [1, 4]
TOP_KS = [2, 6]
DEVICE = ["npu"]


def torch_moe(a, w1, w2, score, topk, expert_map):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    if expert_map is not None:
        topk_ids = expert_map[topk_ids]
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)


@pytest.mark.parametrize("m", [1, 33, 64, 222, 1024 * 128])
@pytest.mark.parametrize("n", [128, 1024, 2048])
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", DEVICE)
def test_fused_moe(m: int, n: int, k: int, e: int, topk: int, ep_size: int, dtype: torch.dtype, device: str):
    a = torch.randn((m, k), device=device, dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device=device, dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device=device, dtype=dtype) / 10

    score = torch.randn((m, e), device=device, dtype=dtype)

    if ep_size > 1:
        local_e = e // ep_size
        e_ids = torch.randint(0, e, (local_e,), device=device, dtype=torch.int32)
        e_map = torch.full((e,), -1, device=device, dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device=device, dtype=torch.int32)
        w1 = w1[e_ids]
        w2 = w2[e_ids]

        output = fused_experts_with_ep(
            a, w1, w2, score, topk, global_num_experts=e, expert_map=e_map, renormalize=False
        )
    else:
        fused_moe = fused_experts

    output = fused_moe(a, w1, w2, score, topk, global_num_experts=e, expert_map=e_map, renormalize=False)
    torch_output = torch_moe(a, w1, w2, score, topk, e_map)
    torch.testing.assert_close(output, torch_output, atol=2e-2, rtol=0)
