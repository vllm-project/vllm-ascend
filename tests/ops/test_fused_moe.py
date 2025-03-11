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
# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.

Run `pytest tests/ops/test_moe.py`.
"""
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.activation import SiluAndMul

from vllm_ascend.ops.fused_moe import forward_oot

NUM_EXPERTS = [8, 64]
EP_SIZE = [1]
TOP_KS = [2, 6]
DEVICE = ["npu:0"]


def torch_moe(a, w1, w2, score, topk, renormalize, num_expert_group,
              topk_group, scoring_func, e_score_correction_bias):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    if scoring_func == "softmax":
        score = torch.softmax(score, dim=-1)
    elif scoring_func == "sigmoid":
        score = score.sigmoid()

    if e_score_correction_bias is not None:
        original_scores = score
        score = score + e_score_correction_bias.unsqueeze(0)

    # group_topk
    num_token = score.shape[0]
    group_score = score.view(num_token, num_expert_group,
                             -1).max(dim=-1).values
    group_idx = torch.topk(group_score, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_score)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        score.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    score = score.masked_fill(~score_mask.bool(), 0.0)  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(score, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weight = original_scores.gather(1, topk_ids)
    else:
        topk_weight, topk_ids = torch.topk(score, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(
                a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)


@pytest.mark.parametrize("m", [1])
@pytest.mark.parametrize("n", [128])
@pytest.mark.parametrize("k", [128])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("dtype", [torch.float16])  #, torch.bfloat16])
@pytest.mark.parametrize("device", DEVICE)
def test_fused_moe(m: int, n: int, k: int, e: int, topk: int, ep_size: int,
                   dtype: torch.dtype, device: str):
    a = torch.randn((m, k), device=device, dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device=device, dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device=device, dtype=dtype) / 10

    score = torch.randn((m, e), device=device, dtype=dtype)
    topk_weights, topk_ids = score.topk(topk, dim=-1)
    topk_weights = topk_weights.to(dtype)
    topk_ids = topk_ids.to(torch.int32)

    layer = SimpleNamespace(w13_weight=w1, w2_weight=w2)

    if ep_size > 1:
        local_e = e // ep_size
        e_ids = torch.randint(0,
                              e, (local_e, ),
                              device=device,
                              dtype=torch.int32)
        e_map = torch.full((e, ), -1, device=device, dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device=device, dtype=torch.int32)
        w1 = w1[e_ids]
        w2 = w2[e_ids]
    else:
        e_map = None

    output = forward_oot(None, layer, a, True, topk, topk_weights, True, 1, 1,
                         -1, e_map)

    torch_output = torch_moe(a,
                             w1,
                             w2,
                             score,
                             topk,
                             renormalize=True,
                             num_expert_group=1,
                             topk_group=1,
                             scoring_func='sigmoid',
                             e_score_correction_bias=None)

    torch.testing.assert_close(output, torch_output, atol=2e-2, rtol=0)
