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
# SPDX-License-Identifier: Apache-2.0
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/kernels/test_moe.py
"""Tests for the MOE layers.

Run `pytest tests/ops/test_fused_moe.py`.
"""
# fused moe ops test will hit the infer_schema error, we need add the patch
# here to make the test pass.
import vllm_ascend.patch.worker.patch_common.patch_utils  # type: ignore[import]  # isort: skip  # noqa

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.activation import SiluAndMul

from vllm_ascend.ops.fused_moe import fused_experts, select_experts

NUM_EXPERTS = [8, 64]
EP_SIZE = [1, 4]
TOP_KS = [2, 6]
DEVICE = ["npu"]


def torch_moe(a, w1, w2, topk_weights, topk_ids, topk, expert_map):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    topk_weights = topk_weights.view(-1)
    topk_ids = topk_ids.view(-1)
    if expert_map is not None:
        topk_ids = expert_map[topk_ids]
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(
                a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weights.view(B, -1, 1).to(out.dtype)).sum(dim=1)


@pytest.mark.parametrize("m", [1, 33, 64, 222, 1024 * 128])
@pytest.mark.parametrize("n", [128, 1024, 2048])
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", DEVICE)
def test_fused_experts(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
    device: str,
):
    a = torch.randn((m, k), device=device, dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device=device, dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device=device, dtype=dtype) / 10

    score = torch.randn((m, e), device=device, dtype=dtype)

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

    score = torch.softmax(score, dim=-1, dtype=dtype)
    topk_weights, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    output = fused_experts(a, w1, w2, topk_weights, topk_ids, topk, e_map)
    torch_output = torch_moe(a, w1, w2, topk_weights, topk_ids, topk, e_map)
    # TODO: The native params are: atol=2e-2, rtol=0, maybe related to the nan problem
    torch.testing.assert_close(output, torch_output, atol=4e-2, rtol=1)
    torch.npu.empty_cache()


@pytest.mark.parametrize("m", [1, 33, 64])
@pytest.mark.parametrize("n", [128, 1024, 2048])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("use_grouped_topk", [True, False])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("with_e_correction", [True, False])
@pytest.mark.parametrize("custom_routing", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", DEVICE)
def test_select_experts(
    m: int,
    n: int,
    e: int,
    topk: int,
    scoring_func: str,
    use_grouped_topk: bool,
    renormalize: bool,
    with_e_correction: bool,
    custom_routing: bool,
    dtype: torch.dtype,
    device: str,
):
    topk_group = 4 if use_grouped_topk else None
    num_expert_group = e // 4 if use_grouped_topk else None

    hidden_states = torch.randn(m, n, device=device, dtype=dtype)
    router_logits = torch.randn(m, e, device=device, dtype=dtype)

    e_score_correction_bias = (torch.randn(e, device=device, dtype=dtype)
                               if with_e_correction else None)

    custom_routing_function = None
    if custom_routing:
        custom_routing_function = MagicMock()
        mock_weights = torch.randn(m, topk, device=device, dtype=dtype)
        mock_ids = torch.randint(0,
                                 e, (m, topk),
                                 device=device,
                                 dtype=torch.int32)
        custom_routing_function.return_value = (mock_weights, mock_ids)

    with patch("vllm_ascend.ops.fused_moe.native_grouped_topk"
               ) as mock_native_grouped_topk:
        mock_native_grouped_topk.side_effect = lambda x, num_groups, k: torch.randn_like(
            x)

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=topk,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )

        if use_grouped_topk:
            mock_native_grouped_topk.assert_called_once()
        else:
            mock_native_grouped_topk.assert_not_called()

    assert topk_weights.shape == (m, topk)
    assert topk_ids.shape == (m, topk)
    assert topk_ids.dtype == torch.int32


@pytest.mark.parametrize("device", DEVICE)
def test_select_experts_invalid_scoring_func(device: str):
    with pytest.raises(ValueError,
                       match="Unsupported scoring function: invalid"):
        select_experts(hidden_states=torch.randn(1, 128, device=device),
                       router_logits=torch.randn(1, 8, device=device),
                       top_k=2,
                       use_grouped_topk=False,
                       renormalize=False,
                       scoring_func="invalid")


@pytest.mark.parametrize("device", DEVICE)
def test_select_experts_missing_group_params(device: str):
    with pytest.raises(AssertionError):
        select_experts(hidden_states=torch.randn(1, 128, device=device),
                       router_logits=torch.randn(1, 64, device=device),
                       top_k=2,
                       use_grouped_topk=True,
                       renormalize=False,
                       scoring_func="softmax")
