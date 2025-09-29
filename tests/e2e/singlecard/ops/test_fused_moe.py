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

import gc
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_npu
from vllm.model_executor.layers.activation import SiluAndMul

from vllm_ascend.ops.moe.experts_selector import select_experts
from vllm_ascend.ops.moe.moe_mlp import unified_apply_mlp
from vllm_ascend.ops.moe.token_dispatcher import TokenDispatcherWithAllGather

NUM_EXPERTS = [8, 64]
EP_SIZE = [1]
TOP_KS = [2, 6]
DEVICE = ["npu"]


def apply_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
) -> torch.Tensor:
    w1 = w1.transpose(1, 2)
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]

    hidden_states = torch_npu.npu_swiglu(hidden_states)

    w2 = w2.transpose(1, 2)
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]

    return hidden_states


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
def test_token_dispatcher_with_all_gather(
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
    expert_map = None
    local_e = e
    w1_local = w1
    w2_local = w2

    score = torch.softmax(score, dim=-1, dtype=dtype)
    topk_weights, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)
    row_idx = (torch.arange(
        0,
        m * topk,
        device=device,
        dtype=torch.int32,
    ).view(topk, -1).permute(1, 0).contiguous())

    dispatcher_kwargs = {
        "num_experts": e,
        "top_k": topk,
        "num_local_experts": local_e,
    }
    dispatcher = TokenDispatcherWithAllGather(**dispatcher_kwargs)

    apply_router_weight_on_input = False
    dispatch_output = dispatcher.token_dispatch(
        hidden_states=a,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        row_idx=row_idx,
        expert_map=expert_map,
        apply_router_weight_on_input=apply_router_weight_on_input)

    sorted_hidden_states = dispatch_output["hidden_states"]
    group_list = dispatch_output["group_list"]
    group_list_type = dispatch_output.get("group_list_type", 1)

    expert_output = apply_mlp(hidden_states=sorted_hidden_states,
                              w1=w1_local,
                              w2=w2_local,
                              group_list=group_list,
                              group_list_type=group_list_type)

    combined_output = dispatcher.token_combine(hidden_states=expert_output,
                                               bias=None)

    torch_output = torch_moe(a, w1, w2, topk_weights, topk_ids, topk,
                             expert_map)

    torch.testing.assert_close(combined_output,
                               torch_output,
                               atol=4e-2,
                               rtol=1)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("m", [1, 33, 64])
@pytest.mark.parametrize("n", [128, 1024, 2048])
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", DEVICE)
def test_token_dispatcher_with_all_gather_quant(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
    device: str,
):
    context_mock = MagicMock()
    context_mock.fused_moe_state = 0
    with patch("vllm_ascend.ops.moe.moe_mlp.get_forward_context",
               return_value=context_mock):
        a = torch.randn((m, k), device=device, dtype=dtype) / 10
        w1 = torch.randn((e, k, 2 * n), device=device, dtype=torch.int8)
        w1_scale = torch.empty((e, 2 * n), device=device, dtype=dtype)
        w2 = torch.randn((e, n, k), device=device, dtype=torch.int8)
        w2_scale = torch.empty((e, k), device=device, dtype=dtype)

        score = torch.randn((m, e), device=device, dtype=dtype)
        expert_map = None
        local_e = e

        score = torch.softmax(score, dim=-1, dtype=dtype)
        topk_weights, topk_ids = torch.topk(score, topk)
        topk_ids = topk_ids.to(torch.int32)
        row_idx = (torch.arange(
            0,
            m * topk,
            device=device,
            dtype=torch.int32,
        ).view(topk, -1).permute(1, 0).contiguous())

        dispatcher_kwargs = {
            "num_experts": e,
            "top_k": topk,
            "num_local_experts": local_e,
        }
        dispatcher = TokenDispatcherWithAllGather(**dispatcher_kwargs)

        apply_router_weight_on_input = False
        dispatch_output = dispatcher.token_dispatch(
            hidden_states=a,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            row_idx=row_idx,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            with_quant=True)

        sorted_hidden_states = dispatch_output["hidden_states"]
        group_list = dispatch_output["group_list"]
        group_list_type = dispatch_output.get("group_list_type", 1)
        dynamic_scale = dispatch_output["dynamic_scale"]

        expert_output = unified_apply_mlp(hidden_states=sorted_hidden_states,
                                          w1=w1,
                                          w1_scale=w1_scale,
                                          w2=w2,
                                          w2_scale=w2_scale,
                                          group_list=group_list,
                                          group_list_type=group_list_type,
                                          dynamic_scale=dynamic_scale,
                                          with_quant=True)
        combined_output = dispatcher.token_combine(hidden_states=expert_output,
                                                   bias=None)
        assert combined_output.shape == (m, k)
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()


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

    with patch("vllm_ascend.ops.moe.experts_selector._native_grouped_topk"
               ) as mock_native_grouped_topk:
        mock_native_grouped_topk.side_effect = lambda x, num_groups, k: torch.randn_like(
            x)

        topk_weights, topk_ids, row_idx = select_experts(
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
    assert row_idx.shape == (m, topk)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


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
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("device", DEVICE)
def test_select_experts_missing_group_params(device: str):
    with pytest.raises(AssertionError):
        select_experts(hidden_states=torch.randn(1, 128, device=device),
                       router_logits=torch.randn(1, 64, device=device),
                       top_k=2,
                       use_grouped_topk=True,
                       renormalize=False,
                       scoring_func="softmax")
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
