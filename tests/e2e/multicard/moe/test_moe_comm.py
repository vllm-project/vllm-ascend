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

from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig
from vllm import forward_context

from vllm_ascend.distributed import moe_comm_method
from vllm_ascend.distributed.moe_comm_method import (AllGatherCommImpl,
                                                     NativeAllGatherCommImpl)


@pytest.mark.parametrize("num_tokens", [16, 128])
@pytest.mark.parametrize("hidden_size", [64, 128])
@pytest.mark.parametrize("global_num_experts", [8, 16])
@pytest.mark.parametrize("top_k_num", [2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_local_experts", [4, 8])
@pytest.mark.parametrize("ep_rank", [0, 1])
def test_all_gather_comm_impl(
    num_tokens,
    hidden_size,
    global_num_experts,
    top_k_num,
    dtype,
    num_local_experts,
    ep_rank,
):
    """
    Tests the AllGatherCommImpl against the NativeAllGatherCommImpl.

    This test compares the outputs of the NPU-optimized AllGatherCommImpl
    with a native PyTorch implementation (NativeAllGatherCommImpl) to ensure
    correctness across various configurations.
    """
    if top_k_num > global_num_experts:
        pytest.skip("top_k_num cannot be greater than global_num_experts")
    if num_local_experts > global_num_experts:
        pytest.skip(
            "num_local_experts cannot be greater than global_num_experts")

    device = torch.device("npu")
    hf_config = PretrainedConfig(
        num_experts_per_tok=top_k_num,
        num_experts=global_num_experts,
    )

    # Instantiate implementations
    native_impl = NativeAllGatherCommImpl(device, dtype, hf_config)

    all_gather_impl = AllGatherCommImpl(device, dtype, hf_config)

    # TODO: Find out if this is the correct way to mock the forward context and ep group
    # Mock get_forward_context to return an object with moe_comm_method
    forward_context._forward_context = SimpleNamespace(
        moe_comm_method=all_gather_impl)
    # Mock get_ep_group to return a fake group with the specified ep_rank
    fake_ep_group = SimpleNamespace(rank_in_group=ep_rank)
    moe_comm_method.get_ep_group = lambda: fake_ep_group

    # --- Input Data ---
    hidden_states = torch.randn(num_tokens,
                                hidden_size,
                                device=device,
                                dtype=dtype)
    topk_ids = torch.randint(0,
                             global_num_experts, (num_tokens, top_k_num),
                             device=device,
                             dtype=torch.int32)
    topk_weights = torch.rand(num_tokens, top_k_num, device=device).to(dtype)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=1)

    num_experts = global_num_experts

    expert_map = None
    if num_local_experts < global_num_experts:
        # Create a map where some experts are local and some are not
        expert_map = torch.full((global_num_experts, ), -1, device=device)
        expert_map[ep_rank * num_local_experts:(ep_rank + 1) *
                   num_local_experts] = torch.arange(num_local_experts,
                                                     device=device)
    num_experts = num_local_experts

    # --- Run Native Implementation (Golden Reference) ---
    native_hidden_states_out = hidden_states.clone()
    (
        native_permuted_hidden,
        native_expert_tokens,
        _,
    ) = native_impl._pre_process(hidden_states, topk_ids, topk_weights,
                                 expert_map, num_experts)
    # Simulate MLP output
    native_mlp_output = torch.randn_like(native_permuted_hidden)
    native_impl._post_process(native_mlp_output, native_hidden_states_out)

    # --- Run AllGather Implementation ---
    all_gather_hidden_states_out = hidden_states.clone()
    (
        all_gather_permuted_hidden,
        all_gather_expert_tokens,
        _,
    ) = torch.ops.vllm.moe_comm_pre_process(hidden_states, topk_ids,
                                            topk_weights, expert_map,
                                            num_experts)

    # Use the same simulated MLP output for a fair comparison
    all_gather_mlp_output = native_mlp_output.clone()

    torch.ops.vllm.moe_comm_post_process(all_gather_mlp_output,
                                         all_gather_hidden_states_out)

    # --- Assertions ---
    # Define tolerance based on dtype
    atol = 1e-3 if dtype == torch.float16 else 1e-2
    rtol = 1e-3 if dtype == torch.float16 else 1e-2

    # 1. Compare expert_tokens from pre_process
    assert torch.allclose(native_expert_tokens.to(
        all_gather_expert_tokens.device),
                          all_gather_expert_tokens,
                          atol=atol,
                          rtol=rtol), "Expert tokens do not match."

    # 2. Compare permuted_hidden_states from pre_process
    num_valid_tokens = native_expert_tokens.sum()
    assert torch.allclose(native_permuted_hidden[:num_valid_tokens].to(
        all_gather_permuted_hidden.device),
                          all_gather_permuted_hidden[:num_valid_tokens],
                          atol=atol,
                          rtol=rtol), "Permuted hidden states do not match."

    # 3. Compare final hidden_states from post_process
    assert torch.allclose(native_hidden_states_out.to(
        all_gather_hidden_states_out.device),
                          all_gather_hidden_states_out,
                          atol=atol,
                          rtol=rtol), "Final hidden states do not match."
