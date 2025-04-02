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
# Adapted from vllm/tests/kernels/test_moe.py

import os
from typing import Callable, Optional

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_dp_group
from vllm.model_executor.layers.fused_moe.layer import \
    UnquantizedFusedMoEMethod, FusedMoE, determine_expert_map
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm_ascend.distributed.parallel_state import get_ep_group, get_etp_group
from vllm.forward_context import ForwardContext, get_forward_context


def fused_experts_with_mc2(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    expert_map: torch.Tensor = None,
    moe_all_to_all_group_name: str = None,
) -> torch.Tensor:
    # Decoder Stage, seq = 1
    # if torch.distributed.get_rank() == 0:
    #     print(w1.shape)
    #     print(hidden_states.shape)

    global_bs = 0
    moe_expert_num = len(expert_map)
    # hidden_states = hidden_states.bfloat16()
    kwargs = {
        "x": hidden_states,
        "expert_ids": topk_ids,
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": moe_expert_num,
        "global_bs": global_bs,
    }

    rank = torch.distributed.get_rank()

    quant_mode = 0
    ep_group = get_ep_group().device_group
    local_rank = torch.distributed.get_rank(group=ep_group)
    all_to_all_group_size = torch.distributed.get_world_size(ep_group)

    world_szie = torch.distributed.get_world_size()
    tp_size = world_szie // all_to_all_group_size
    tp_rank = rank % tp_size

    stage1_kwargs = {
        "scales": None,
        "quant_mode": quant_mode,
        "group_ep": moe_all_to_all_group_name,
        "ep_world_size": all_to_all_group_size,
        "ep_rank_id": local_rank,
        # "group_tp": self.moe_rs_group_name,
        "group_tp": moe_all_to_all_group_name,
        "tp_world_size": tp_size,
        "tp_rank_id": tp_rank,
    }
    kwargs.update(stage1_kwargs)

    output = torch_npu.npu_moe_distribute_dispatch(**kwargs)
    # comm_stream.wait_stream(torch.npu.current_stream())
    expand_x, dynamic_scale, expand_idx, expert_token_nums, ep_recv_counts = output[
        0:5]

    w1 = w1.transpose(1, 2)
    expert_token_nums = torch.cumsum(expert_token_nums,
                                     dim=0,
                                     dtype=torch.int64)
    group_list = expert_token_nums.to(torch.int64)
    gate_up_out_list = torch_npu.npu_grouped_matmul(
        x=[expand_x],
        weight=[w1],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=group_list,
    )

    # TODO: Remove this in the future.
    gate_up_out = torch.cat(gate_up_out_list, dim=0)
    gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    w2 = w2.transpose(1, 2)
    down_out_list = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=group_list,
    )

    down_out_list = torch.cat(down_out_list, dim=0)

    # moeCombine
    kwargs = {
        "expand_x": down_out_list,
        "expert_ids": topk_ids,
        "expand_idx": expand_idx,
        "expert_scales": topk_weights.to(torch.float32),
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": moe_expert_num,
        "global_bs": 0,
    }
    tp_recv_counts = torch.empty(1,
                                 dtype=torch.int32,
                                 device=hidden_states.device)
    stage3_kwargs = {
        "ep_send_counts": ep_recv_counts,
        "group_ep": moe_all_to_all_group_name,
        "ep_world_size": all_to_all_group_size,
        "ep_rank_id": local_rank,
        "tp_send_counts": tp_recv_counts,
        # "group_tp": self.moe_rs_group_name,
        "group_tp": moe_all_to_all_group_name,
        "tp_world_size": tp_size,
        "tp_rank_id": tp_rank,
    }
    kwargs.update(stage3_kwargs)

    hidden_states = torch_npu.npu_moe_distribute_combine(**kwargs)

    return hidden_states


def fused_experts_with_mc2(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        expert_map: torch.Tensor = None,
        moe_all_to_all_group_name: str = None,
) -> torch.Tensor:
    global_bs = 0
    moe_expert_num = len(expert_map)
    kwargs = {
        "x": hidden_states,
        "expert_ids": topk_ids,
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": moe_expert_num,
        "global_bs": global_bs,
    }

    rank = torch.distributed.get_rank()

    quant_mode = 0
    ep_group = get_ep_group().device_group
    local_rank = torch.distributed.get_rank(group=ep_group)
    all_to_all_group_size = torch.distributed.get_world_size(ep_group)

    world_szie = torch.distributed.get_world_size()
    tp_size = world_szie // all_to_all_group_size
    tp_rank = rank % tp_size

    stage1_kwargs = {
        "scales": None,
        "quant_mode": quant_mode,
        "group_ep": moe_all_to_all_group_name,
        "ep_world_size": all_to_all_group_size,
        "ep_rank_id": local_rank,
        # "group_tp": self.moe_rs_group_name,
        "group_tp": moe_all_to_all_group_name,
        "tp_world_size": tp_size,
        "tp_rank_id": tp_rank,
    }
    kwargs.update(stage1_kwargs)

    output = torch_npu.npu_moe_distribute_dispatch(**kwargs)
    # comm_stream.wait_stream(torch.npu.current_stream())
    expand_x, dynamic_scale, expand_idx, expert_token_nums, ep_recv_counts = output[0:5]

    w1 = w1.transpose(1, 2)
    expert_token_nums = torch.cumsum(expert_token_nums, dim=0, dtype=torch.int64)
    group_list = expert_token_nums.to(torch.int64)
    gate_up_out_list = torch_npu.npu_grouped_matmul(
        x=[expand_x],
        weight=[w1],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=group_list,
    )

    # TODO: Remove this in the future.
    gate_up_out = torch.cat(gate_up_out_list, dim=0)
    gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    w2 = w2.transpose(1, 2)
    down_out_list = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=group_list,
    )

    down_out_list = torch.cat(down_out_list, dim=0)

    # moeCombine
    kwargs = {
        "expand_x": down_out_list,
        "expert_ids": topk_ids,
        "expand_idx": expand_idx,
        "expert_scales": topk_weights.to(torch.float32),
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": moe_expert_num,
        "global_bs": 0,
    }
    tp_recv_counts = output[5]
    stage3_kwargs = {
        "ep_send_counts": ep_recv_counts,
        "group_ep": moe_all_to_all_group_name,
        "ep_world_size": all_to_all_group_size,
        "ep_rank_id": local_rank,
        "tp_send_counts": tp_recv_counts,
        # "group_tp": self.moe_rs_group_name,
        "group_tp": moe_all_to_all_group_name,
        "tp_world_size": tp_size,
        "tp_rank_id": tp_rank,
    }
    kwargs.update(stage3_kwargs)

    hidden_states = torch_npu.npu_moe_distribute_combine(**kwargs)

    return hidden_states


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    expert_map: torch.Tensor = None,
) -> torch.Tensor:
    """
    Fused experts with top-k routing.

    Args:
        hidden_states: Hidden states of shape (num_tokens, hidden_size).
        w1: Expert weights1 of shape (num_experts, intermediate_size * 2, hidden_size).
        w2: Expert weights2 of shape (num_experts, hidden_size, intermediate_size).
        topk_weights: Routing weights of shape (num_tokens, top_k).
        topk_ids: Selected expert IDs of shape (num_tokens, top_k).
        top_k: Number of experts to select.
        expert_map: Expert mapping of shape (num_experts,).

    Returns:
        hidden_states: Hidden states after routing.
    """
    """
    # Check constraints.
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    """
    # if torch.distributed.get_rank() == 0:
    #     print(w1.shape)
    #     print(hidden_states.shape)

    original_shape = hidden_states.shape
    # assert len(original_shape) == 2

    num_tokens = hidden_states.shape[:-1].numel()
    num_experts = w1.shape[0]
    dtype = hidden_states.dtype
    device = hidden_states.device
    # assert dtype in [torch.float32, torch.float16, torch.bfloat16
    #                  ], "Only float32, float16, and bfloat16 are supported"

    if expert_map is not None:
        # Generate token indices and flatten
        token_indices = (torch.arange(num_tokens,
                                      device=device,
                                      dtype=torch.int64).unsqueeze(1).expand(
                                          -1, top_k).reshape(-1))

        # Flatten token-to-expert mappings and map to local experts
        weights_flat = topk_weights.view(-1)
        experts_flat = topk_ids.view(-1)
        local_experts_flat = expert_map[experts_flat]

        # Filter valid token-expert pairs
        mask = local_experts_flat != -1
        filtered_weights = torch.where(
            mask, weights_flat, torch.zeros_like(weights_flat)).to(dtype)
        filtered_experts = torch.where(
            mask, local_experts_flat,
            torch.full_like(local_experts_flat,
                            num_experts)).to(topk_ids.dtype)

        # Sort by local expert IDs
        sort_indices = torch.argsort(filtered_experts)
        sorted_token_indices = token_indices[sort_indices]
        sorted_weights = filtered_weights[sort_indices]

        # Compute token counts with minlength of num_experts
        # This is equivalent to but faster than:
        # >>> token_counts = torch.bincount(filtered_experts, minlength=num_experts)[:-1]
        token_counts = torch.zeros(num_experts + 1,
                                   device=device,
                                   dtype=torch.int64)
        ones = torch.ones_like(filtered_experts, dtype=torch.int64)
        token_counts.scatter_add_(0, filtered_experts.to(torch.int64), ones)
        token_counts = token_counts[:num_experts]
        expert_tokens = torch.cumsum(token_counts, dim=0, dtype=torch.int64)

        # Rearrange hidden_states
        sorted_hidden_states = hidden_states[sorted_token_indices]
    else:
        row_idx_len = num_tokens * top_k
        row_idx = (torch.arange(0,
                                row_idx_len,
                                dtype=torch.int32,
                                device=device).view(top_k, -1).permute(
                                    1, 0).contiguous())
        sorted_hidden_states, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=topk_ids,
            active_num=num_tokens)

        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts)
        expert_tokens = expert_tokens.to(torch.int64)

    w1 = w1.transpose(1, 2)
    gate_up_out_list = torch_npu.npu_grouped_matmul(
        x=[sorted_hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
    )

    # TODO: Remove this in the future.
    gate_up_out = torch.cat(gate_up_out_list, dim=0)
    gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    w2 = w2.transpose(1, 2)
    down_out_list = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
    )

    down_out_list = torch.cat(down_out_list, dim=0)

    if expert_map is not None:
        weighted_down_out = down_out_list * sorted_weights.unsqueeze(1)

        final_hidden_states = torch.zeros(*original_shape,
                                          device=hidden_states.device,
                                          dtype=dtype)

        # TODO: npu_grouped_matmul output random values at [num_valid_tokens:, ...]
        # This created multiple NaN and index_add_ will mix them up which harms accracy
        # remove this mask and filter after it being fixed
        num_valid_tokens = mask.sum()
        valid_token_mask = torch.arange(0, sorted_token_indices.shape[0], device=device).unsqueeze(1) < num_valid_tokens
        valid_output = torch.where(valid_token_mask, weighted_down_out, torch.zeros_like(weighted_down_out)).to(dtype)
        final_hidden_states.index_add_(0, sorted_token_indices, valid_output)
    else:
        # TODO: Reorder device memory 2 times here, replace the current
        # implementation here when suitable operators become available.
        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            down_out_list,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
        )

    return final_hidden_states


def native_grouped_topk(
    topk_weights: torch.Tensor,
    num_expert_group: Optional[int],
    topk_group: Optional[int],
):
    topk_group = 0 if topk_group is None else topk_group
    num_expert_group = 0 if num_expert_group is None else num_expert_group

    num_token = topk_weights.shape[0]
    grouped_weights = topk_weights.view(num_token, num_expert_group,
                                        -1).max(dim=-1).values
    topk_group_indices = torch.topk(grouped_weights.to(torch.float32),
                                    k=topk_group,
                                    dim=-1,
                                    sorted=False)[1]
    topk_group_mask = torch.zeros_like(grouped_weights)
    topk_group_mask.scatter_(1, topk_group_indices, 1)
    topk_weight_mask = (topk_group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        topk_weights.shape[-1] // num_expert_group).reshape(num_token, -1))
    topk_weights = topk_weights.masked_fill(~topk_weight_mask.bool(), 0.0)

    return topk_weights


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    is_prefill: Optional[bool] = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-k experts based on router logits.

    Args:
        hidden_states: Hidden states of shape (num_tokens, hidden_size).
        router_logits: Router logits of shape (num_tokens, num_experts).
        top_k: Number of experts to select.
        use_grouped_topk: Whether to group experts before selecting top-k.
        renormalize: Whether to renormalize the routing weights.
        topk_group: Number of expert groups to select from.
        num_expert_group: Number of experts in each group.
        custom_routing_function: Custom routing function.
        scoring_func: Scoring function to use.
        e_score_correction_bias: Correction bias to apply to expert scores.

    Returns:
        topk_weights: Routing weights of shape (num_tokens, top_k).
        topk_ids: Selected expert IDs of shape (num_tokens, top_k).

    Raises:
        ValueError: If an unsupported scoring function is provided.
    """
    # assert hidden_states.shape[0] == router_logits.shape[0], (
    #     "Number of tokens mismatch")
    if os.environ.get("VLLM_ENABLE_GRAPH_MODE") == "1" and not is_prefill:
        topk_weight, topk_idx, _ = torch.ops.npu_inference.npu_moe_gating_top_k(
            router_logits, 
            k=top_k, # topk当前写8
            bias=e_score_correction_bias, 
            k_group=topk_group, # fix: 4 
            group_count=num_expert_group, # fix 8
            group_select_mode=1, # 0: group中的最大; 1: topk2.sum(fix)
            renorm=0, # 0: softmax->topk(fix); 1: topk->softmax
            norm_type=1, # 0: softmax; 1: sigmoid(fix) 
            # out_flag=False, # todo new api; 第三个输出是否输出 
            # y2_flag=False, # old api; 第三个输出是否输出
            routed_scaling_factor=1,
            eps=float(1e-20))
        return topk_weight, topk_idx

    if custom_routing_function is not None:
        raise NotImplementedError(
            "Custom routing function is not supported now")

    if scoring_func == "softmax":
        # NOTE: vLLM use dtype=torch.float here
        topk_weights = router_logits.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        topk_weights = router_logits.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None

        if e_score_correction_bias is not None:
            # Store original scores before applying correction bias. We use biased
            # scores for expert selection but original scores for routing weights
            original_weights = topk_weights
            topk_weights = topk_weights + e_score_correction_bias.unsqueeze(0)

        # TODO: Change to npu_group_topk when the latest CANN and NNAL is available
        # >>> torch_npu._npu_group_topk(topk_weights, group_num=num_expert_group, k=topk_group)
        topk_weights = native_grouped_topk(topk_weights, num_expert_group,
                                           topk_group)
        # TODO bfloat16 is not supported in torch.topk with ge graph.
        if e_score_correction_bias is not None:
            topk_ids = torch.topk(topk_weights.to(torch.float32),
                                  k=top_k,
                                  dim=-1,
                                  sorted=False)[1]
            # Use original unbiased scores for the routing weights
            topk_weights = original_weights.gather(1, topk_ids)
        else:
            topk_weights, topk_ids = torch.topk(topk_weights.to(torch.float32),
                                                k=top_k,
                                                dim=-1,
                                                sorted=False)
    else:
        topk_weights, topk_ids = topk_weights.topk(top_k, dim=-1)
        topk_weights = topk_weights.to(hidden_states.dtype)

    # Required by npu_moe_init_routing
    topk_ids = topk_ids.to(torch.int32)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):

    def __init__(self):
        super().__init__()
        vllm_config = get_current_vllm_config()

        ep_group = get_ep_group()
        self.ep_size = ep_group.world_size
        self.global_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.local_batch_size = self.global_batch_size // self.ep_size

        try:
            device_group = ep_group.device_group
            # TODO: Try local_rank = ep_group.rank_in_group
            local_rank = torch.distributed.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self.moe_all_to_all_group_name = backend.get_hccl_comm_name(
                local_rank)
        except AttributeError as e:
            self.moe_all_to_all_group_name = None

    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod, self).process_weights_after_loading(layer)
        layer.w13_weight = torch.nn.Parameter(self._maybe_pad_weight(
            layer.w13_weight.data),
                                              requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(self._maybe_pad_weight(
            layer.w2_weight.data),
                                             requires_grad=False)

    def forward_oot(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        is_prefill = False,
        **kwargs,
    ):
        # assert router_logits.shape[
        #     1] == global_num_experts, "Number of global experts mismatch"
        # set prefill as false always, should fix this
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            is_prefill=is_prefill
        )

        if os.environ.get("VLLM_ENABLE_MC2") == "1" and not is_prefill:
            return fused_experts_with_mc2(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                top_k=top_k,
                expert_map=expert_map,
                moe_all_to_all_group_name=self.moe_all_to_all_group_name)
        else:
            return fused_experts(hidden_states=x,
                                 w1=layer.w13_weight,
                                 w2=layer.w2_weight,
                                 topk_weights=topk_weights,
                                 topk_ids=topk_ids,
                                 top_k=top_k,
                                 expert_map=expert_map)


class AscendFusedMoE(FusedMoE):

    def __init__(self,
                 num_experts,
                 top_k,
                 hidden_size,
                 intermediate_size,
                 params_dtype=None,
                 reduce_results=False,
                 renormalize=True,
                 use_grouped_topk=False,
                 num_expert_group=None,
                 topk_group=None,
                 quant_config=None,
                 tp_size=None,
                 ep_size=None,
                 dp_size=None,
                 prefix="",
                 custom_routing_function=None,
                 scoring_func="softmax",
                 e_score_correction_bias=None,
                 activation="silu"):
        super(FusedMoE, self).__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.ep_size = get_ep_group().world_size
        self.tp_size = get_etp_group().world_size
        self.dp_size = (dp_size
                        if dp_size is not None else get_dp_group().world_size)
        self.dp_rank = (0
                        if self.dp_size == 1 else get_dp_group().rank_in_group)

        self.top_k = top_k
        self.num_experts = num_experts
        self.global_num_experts = num_experts
        assert intermediate_size % self.tp_size == 0
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias
        self.expert_map = None
        self.activation = activation

        if self.ep_size > 1:
            # Create a tensor of size num_experts filled with -1
            self.local_num_experts, self.expert_map = determine_expert_map(self.ep_size, get_ep_group().rank_in_group, self.global_num_experts)
            self.tp_rank = get_etp_group().rank_in_group
            self.ep_rank = get_ep_group().rank_in_group
        else:
            # Adjust TP size for DP attention
            # haven't test its functionality yet, may remove in the future
            self.tp_rank = self.tp_size * self.dp_rank
            self.ep_rank = 0
            self.tp_size = self.tp_size * self.dp_size
            self.ep_size = 1
            self.local_num_experts = self.global_num_experts
            self.expert_map = None

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError("Only softmax scoring function is supported for "
                             "non-grouped topk.")

        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = (
                AscendUnquantizedFusedMoEMethod())
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix)
        assert self.quant_method is not None

        local_num_experts = torch.sum(self.expert_map != -1) \
            if self.expert_map is not None else num_experts

        moe_quant_params = {
            "num_experts": local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition":
            self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if (self.quant_method.__class__.__name__
                in ("GPTQMarlinMoEMethod", "CompressedTensorsWNA16MoEMethod")):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        self.quant_method.create_weights(layer=self, **moe_quant_params)

    def forward(self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                is_prefill: bool, 
                top_k=None):
        assert self.quant_method is not None

        if top_k:
            real_top_k = top_k
        else:
            real_top_k = self.top_k

        if self.dp_size > 1:
            cu_tokens_across_dp_cpu = get_forward_context(
            ).dp_metadata.cu_tokens_across_dp_cpu
            hidden_states = self.naive_multicast(hidden_states,
                                                 cu_tokens_across_dp_cpu)
            router_logits = self.naive_multicast(router_logits,
                                                 cu_tokens_across_dp_cpu)

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=real_top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            is_prefill=is_prefill)

        if self.dp_size > 1:
            start = 0 if self.dp_rank == 0 else cu_tokens_across_dp_cpu[
                self.dp_rank - 1]
            end = cu_tokens_across_dp_cpu[self.dp_rank]

            all_hidden_states = get_dp_group().all_reduce(final_hidden_states)
            final_hidden_states = all_hidden_states[start:end, :]

        # if self.reduce_results and self.tp_size > 1:
        if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states
