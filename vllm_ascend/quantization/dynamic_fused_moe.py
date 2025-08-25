from typing import Any, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch_npu
from torch.nn.functional import pad
from vllm.distributed import GroupCoordinator, get_ep_group

from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.torchair.utils import npu_stream_switch, npu_wait_tensor
from vllm_ascend.utils import (AscendSocVersion, dispose_tensor,
                               get_ascend_soc_version)


def dynamic_quant(
        hidden_states: torch.Tensor,
        dynamic_scale: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if dynamic_scale is None:
        unquantized_hidden_states = hidden_states
        hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(
            hidden_states)
        # Dispose the original unquantized hidden states
        # to save npu memory because they're no longer used.
        dispose_tensor(unquantized_hidden_states)
    else:
        pertoken_scale = dynamic_scale
    return hidden_states, pertoken_scale


def init_routing_quant(hidden_states, top_k, topk_ids, global_num_experts):
    num_tokens, _ = hidden_states.shape
    row_idx_len = num_tokens * top_k
    row_idx = (torch.arange(0,
                            row_idx_len,
                            dtype=torch.int32,
                            device=hidden_states.device).view(
                                top_k, -1).permute(1, 0).contiguous())
    hidden_states, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
        hidden_states,
        row_idx=row_idx,
        expert_idx=topk_ids,
        active_num=num_tokens)

    expanded_row_idx = (expanded_row_idx.view(top_k, -1).permute(
        1, 0).contiguous().view(-1))
    global_expert_tokens = torch.bincount(expanded_expert_idx,
                                          minlength=global_num_experts)
    global_expert_tokens = global_expert_tokens.to(torch.int32)
    quantized_tokens, token_scales = torch_npu.npu_dynamic_quant(hidden_states)
    return quantized_tokens, expanded_row_idx, global_expert_tokens, token_scales


def cumsum_group_list(group_list: torch.Tensor,
                      group_list_type: int,
                      active_num: int = 0,
                      expert_num: int = 0) -> torch.Tensor:
    if group_list_type not in [0, 1, 2]:
        raise ValueError(
            f"group_list_type should be in [0, 1, 2], but received {group_list_type}"
        )

    if group_list_type == 0:
        return group_list
    if group_list_type == 1:
        return group_list.cumsum(dim=0)

    experts = pad(group_list[:, 0], (1, 0))
    tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
    cumsum_group_list = torch.full(size=(expert_num, ),
                                   fill_value=active_num,
                                   dtype=group_list.dtype,
                                   device=group_list.device)

    for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
        if end > start:
            cumsum_group_list[start:end] = tokens[i]

    return cumsum_group_list


def apply_mlp_decode(hidden_states: torch.Tensor,
                     w1: torch.Tensor,
                     w1_scale: torch.Tensor,
                     w2: torch.Tensor,
                     w2_scale: torch.Tensor,
                     group_list: torch.Tensor,
                     dynamic_scale: torch.Tensor = None,
                     group_list_type: int = 1,
                     fusion: bool = False) -> torch.Tensor:
    """
    apply MLP: gate_up_proj -> swiglu -> down_proj
    Args:
        hidden_states: input hidden states with shape (num_tokens, hidden_size).
        w1: expert weights1 with shape
            (num_experts, hidden_size, intermediate_size * 2).
        w1_scale: weights1 scale with shape (num_experts, intermediate_size * 2).
        w2: expert weights2 with shape
            (num_experts, intermediate_size, hidden_size).
        w2_scale: weights2 scale with shape (num_experts, hidden_size).
        group_list: number of tokens for each expert, follow cumsum mode, and
            with shape (num_experts).
        dynamic_scale: hidden_states scale with shape (num_tokens).
        group_list_type: type of group_list. 0: cumsum; 1: count; 2: key-value.
        fusion: whether to fuse gate_up_proj, swiglu and dynamic_quant.
    Returns:
        hidden_states: output hidden states after MLP.
    """

    hidden_states, pertoken_scale = dynamic_quant(hidden_states, dynamic_scale)

    if fusion:
        # gmm1: gate_up_proj & act_fn: swiglu
        hidden_states, swiglu_out_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
            x=hidden_states,
            weight=w1,
            group_list=cumsum_group_list(group_list, group_list_type),
            weight_scale=w1_scale,
            x_scale=pertoken_scale)
    else:
        # gmm1: gate_up_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w1],
            split_item=3,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=torch.int32)[0]
        # act_fn: swiglu
        hidden_states, swiglu_out_scale = torch_npu.npu_dequant_swiglu_quant(
            x=hidden_states,
            weight_scale=w1_scale,
            activation_scale=pertoken_scale,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=group_list,
            activate_left=True,
            quant_mode=1,
        )

    # gmm2: down_proj
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=[w2_scale],
        per_token_scale=[swiglu_out_scale],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=w2_scale.dtype)[0]

    return hidden_states


def apply_mlp(hidden_states: torch.Tensor,
              w1: torch.Tensor,
              w1_scale: torch.Tensor,
              w2: torch.Tensor,
              w2_scale: torch.Tensor,
              group_list: torch.Tensor,
              w1_scale_bias: torch.Tensor = None,
              w2_scale_bias: torch.Tensor = None,
              dynamic_scale: torch.Tensor = None,
              group_list_type: int = 1,
              fusion: bool = False) -> torch.Tensor:
    """
    apply MLP: gate_up_proj -> swiglu -> down_proj
    Args:
        hidden_states: input hidden states with shape (num_tokens, hidden_size).
        w1: expert weights1 with shape
            (num_experts, hidden_size, intermediate_size * 2).
        w1_scale: weights1 scale with shape (num_experts, intermediate_size * 2).
        w2: expert weights2 with shape
            (num_experts, intermediate_size, hidden_size).
        w2_scale: weights2 scale with shape (num_experts, hidden_size).
        group_list: number of tokens for each expert, follow cumsum mode, and
            with shape (num_experts).
        w1_scale_bias: weights1 bias with shape (num_experts, intermediate_size * 2).
        w2_scale_bias: weights2 bias with shape (num_experts, hidden_size).
        dynamic_scale: hidden_states scale with shape (num_tokens).
        group_list_type: type of group_list. 0: cumsum; 1: count; 2: key-value.
        fusion: whether to fuse gate_up_proj, swiglu and dynamic_quant.
    Returns:
        hidden_states: output hidden states after MLP.
    """

    hidden_states, pertoken_scale = dynamic_quant(hidden_states, dynamic_scale)

    bias1, bias2 = None, None
    _output_dtype = w2_scale.dtype

    if w1_scale_bias is not None:
        if group_list_type == 0:
            group_list = torch.cat(
                [group_list[:1], torch.diff(group_list, dim=0)])
            group_list_type = 1
        bias1 = [w1_scale_bias] if not fusion else w1_scale_bias
        bias2 = [w2_scale_bias]
        # TODO w4a8 scene: dynamic acquisition of dtype in the future
        _output_dtype = torch.bfloat16

    if fusion:
        # gmm1: gate_up_proj & act_fn: swiglu
        hidden_states, swiglu_out_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
            x=hidden_states,
            weight=w1,
            bias=bias1,
            group_list=cumsum_group_list(group_list, group_list_type),
            weight_scale=w1_scale,
            x_scale=pertoken_scale)
    else:
        # gmm1: gate_up_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w1],
            scale=[w1_scale.to(w2_scale.dtype)],
            bias=bias1,
            per_token_scale=[pertoken_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype)[0]
        # act_fn: swiglu
        hidden_states = torch_npu.npu_swiglu(hidden_states)
        hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(
            hidden_states)

    # gmm2: down_proj
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=[w2_scale],
        bias=bias2,
        per_token_scale=[swiglu_out_scale],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=_output_dtype)[0]

    return hidden_states


def fused_experts_with_mc2(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    expert_map: torch.Tensor = None,
    moe_all_to_all_group_name: str = "",
    log2phy: torch.Tensor = None,
    global_redundant_expert_num: int = 0,
    shared_experts: Optional[Any] = None,
    is_torchair: bool = False,
    quantized_x_for_share: Optional[Any] = None,
    dynamic_scale_for_share: Optional[Any] = None,
    mc2_mask: Optional[torch.Tensor] = None,
    shared_gate_up: Optional[Any] = None,
    shared_dequant_scale: Optional[Any] = None,
    w1_scale_bias: torch.Tensor = None,
    w2_scale_bias: torch.Tensor = None,
    fusion_mlp: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    assert mc2_mask is not None
    if log2phy is not None:
        topk_ids = log2phy[topk_ids]

    quant_mode = 2
    ep_group = get_mc2_group()
    ep_rank_id = ep_group.rank_in_group
    ep_world_size = ep_group.world_size

    # NOTE: Currently, when in A3 or in torchair graph, we need to pass in some extra param into dispatch & combine
    need_extra_args = (get_ascend_soc_version() == AscendSocVersion.A3
                       or is_torchair)

    # NOTE: Currently, when in A3, we need to pass in some extra param into dispatch & combine
    a3_need_extra_args = get_ascend_soc_version() == AscendSocVersion.A3

    enable_dispatch_v2 = hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")

    if (expert_map is not None):
        moe_expert_num = len(expert_map) + global_redundant_expert_num
    else:
        moe_expert_num = global_redundant_expert_num
    # hidden_states = hidden_states.bfloat16()
    kwargs_mc2 = {
        "x": hidden_states,
        "expert_ids": topk_ids,
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": moe_expert_num,
        "global_bs": 0,
    }

    stage1_kwargs = {
        "scales": None,
        "quant_mode": quant_mode,
        "group_ep": moe_all_to_all_group_name,
        "ep_world_size": ep_world_size,
        "ep_rank_id": ep_rank_id,
    }
    if need_extra_args:
        stage1_kwargs.update({
            "group_tp": moe_all_to_all_group_name,
            "tp_world_size": 1,
            "tp_rank_id": 0,
        })
    if a3_need_extra_args and enable_dispatch_v2:
        stage1_kwargs.update({
            "x_active_mask": mc2_mask,
        })
    kwargs_mc2.update(stage1_kwargs)

    output = torch_npu.npu_moe_distribute_dispatch_v2(
        **kwargs_mc2
    ) if enable_dispatch_v2 else torch_npu.npu_moe_distribute_dispatch(
        **kwargs_mc2)
    # comm_stream.wait_stream(torch.npu.current_stream())
    expand_x, dynamic_scale, assist_info_for_combine, expert_token_nums, ep_recv_counts = output[
        0:5]

    if shared_experts is not None:
        with npu_stream_switch("moe_secondary", 0):
            npu_wait_tensor(shared_gate_up, expand_x)
            shared_act_out = shared_experts.act_fn(
                (shared_gate_up, shared_dequant_scale))
            shared_act, swiglu_out_scale = shared_act_out[0], shared_act_out[1]

    # `expand_x` will be disposed in the `apply_mlp` function
    if w1_scale_bias is None:
        down_out_list = apply_mlp_decode(expand_x,
                                         w1,
                                         w1_scale,
                                         w2,
                                         w2_scale,
                                         expert_token_nums,
                                         dynamic_scale=dynamic_scale,
                                         fusion=fusion_mlp)
    else:
        # w4a8 scene, cannot use apply_mlp_decode because the operator is not supported
        down_out_list = apply_mlp(expand_x,
                                  w1,
                                  w1_scale,
                                  w2,
                                  w2_scale,
                                  expert_token_nums,
                                  dynamic_scale=dynamic_scale,
                                  w1_scale_bias=w1_scale_bias,
                                  w2_scale_bias=w2_scale_bias,
                                  fusion=fusion_mlp)

    # moeCombine
    kwargs_mc2 = {
        "expand_x": down_out_list,
        "expert_ids": topk_ids,
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
        "ep_world_size": ep_world_size,
        "ep_rank_id": ep_rank_id,
    }
    if enable_dispatch_v2:
        stage3_kwargs.update({
            "assist_info_for_combine":
            assist_info_for_combine,
        })
    else:
        stage3_kwargs.update({
            "expand_idx": assist_info_for_combine,
        })
    if need_extra_args:
        stage3_kwargs.update({
            "tp_send_counts": tp_recv_counts,
            "group_tp": moe_all_to_all_group_name,
            "tp_world_size": 1,
            "tp_rank_id": 0,
        })
    if a3_need_extra_args and enable_dispatch_v2:
        stage3_kwargs.update({
            "x_active_mask": mc2_mask,
        })
    kwargs_mc2.update(stage3_kwargs)

    hidden_states = torch_npu.npu_moe_distribute_combine_v2(
        **kwargs_mc2
    ) if enable_dispatch_v2 else torch_npu.npu_moe_distribute_combine(
        **kwargs_mc2)

    if shared_experts is None:
        return hidden_states
    else:
        with npu_stream_switch("moe_secondary", 0):
            npu_wait_tensor(shared_act, down_out_list)
            shared_output, _ = shared_experts.down_proj(
                (shared_act, swiglu_out_scale))
        return hidden_states, shared_output


# currently expert parallelism implemented with all2all
# is under-optimized.
def fused_experts_with_all2all(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    expert_map: torch.Tensor = None,
    ep_group: GroupCoordinator = None,
    log2phy: torch.Tensor = None,
    global_redundant_expert_num: int = 0,
    w1_scale_bias: torch.Tensor = None,
    w2_scale_bias: torch.Tensor = None,
    fusion_mlp: bool = False,
):
    if log2phy is not None:
        topk_ids = log2phy[topk_ids]
    original_shape = hidden_states.shape
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    num_tokens, _ = hidden_states.shape
    num_experts = w1.shape[0]

    if expert_map is not None:
        global_num_experts = len(expert_map) + global_redundant_expert_num
        if hasattr(torch_npu, "npu_moe_init_routing_quant"):
            quantized_tokens, expanded_row_idx, global_expert_tokens, _, token_scales = torch_npu.npu_moe_init_routing_quant(
                hidden_states,
                expert_idx=topk_ids.to(torch.int32),
                active_num=0,
                expert_capacity=0,
                expert_num=global_num_experts,
                drop_pad_mode=0,
                expert_tokens_num_mode=2,
                expert_tokens_before_capacity_flag=False,
                quant_mode=1,
            )
        else:
            quantized_tokens, expanded_row_idx, global_expert_tokens, token_scales = init_routing_quant(
                hidden_states, top_k, topk_ids, global_num_experts)

        gather_sizes = global_expert_tokens.new_empty(
            global_expert_tokens.shape[0])
        dist.all_to_all_single(gather_sizes, global_expert_tokens)

        token_counts_combined = torch.stack(
            [gather_sizes, global_expert_tokens], dim=0)
        token_counts_combined = token_counts_combined.view(
            2, ep_group.world_size, -1).sum(dim=2)
        token_counts_combined_cpu = token_counts_combined.to(
            torch.device("cpu"), non_blocking=True).numpy()
        all_tokens = gather_sizes.sum()

        gathered_tokens = quantized_tokens.new_empty(all_tokens.item(),
                                                     quantized_tokens.shape[1])
        dynamic_scale = token_scales.new_empty(gathered_tokens.shape[0])
        gather_size_list = token_counts_combined_cpu[1]
        scatter_size_list = token_counts_combined_cpu[0]

        dist.all_to_all_single(gathered_tokens, quantized_tokens,
                               scatter_size_list, gather_size_list)
        dist.all_to_all_single(dynamic_scale, token_scales, scatter_size_list,
                               gather_size_list)

        hidden_states, dynamic_scale, inverse_indices, expert_tokens = torch_npu.npu_moe_re_routing(
            gathered_tokens,
            gather_sizes.view(ep_group.world_size, -1),
            per_token_scales=dynamic_scale)
        expert_tokens = expert_tokens.to(torch.int64)
        group_list_type = 1
    else:
        row_idx_len = num_tokens * top_k
        row_idx = torch.arange(0,
                               row_idx_len,
                               dtype=torch.int32,
                               device=topk_weights.device).view(
                                   top_k, -1).permute(1, 0).contiguous()
        hidden_states, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=topk_ids,
            active_num=num_tokens)

        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts)
        expert_tokens = expert_tokens.to(torch.int64)
        group_list_type = 0
        dynamic_scale = None

    # `hidden_states` will be disposed in the `apply_mlp` function
    hidden_states = apply_mlp(
        hidden_states,
        w1,
        w1_scale,  #17
        w2,
        w2_scale,
        expert_tokens,  #16
        dynamic_scale=dynamic_scale,
        group_list_type=group_list_type,
        w1_scale_bias=w1_scale_bias,
        w2_scale_bias=w2_scale_bias,
        fusion=fusion_mlp)

    if expert_map is not None:
        reordered_outputs = torch.index_select(
            hidden_states,
            dim=0,
            # Workaround: Convert to float so that argsort runs on AI Core instead of slower AICPU
            index=inverse_indices.to(torch.float32).argsort().to(torch.int32))

        hidden_states = reordered_outputs.new_empty(*quantized_tokens.shape)
        dist.all_to_all_single(hidden_states, reordered_outputs,
                               gather_size_list, scatter_size_list)

        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None,
            drop_pad_mode=2)
    else:
        # TODO: Reorder device memory 2 times here, replace the current
        # implementation here when suitable operators become available.
        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
        )
    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


def fused_experts_with_allgather(hidden_states: torch.Tensor,
                                 w1: torch.Tensor,
                                 w1_scale: torch.Tensor,
                                 w2: torch.Tensor,
                                 w2_scale: torch.Tensor,
                                 topk_weights: torch.Tensor,
                                 topk_ids: torch.Tensor,
                                 top_k: int,
                                 expert_map: torch.Tensor = None,
                                 fusion_mlp: bool = False):
    original_shape = hidden_states.shape
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    num_tokens = hidden_states.shape[0]
    batch_size, hidden_size = hidden_states.shape
    topk_weights = topk_weights.to(hidden_states.dtype)

    ep_group = get_ep_group().device_group
    ep_rank = torch.distributed.get_rank(group=ep_group)
    ep_size = torch.distributed.get_world_size(ep_group)

    global_num_experts = len(expert_map)
    local_num_experts = global_num_experts // ep_size

    hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)

    hidden_states, expanded_x_idx, expert_tokens, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
        hidden_states,
        topk_ids,
        scale=pertoken_scale,
        offset=None,
        active_num=num_tokens * top_k,
        expert_num=global_num_experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[
            ep_rank * local_num_experts, (ep_rank + 1) * local_num_experts
        ],
        quant_mode=-1,
        row_idx_type=1)
    group_list_type = 1

    sorted_topk_weight = torch.index_select(topk_weights.view(-1), 0,
                                            expanded_x_idx)
    row_index = expanded_x_idx // topk_ids.shape[-1]
    row_index = row_index.to(torch.int64)
    share_input = torch.zeros((batch_size, hidden_size),
                              dtype=torch.bfloat16,
                              device="npu")

    if fusion_mlp:
        hidden_states, pertoken_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
            x=hidden_states,
            weight=w1,
            group_list=cumsum_group_list(group_list=expert_tokens,
                                         group_list_type=group_list_type,
                                         active_num=num_tokens * top_k,
                                         expert_num=global_num_experts),
            weight_scale=w1_scale,
            x_scale=pertoken_scale)
    else:
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w1],
            split_item=3,
            group_list_type=group_list_type,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=torch.int32)[0]
        hidden_states, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            x=hidden_states,
            weight_scale=w1_scale,
            activation_scale=pertoken_scale,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=expert_tokens,
            activate_left=True,
            quant_mode=1,
        )

    final_hidden_states = torch_npu.npu_grouped_matmul_finalize_routing(
        hidden_states,
        w2,
        scale=w2_scale.to(torch.float32),
        bias=None,
        pertoken_scale=pertoken_scale.view(-1),
        group_list=expert_tokens,
        shared_input=share_input,
        logit=sorted_topk_weight.to(torch.float32),
        row_index=row_index,
        output_bs=batch_size).to(torch.bfloat16)

    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)

    return final_hidden_states


def fused_experts(hidden_states: torch.Tensor,
                  w1: torch.Tensor,
                  w1_scale: torch.Tensor,
                  w2: torch.Tensor,
                  w2_scale: torch.Tensor,
                  topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor,
                  top_k: int,
                  expert_map: torch.Tensor = None,
                  fusion_mlp: bool = False):
    original_shape = hidden_states.shape
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    num_tokens, _ = hidden_states.shape
    num_experts = w1.shape[0]
    dtype = hidden_states.dtype
    device = hidden_states.device

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
        expert_tokens = token_counts[:num_experts]
        # Rearrange hidden_states
        hidden_states = hidden_states[sorted_token_indices]
        group_list_type = 1
    else:
        row_idx_len = num_tokens * top_k
        row_idx = torch.arange(0,
                               row_idx_len,
                               dtype=torch.int32,
                               device=topk_weights.device).view(
                                   top_k, -1).permute(1, 0).contiguous()
        hidden_states, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=topk_ids,
            active_num=num_tokens)

        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts)
        expert_tokens = expert_tokens.to(torch.int64)
        group_list_type = 0

    # `hidden_states` will be disposed in the `apply_mlp` function
    hidden_states = apply_mlp(hidden_states,
                              w1,
                              w1_scale,
                              w2,
                              w2_scale,
                              expert_tokens,
                              group_list_type=group_list_type,
                              fusion=fusion_mlp)

    if expert_map is not None:
        hidden_states.mul_(sorted_weights.unsqueeze(1))
        final_hidden_states = torch.zeros(*original_shape,
                                          device=device,
                                          dtype=dtype)

        num_valid_tokens = mask.sum()
        valid_token_mask = torch.arange(
            0, sorted_token_indices.shape[0],
            device=device).unsqueeze(1) < num_valid_tokens
        hidden_states = hidden_states.masked_fill_(~valid_token_mask,
                                                   0).to(dtype)
        final_hidden_states.index_add_(0, sorted_token_indices, hidden_states)
    else:
        # TODO: Reorder device memory 2 times here, replace the current
        # implementation here when suitable operators become available.
        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
        )

    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states
