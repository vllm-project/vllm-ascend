# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Fused MoE kernel."""
from typing import Optional, Tuple
import torch_npu
import torch
import torchair as tng
import numpy as np
import torch.distributed as dist

from vllm.platforms import current_platform
from vllm.distributed import get_world_group
from vllm.forward_context import get_forward_context

from vllm.distributed import (
    get_world_group,
    get_dp_group,
    get_ep_group
)
from omni.models.common.config.model_config import model_extra_config


def fused_topk(
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
):
    topk_weights, topk_ids, row_idx = torch_npu.npu_moe_gating_top_k_softmax(gating_output, k=topk)

    if renormalize:
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids, row_idx


# This is used by the Deepseek-V2 and Deepseek-V3 model
def grouped_topk(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        num_expert_group: int = 0,
        topk_group: int = 0,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None
):
    gating_output = gating_output.float()
    # scores = torch.softmax(gating_output, dim=-1)
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias.unsqueeze(0)
    num_token = scores.shape[0]
    group_scores = scores.view(num_token, num_expert_group,
                               -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(tmp_scores,
                                        k=topk,
                                        dim=-1,
                                        sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    topk_ids = topk_ids.int()
    # adapt add row_idx
    row_idx = torch.arange(topk_ids.numel(), device=topk_ids.device, dtype=topk_ids.dtype)
    row_idx = row_idx.reshape(topk_ids.shape[1], topk_ids.shape[0]).transpose(1, 0).contiguous()
    # adapt end

    return topk_weights, topk_ids, row_idx


def fused_experts_allgather_ep(hidden_states: torch.Tensor,
                               w1: Tuple,
                               w2: Tuple,
                               topk_weights: torch.Tensor,
                               topk_ids: torch.Tensor,
                               row_idx: torch.Tensor,
                               warm_up: bool,
                               n_routed_experts: int,
                               local_expert_indices: list,
                               best_expert_tokens: Optional[torch.Tensor] = None
                               ):
    expert_parallel_size = get_ep_group().world_size

    if expert_parallel_size > 1:
        num_tokens, hidden_size = hidden_states.shape

        if warm_up:
            topk = topk_ids.shape[1]
            sorted_tokens = torch.randn(topk * num_tokens, hidden_size, dtype=hidden_states.dtype,
                                        device=hidden_states.device)
            expanded_src_to_dst_row = row_idx.reshape(-1)
            expanded_expert_idx = torch.zeros_like(topk_ids).reshape(-1).to(torch.int32)

            expert_tokens = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx, n_routed_experts)
            expert_tokens = expert_tokens.to(torch.int64)

            gate_up_proj = torch_npu.npu_grouped_matmul(
                x=[sorted_tokens],
                weight=w1,
                group_list_type=0,
                group_type=0,
                group_list=expert_tokens,
            )

            gate_up_proj = torch_npu.npu_swiglu(gate_up_proj)

            out = torch_npu.npu_grouped_matmul(
                x=[gate_up_proj],
                weight=w2,
                group_list_type=0,
                group_type=0,
                group_list=expert_tokens,
            )

            output = torch_npu.npu_moe_finalize_routing(
                out,
                skip1=None,
                skip2=None,
                bias=None,
                scales=topk_weights,
                expanded_src_to_dst_row=expanded_src_to_dst_row,
                export_for_source_row=topk_ids,
            )
        else:
            global_local_mask = (topk_ids >= local_expert_indices[0]) & \
                                (topk_ids <= local_expert_indices[-1])
            non_global_local_mask = (~global_local_mask).to(torch.int32)
            global_local_mask = global_local_mask.to(torch.int32)

            topk_ids -= get_ep_group().rank_in_group * n_routed_experts
            local_topk_ids_mask_with_max = topk_ids * global_local_mask + non_global_local_mask * n_routed_experts

            sorted_tokens, expanded_src_to_dst_row, expanded_expert_idx = torch_npu.npu_moe_init_routing(
                hidden_states,
                row_idx=row_idx,
                expert_idx=local_topk_ids_mask_with_max,
                active_num=_MAX_NUM_TOKEN)

            if expanded_expert_idx.shape[0] > 8192:
                expert_tokens = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx, n_routed_experts + 1)
                expert_tokens = expert_tokens[:-1]
            else:
                expert_tokens = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx, n_routed_experts)

            if best_expert_tokens is not None:
                expert_tokens = best_expert_tokens

            gate_up_proj = torch_npu.npu_grouped_matmul(
                x=[sorted_tokens],
                weight=w1,
                group_list_type=0,
                group_type=0,
                group_list=expert_tokens,
            )

            gate_up_proj = torch_npu.npu_swiglu(gate_up_proj)

            out = torch_npu.npu_grouped_matmul(
                x=[gate_up_proj],
                weight=w2,
                group_list_type=0,
                group_type=0,
                group_list=expert_tokens,
            )
            if out.shape[0] < 12288:
                sorted_tokens_mask = expanded_expert_idx != n_routed_experts
                out *= sorted_tokens_mask.unsqueeze(1)
            else:
                out[expert_tokens[-1]:] = 0

            output = torch_npu.npu_moe_finalize_routing(
                out,
                skip1=None,
                skip2=None,
                bias=None,
                scales=topk_weights,
                expanded_src_to_dst_row=expanded_src_to_dst_row,
                export_for_source_row=topk_ids,
            )

        return output
    else:
        raise ValueError("expert_parallel_size should be larger than 1 if enable_moe_expert_parallel")


def fused_experts_alltoall_ep(hidden_states: torch.Tensor,
                              w1: Tuple,
                              w2: Tuple,
                              topk_weights: torch.Tensor,
                              topk_ids: torch.Tensor,
                              row_idx: torch.Tensor,
                              warm_up: bool,
                              ):
    ep_size = get_ep_group().world_size

    if ep_size > 1:
        num_tokens, hidden_size = hidden_states.shape

        num_local_experts = w1.shape[0]
        num_global_experts = num_local_experts * ep_size

        if warm_up:
            num_global_experts = num_local_experts * ep_size
            idx_all_prefill = num_tokens * topk_ids.shape[1]
            topk_idx = torch.arange(0, idx_all_prefill, dtype=torch.int32, device=current_platform.device_type).reshape(
                -1, topk_ids.shape[
                    1]) % num_global_experts
            tokens_per_expert = torch.full((num_global_experts,), idx_all_prefill / num_global_experts,
                                           dtype=torch.int32, device=current_platform.device_type)

            sorted_local_tokens, expanded_src_to_dst_row, expanded_expert_idx = torch_npu.npu_moe_init_routing(
                hidden_states,
                row_idx=row_idx,
                expert_idx=topk_idx,
                active_num=num_tokens)

            global_local_tokens = torch.zeros_like(sorted_local_tokens)

            dist.all_to_all_single(
                global_local_tokens,
                sorted_local_tokens,
            )

            global_local_indexs = torch.zeros((global_local_tokens.shape[0]), device=global_local_tokens.device,
                                              dtype=torch.int32)

            s = 0
            for i, k in enumerate(tokens_per_expert.cpu()):
                global_local_indexs[s: s + k] = i % num_local_experts
                s += k
            permuted_row_idx = torch.arange(global_local_indexs.numel(), device=global_local_indexs.device,
                                            dtype=global_local_indexs.dtype).unsqueeze(-1)

            global_local_indexs = global_local_indexs.unsqueeze(-1)

            sorted_local_tokens1, expanded_src_to_dst_row1, expanded_expert_idx1 = torch_npu.npu_moe_init_routing(
                global_local_tokens.to(current_platform.device_type),
                row_idx=permuted_row_idx,
                expert_idx=global_local_indexs,
                active_num=global_local_tokens.shape[0])

            if expanded_expert_idx.shape[0] > 8192:
                group_list = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx1, num_local_experts + 1)
                group_list = group_list[:-1]
            else:
                group_list = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx1, num_local_experts)

            if len(sorted_local_tokens1) > 0:
                gate_up_proj = torch_npu.npu_grouped_matmul(
                    x=[sorted_local_tokens1],
                    weight=w1,
                    group_list_type=0,
                    group_type=0,
                    group_list=group_list,
                )

                gate_up_proj = torch_npu.npu_swiglu(gate_up_proj)

                out = torch_npu.npu_grouped_matmul(
                    x=[gate_up_proj],
                    weight=w2,
                    group_list_type=0,
                    group_type=0,
                    group_list=group_list,
                )
                topk_weights1 = torch.ones(global_local_indexs.shape, dtype=torch.float32, device=out.device)
                new_x = torch_npu.npu_moe_finalize_routing(
                    out,
                    skip1=None,
                    skip2=None,
                    bias=None,
                    scales=topk_weights1,
                    expanded_src_to_dst_row=expanded_src_to_dst_row1,
                    export_for_source_row=global_local_indexs,
                )
            else:
                new_x = sorted_local_tokens1

            gathered_tokens = torch.zeros_like(new_x)
            dist.all_to_all_single(
                gathered_tokens,
                new_x)
            final_out = torch_npu.npu_moe_finalize_routing(
                gathered_tokens,
                skip1=None,
                skip2=None,
                bias=None,
                scales=topk_weights,
                expanded_src_to_dst_row=expanded_src_to_dst_row,
                export_for_source_row=topk_idx,
            )

        else:
            tokens_per_expert = torch.histc(topk_ids, bins=num_global_experts, min=0, max=num_global_experts - 1)

            sorted_local_tokens, expanded_src_to_dst_row, expanded_expert_idx = torch_npu.npu_moe_init_routing(
                hidden_states,
                row_idx=row_idx,
                expert_idx=topk_ids,
                active_num=num_tokens)
            sorted_local_tokens_shape = sorted_local_tokens.shape

            tokens_per_ep_rank = tokens_per_expert.view(ep_size, -1).sum(dim=1)
            tokens_per_expert_group = tokens_per_expert.new_empty(
                tokens_per_expert.shape[0]
            )
            dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)

            output_splits = tokens_per_expert_group.view(ep_size, -1).sum(1).tolist()
            global_local_tokens = sorted_local_tokens.new_empty(
                tokens_per_expert_group.sum(dim=0), sorted_local_tokens.shape[1]
            )
            input_split_sizes = tokens_per_ep_rank.tolist()

            dist.all_to_all_single(
                global_local_tokens,
                sorted_local_tokens,
                output_split_sizes=output_splits,
                input_split_sizes=input_split_sizes,
            )

            tokens_per_expert_post_gather = tokens_per_expert_group.view(
                ep_size, num_local_experts
            ).sum(dim=0)
            global_local_indexs = np.zeros(shape=(global_local_tokens.shape[0],), dtype=np.int32)

            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu()):
                global_local_indexs[s: s + k] = i % num_local_experts
                s += k

            gatherd_idxs = global_local_indexs.argsort()
            gatherd_idxs = torch.from_numpy(gatherd_idxs).npu()
            sorted_local_tokens1 = global_local_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_post_gather
            group_list = torch.cumsum(tokens_per_expert, dim=0)

            if len(sorted_local_tokens1) > 0:
                gate_up_proj = torch_npu.npu_grouped_matmul(
                    x=[sorted_local_tokens1],
                    weight=w1,
                    group_list_type=0,
                    group_type=0,
                    group_list=group_list,
                )

                gate_up_proj = torch_npu.npu_swiglu(gate_up_proj)
                out = torch_npu.npu_grouped_matmul(
                    x=[gate_up_proj],
                    weight=w2,
                    group_list_type=0,
                    group_type=0,
                    group_list=group_list,
                )


            else:
                out = sorted_local_tokens1

            new_x = torch.empty_like(out)
            new_x[gatherd_idxs] = out
            gathered_tokens = new_x.new_empty(*sorted_local_tokens_shape)
            dist.all_to_all_single(
                gathered_tokens,
                new_x,
                output_split_sizes=input_split_sizes,
                input_split_sizes=output_splits,
            )
            final_out = torch_npu.npu_moe_finalize_routing(
                gathered_tokens,
                skip1=None,
                skip2=None,
                bias=None,
                scales=topk_weights,
                expanded_src_to_dst_row=expanded_src_to_dst_row,
                export_for_source_row=topk_ids,
            )
        return final_out
    else:
        raise ValueError("expert_parallel_size should be larger than 1 if enable_moe_expert_parallel")


def fused_experts_ep_best_alltoall(hidden_states: torch.Tensor,
                                   w1: Tuple,
                                   w2: Tuple,
                                   topk_weights: torch.Tensor,
                                   topk_ids: torch.Tensor,
                                   row_idx: torch.Tensor
                                   ):
    ep_size = get_ep_group().world_size

    num_tokens, hidden_size = hidden_states.shape

    num_local_experts = w1.shape[0]
    num_global_experts = num_local_experts * ep_size

    if num_tokens == ep_size:
        idx_all_decode = num_tokens * topk_ids.shape[1] * num_local_experts
        topk_idx = torch.arange(0, idx_all_decode, num_local_experts, dtype=torch.int32,
                                device=current_platform.device_type).reshape(-1, topk_ids.shape[1]) % num_global_experts
        tokens_per_expert = torch.zeros([num_global_experts], device=current_platform.device_type,
                                        dtype=torch.int32).reshape(ep_size, num_local_experts)
        tokens_per_expert[:, 0] = topk_ids.shape[1]
        tokens_per_expert = tokens_per_expert.view(-1)
    else:
        idx_all_prefill = num_tokens * topk_ids.shape[1]
        topk_idx = torch.arange(0, idx_all_prefill, dtype=torch.int32, device=current_platform.device_type).reshape(-1,
                                                                                                                    topk_ids.shape[
                                                                                                                        1]) % num_global_experts
        tokens_per_expert = torch.full((num_global_experts,), idx_all_prefill / num_global_experts, dtype=torch.int32,
                                       device=current_platform.device_type)

    sorted_local_tokens, expanded_src_to_dst_row, expanded_expert_idx = torch_npu.npu_moe_init_routing(
        hidden_states,
        row_idx=row_idx,
        expert_idx=topk_idx,
        active_num=num_tokens)

    global_local_tokens = torch.zeros_like(sorted_local_tokens)

    dist.all_to_all_single(
        global_local_tokens,
        sorted_local_tokens,
    )

    global_local_indexs = np.zeros(shape=(global_local_tokens.shape[0],), dtype=np.int32)

    s = 0
    for i, k in enumerate(tokens_per_expert.cpu()):
        global_local_indexs[s: s + k] = i % num_local_experts
        s += k

    tokens_per_expert_post_gather = tokens_per_expert.view(
        ep_size, num_local_experts
    ).sum(dim=0)
    gatherd_idxs = global_local_indexs.argsort()
    sorted_local_tokens1 = global_local_tokens[gatherd_idxs]

    group_list = torch.cumsum(tokens_per_expert_post_gather, dim=0)

    if len(sorted_local_tokens1) > 0:
        gate_up_proj = torch_npu.npu_grouped_matmul(
            x=[sorted_local_tokens1],
            weight=w1,
            group_list_type=0,
            group_type=0,
            group_list=group_list,
        )

        gate_up_proj = torch_npu.npu_swiglu(gate_up_proj)
        out = torch_npu.npu_grouped_matmul(
            x=[gate_up_proj],
            weight=w2,
            group_list_type=0,
            group_type=0,
            group_list=group_list,
        )
    else:
        out = sorted_local_tokens1
    new_x = torch.empty_like(out)
    new_x[gatherd_idxs] = out
    gathered_tokens = torch.zeros_like(new_x)
    dist.all_to_all_single(
        gathered_tokens,
        new_x)
    outs = gathered_tokens

    final_out = torch_npu.npu_moe_finalize_routing(
        outs,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_src_to_dst_row,
        export_for_source_row=topk_idx,
    )
    return final_out


def fused_experts_allgather_ep_a3(layer: torch.nn.Module,
                                  hidden_states: torch.Tensor,
                                  pertoken_scale: torch.Tensor,
                                  topk_weights: torch.Tensor,
                                  topk_ids: torch.Tensor,
                                  n_routed_experts: int,
                                  is_prefill: bool,
                                  max_num_deployed_expert_per_rank: int  # ENABLE_OMNI_PLANNER
                                  ):
    redundancy_shared_expert_num = model_extra_config.parall_config.redundancy_shared_expert_num
    expert_parallel_size = get_ep_group().world_size - redundancy_shared_expert_num

    if expert_parallel_size > 1:
        batch_size, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        n_total_expert = n_routed_experts * expert_parallel_size

        experts_start_idx = (
                                    get_world_group().rank_in_group - redundancy_shared_expert_num) * max_num_deployed_expert_per_rank  # ENABLE_OMNI_PLANNER
        experts_end_idx = experts_start_idx + n_routed_experts
        expert_range = [experts_start_idx, experts_end_idx]

        sorted_tokens, expanded_x_idx, expert_tokens, dynamic_quant_scale = torch_npu.npu_moe_init_routing_v2(
            hidden_states, topk_ids, scale=pertoken_scale, offset=None, active_num=topk_ids.numel(), expert_capacity=-1,
            expert_num=n_total_expert, drop_pad_mode=0, expert_tokens_num_type=1, expert_tokens_num_flag=True,
            quant_mode=-1, active_expert_range=expert_range, row_idx_type=1)

        if is_prefill or not model_extra_config.operator_opt_config.enable_kv_rmsnorm_rope_cache:
            sorted_topk_weight = torch.index_select(topk_weights.reshape(-1), 0, expanded_x_idx)
            row_index = expanded_x_idx // topk_ids.shape[-1]
            row_index = row_index.to(torch.int64)
            share_input = torch.zeros((batch_size // get_dp_group().world_size, hidden_size), dtype=torch.bfloat16,
                                      device=current_platform.device_type)
        else:
            with tng.scope.npu_stream_switch('11'):
                expanded_x_idx = tng.scope.npu_wait_tensor(expanded_x_idx, expanded_x_idx)
                sorted_topk_weight = torch.index_select(topk_weights.reshape(-1), 0, expanded_x_idx)
                row_index = expanded_x_idx // topk_ids.shape[-1]
                row_index = row_index.to(torch.int64)
                share_input = torch.zeros((batch_size // get_dp_group().world_size, hidden_size), dtype=torch.bfloat16,
                                          device=current_platform.device_type)
        if layer.weight_num_bits == 8:
            scale_2 = torch.ones((n_routed_experts, layer.w13_weight_scale.shape[-1] // 2), dtype=torch.float32,
                                 device=current_platform.device_type)
            gate_up_proj = torch_npu.npu_grouped_matmul([sorted_tokens], [layer.w13_weight], bias=None,
                                                        group_list=expert_tokens,
                                                        split_item=3, output_dtype=torch.int32, group_type=0,
                                                        group_list_type=1)[0]
            gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(gate_up_proj,
                                                                              weight_scale=layer.w13_weight_scale,
                                                                              activation_scale=dynamic_quant_scale,
                                                                              bias=None, quant_scale=scale_2,
                                                                              quant_offset=None,
                                                                              group_index=expert_tokens,
                                                                              activate_left=True, quant_mode=1)

            output = torch_npu.npu_grouped_matmul_finalize_routing(gate_up_proj, layer.w2_weight, expert_tokens,
                                                                   scale=layer.w2_weight_scale.to(torch.float),
                                                                   bias=None,
                                                                   pertoken_scale=pertoken_scale,
                                                                   shared_input=share_input,
                                                                   logit=sorted_topk_weight,
                                                                   row_index=row_index,
                                                                   output_bs=batch_size,
                                                                   shared_input_weight=1.0,
                                                                   group_list_type=1, shared_input_offset=0).to(
                torch.bfloat16)
            return output
        elif layer.weight_num_bits == 4:
            gate_up_proj = \
            torch_npu.npu_grouped_matmul([sorted_tokens], [layer.w13_weight], bias=[layer.w13_weight_bias],
                                         scale=[layer.w13_weight_int4_scale],
                                         offset=None, antiquant_scale=None, antiquant_offset=None,
                                         per_token_scale=[dynamic_quant_scale],
                                         group_list=expert_tokens,
                                         activation_input=None, activation_quant_scale=None,
                                         activation_quant_offset=None, split_item=3, group_type=0,
                                         group_list_type=1, act_type=0, output_dtype=torch.bfloat16)[0]

            fake_scale = torch.ones(layer.w13_weight_int4_scale.shape, dtype=torch.float32, device="npu").view(
                -1, layer.w13_weight_int4_scale.shape[-1])
            dynamic_quant_scale = torch.ones(dynamic_quant_scale.shape, dtype=torch.float32, device="npu")
            scale_2 = torch.ones((n_routed_experts, layer.w13_weight_int4_scale.shape[-1] // 2),
                                 dtype=torch.float32, device="npu")
            gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(gate_up_proj,
                                                                              weight_scale=fake_scale,
                                                                              activation_scale=dynamic_quant_scale,
                                                                              bias=None, quant_scale=scale_2,
                                                                              quant_offset=None,
                                                                              group_index=expert_tokens,
                                                                              activate_left=True, quant_mode=1)

            output = torch_npu.npu_grouped_matmul_finalize_routing(gate_up_proj, layer.w2_weight,
                                                                   group_list=expert_tokens,
                                                                   scale=layer.w2_weight_int4_scale,
                                                                   bias=layer.w2_weight_bias,
                                                                   pertoken_scale=pertoken_scale,
                                                                   shared_input=share_input,
                                                                   logit=sorted_topk_weight,
                                                                   row_index=row_index,
                                                                   output_bs=batch_size, group_list_type=1).to(
                torch.bfloat16)
            return output


def gmm_expert(layer, x, expert_tokens, dynamic_scale=None, avg_tokens_per_expert=None):
    # no need to transpose weight here if weight_nz enabled
    hidden_size = x.size(-1)
    h = x
    pertoken_scale = dynamic_scale

    if pertoken_scale.dim() > 1:
        pertoken_scale = pertoken_scale.reshape(-1)
        h = h.view(-1, hidden_size)
    # gmm1: gate_up
    avg_tokens_per_expert = avg_tokens_per_expert or [0]

    if layer.weight_num_bits == 8:
        mm1_mm3 = torch_npu.npu_grouped_matmul([h], [layer.w13_weight],
                                               group_list=expert_tokens, split_item=3,
                                               output_dtype=torch.int32, group_type=0,
                                               group_list_type=1, tuning_config=avg_tokens_per_expert)[0]
        # dequant_swiglu_quant
        scale_2 = torch.ones((expert_tokens.shape[0], layer.w13_weight_scale.shape[-1] // 2), dtype=torch.float32,
                             device=current_platform.device_type)
        intermediate_h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            mm1_mm3, weight_scale=layer.w13_weight_scale,
            activation_scale=pertoken_scale.squeeze(0), bias=None, quant_scale=scale_2, quant_offset=None,
            group_index=expert_tokens, activate_left=True, quant_mode=1)

        if pertoken_scale.dim() > 1:
            inter_size = intermediate_h.size(-1)
            pertoken_scale = pertoken_scale.reshape(-1)
            intermediate_h = intermediate_h.view(-1, inter_size)
        # gmm2: down
        out_dtype = torch.bfloat16
        w2_scale = layer.w2_weight_scale.to(torch.bfloat16)
        out_hidden = torch_npu.npu_grouped_matmul([intermediate_h], [layer.w2_weight], bias=None,
                                                  scale=[w2_scale], per_token_scale=[pertoken_scale],
                                                  group_list=expert_tokens, split_item=3,
                                                  output_dtype=out_dtype, group_type=0,
                                                  group_list_type=1, tuning_config=avg_tokens_per_expert)[0]
        return out_hidden
    elif layer.weight_num_bits == 4:
        mm1_mm3 = torch_npu.npu_grouped_matmul([h], [layer.w13_weight], bias=[layer.w13_weight_bias], scale=[layer.w13_weight_int4_scale],
                                               offset=None, antiquant_scale=None, antiquant_offset=None,
                                               per_token_scale=[pertoken_scale],
                                               group_list=expert_tokens,
                                               activation_input=None, activation_quant_scale=None,
                                               activation_quant_offset=None, split_item=3, group_type=0,
                                               group_list_type=1, act_type=0, output_dtype=torch.bfloat16)[0]

        fake_scale = torch.ones(layer.w13_weight_int4_scale.shape, dtype=torch.float32, device="npu").view(-1, layer.w13_weight_int4_scale.shape[-1])
        pertoken_scale = torch.ones(pertoken_scale.shape, dtype=torch.float32, device="npu")
        intermediate_h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(mm1_mm3, weight_scale=fake_scale,
                                                                            activation_scale=pertoken_scale,
                                                                            bias=None, quant_scale=None,
                                                                            quant_offset=None,
                                                                            group_index=expert_tokens,
                                                                            activate_left=True, quant_mode=1)

        if pertoken_scale.dim() > 1:
            inter_size = intermediate_h.size(-1)
            pertoken_scale = pertoken_scale.reshape(-1)
            intermediate_h = intermediate_h.view(-1, inter_size)

        out_dtype = torch.bfloat16
        out_hidden = torch_npu.npu_grouped_matmul([intermediate_h], [layer.w2_weight], bias=[layer.w2_weight_bias],
                                                  scale=[layer.w2_weight_int4_scale], per_token_scale=[pertoken_scale],
                                                  group_list=expert_tokens, split_item=3,
                                                  output_dtype=out_dtype, group_type=0,
                                                  group_list_type=1, tuning_config=avg_tokens_per_expert)[0]
        return out_hidden
    else:
        raise NotImplementedError(f"Unsupported compress tensor type. num bits: {layer.weight_num_bits}")


def moe_infer_fusion(
        layer,
        x,
        topk_ids,
        topk_weight,
        warm_up=False,
        is_prefill=True,
        comm_group=None,
):
    _, h = x.shape
    hidden_states = x.view(-1, h)
    topk_weight = topk_weight.to(x.dtype)
    max_num_deployed_expert = layer.w13_weight.shape[0] * get_ep_group().world_size
    if warm_up:
        # This is forced balancing, the goal is to reduce peak memory
        global_rank = get_world_group().rank_in_group
        step = hidden_states.shape[0] * 8  # topk 8 expert
        cur_topk_list = [
            (i + global_rank // 1) % max_num_deployed_expert for i in range(
                global_rank // 1 * step, (global_rank // 1 + 1) * step)]
        topk_ids = torch.Tensor(cur_topk_list).int().view(hidden_states.shape[0], -1).npu()
    else:
        topk_ids = topk_ids.int()

    expert_range = [0, max_num_deployed_expert]
    expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
        hidden_states,
        expert_idx=topk_ids,
        scale=None,
        expert_num=max_num_deployed_expert,
        active_expert_range=expert_range,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_num=topk_ids.numel(),
        drop_pad_mode=0,
        row_idx_type=0,
        quant_mode=1)

    tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
    group = comm_group.device_group if comm_group else None
    dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=group)

    # combine tensors, do reduceSum and D2H toghter
    combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
    # view: EP, E//EP
    # sum: EP, the number of tokens each rank receives from other cards
    ep_size = get_ep_group().world_size
    combine_tokens = combine_tokens.view(2, ep_size, -1).sum(2)
    all_tokens = combine_tokens[0].sum()

    # prune
    if model_extra_config.operator_opt_config.experts_pruning:
        all_input_tokens = combine_tokens[1].sum().item()

    combine_tokens_cpu = combine_tokens.cpu().tolist()
    # alltoall input splits, the total number of tokens routed from the current rank to other ranks
    input_splits = combine_tokens_cpu[1]
    # alltoall output splits, the number of tokens each rank receives from other cards
    output_splits = combine_tokens_cpu[0]
    # alltoall output, unfolded into one dimension, the size is the sum of the number of tokens routed from other cards to the current rank.
    gathered_tokens = expanded_x.new_empty(
        all_tokens.item(), expanded_x.shape[1]
    )
    gathered_pertoken_scale = pertoken_scale.new_empty(gathered_tokens.shape[0])

    if model_extra_config.operator_opt_config.experts_pruning:
        x_view=torch.narrow(expanded_x, 0, 0,all_input_tokens)
        scale_view = torch.narrow(pertoken_scale, 0, 0, all_input_tokens)
        dist.all_to_all_single(
            gathered_tokens, x_view, output_splits, input_splits, group=group)
        dist.all_to_all_single(
            gathered_pertoken_scale,
            scale_view, 
            output_splits, 
            input_splits, 
            group=group)
        
    else:
        dist.all_to_all_single(
            gathered_tokens, 
            expanded_x, 
            output_splits, 
            input_splits, 
            group=group)
        dist.all_to_all_single(
            gathered_pertoken_scale, 
            pertoken_scale, 
            output_splits, 
            input_splits, 
            group=group)
    # reroute
    # Tokens merged by experts, scales merged by experts, indices for FinalizeRouting, number of tokens processed by each expert
    hidden_states_sorted_by_experts, gathered_pertoken_scale, gathered_idxs_unsort, tokens_per_local_expert = torch_npu.npu_moe_re_routing(
        gathered_tokens,
        tokens_per_expert_group.view(ep_size, -1),
        per_token_scales=gathered_pertoken_scale
    )
    group_list = tokens_per_local_expert.to(torch.int64)
    if model_extra_config.operator_opt_config.use_omni_placement:
        layer.planner.record_activation(layer.moe_layer_idx, group_list,
                                        support_multi_stream=model_extra_config.operator_opt_config.moe_multi_stream_tune and (
                                            not is_prefill))

    hidden_states_ordered_by_experts = gmm_expert(layer, hidden_states_sorted_by_experts, tokens_per_local_expert.to(torch.int64), gathered_pertoken_scale, None)

    new_x = torch.index_select(hidden_states_ordered_by_experts, 0,
                               gathered_idxs_unsort.to(torch.float32).argsort().to(torch.int32))
    
    # prune 3
    if model_extra_config.operator_opt_config.experts_pruning:
        gathered_tokens = new_x.new_zeros(*expanded_x.shape)
        gathered_view = torch.narrow(gathered_tokens, 0, 0, all_input_tokens)
        dist.all_to_all_single(gathered_view, 
                               new_x, 
                               input_splits, 
                               output_splits, 
                               group=group)
        expanded_row_idx[expanded_row_idx==-1] = torch.arange(
            all_input_tokens, expanded_row_idx.shape[0],
            device=expanded_row_idx.device,   # 同设备
            dtype=expanded_row_idx.dtype  )
    else:
        gathered_tokens = new_x.new_empty(*expanded_x.shape)
        dist.all_to_all_single(gathered_tokens, 
                               new_x, 
                               input_splits, 
                               output_splits, 
                               group=group)
    return hidden_states, gathered_tokens, topk_weight, expanded_row_idx


def moe_expert_quant_forward(layer, sorted_tokens, expert_tokens, act_dtype, dynamic_scale=None):
    if layer.quant_mode:
        pertoken_scale = dynamic_scale
    else:
        sorted_tokens, pertoken_scale = torch_npu.npu_dynamic_quant(sorted_tokens)

    if layer.weight_num_bits == 8:

        gate_up_proj = torch_npu.npu_grouped_matmul([sorted_tokens], [layer.w13_weight], bias=None, group_list=expert_tokens,
                                         split_item=3, output_dtype=torch.int32, group_type=0, group_list_type=1)[0]

        scale_2 = torch.ones((len(layer.w13_weight), layer.w13_weight_scale.shape[-1] // 2), dtype=torch.float32,
                             device=current_platform.device_type)
        gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            gate_up_proj, weight_scale=layer.w13_weight_scale, activation_scale=pertoken_scale, bias=None,
            quant_scale=scale_2, quant_offset=None,
            group_index=expert_tokens, activate_left=True, quant_mode=1)

        if not model_extra_config.operator_opt_config.opt_w2_scale_cast:
            w2_scale = layer.w2_weight_scale.to(torch.bfloat16)
        else:
            w2_scale = layer.w2_weight_scale

        out = torch_npu.npu_grouped_matmul([gate_up_proj], [layer.w2_weight], scale=[w2_scale],
                                           per_token_scale=[pertoken_scale], bias=None,
                                           group_list=expert_tokens, split_item=3, output_dtype=act_dtype,
                                           group_type=0,
                                           group_list_type=1)[0]
        return out
    elif layer.weight_num_bits == 4:
        gate_up_proj = torch_npu.npu_grouped_matmul([sorted_tokens], [layer.w13_weight], bias=[layer.w13_weight_bias], scale=[layer.w13_weight_int4_scale],
                                                    offset=None, antiquant_scale=None, antiquant_offset=None,
                                                    per_token_scale=[pertoken_scale],
                                                    group_list=expert_tokens,
                                                    activation_input=None, activation_quant_scale=None,
                                                    activation_quant_offset=None, split_item=3, group_type=0,
                                                    group_list_type=1, act_type=0,
                                                    tuning_config=model_extra_config.operator_opt_config.decode_gear_list[
                                                                  0:], output_dtype=torch.bfloat16)[0]

        fake_scale = torch.ones(layer.w13_weight_int4_scale.shape, dtype=torch.float32, device="npu").view(-1, layer.w13_weight_int4_scale.shape[-1])
        pertoken_scale = torch.ones(pertoken_scale.shape, dtype=torch.float32, device="npu")
        gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(gate_up_proj, weight_scale=fake_scale,
                                                                          activation_scale=pertoken_scale,
                                                                          bias=None, quant_scale=None,
                                                                          quant_offset=None, group_index=expert_tokens,
                                                                          activate_left=True, quant_mode=1)

        out = torch_npu.npu_grouped_matmul([gate_up_proj], [layer.w2_weight], scale=[layer.w2_weight_int4_scale],
                                           per_token_scale=[pertoken_scale], bias=[layer.w2_weight_bias],
                                           group_list=expert_tokens, split_item=3, output_dtype=act_dtype,
                                           group_type=0,
                                           group_list_type=1,
                                           tuning_config=model_extra_config.operator_opt_config.decode_gear_list[0:])[0]
        return out
    else:
        raise NotImplementedError(f"Unsupported compress tensor type. num bits: {layer.weight_num_bits}")


def fused_experts_moe_dispatch_combine(layer: torch.nn.Module,
                                            hidden_states: torch.Tensor,
                                            topk_weights: torch.Tensor,
                                            topk_ids: torch.Tensor,
                                            max_num_deployed_expert: int,
                                            is_prefill: bool,
                                            is_route_expert: bool,
                                            ):
    expert_parallel_size = get_ep_group().world_size

    if expert_parallel_size > 1:
        # For vllm v1, metadata is a dict {layer_name: metadata}
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[next(iter(attn_metadata))]
        mc2_mask = attn_metadata.decode.mc2_mask if attn_metadata is not None else None
        global_bs = 0
        act_dtype = hidden_states.dtype
        # route
        shared_expert_rank_num = model_extra_config.parall_config.redundancy_shared_expert_num
        kwargs = {
            "x": hidden_states,
            "expert_ids": topk_ids,  # [n*topk]
            "expert_shard_type": 0,
            "shared_expert_rank_num": shared_expert_rank_num,  # 32
            "moe_expert_num": max_num_deployed_expert,  # ENABLE_OMNI_PLANNER, 0 redundancy 256, 1 redundancy expert 320
            "global_bs": global_bs,  # 0 Default (all); all tokens can be set
        }
        experts_tp_size = layer.tp_size
        world_size = get_world_group().world_size
        # In fact, what we get is the die number, and the ep group is not adapted by default.
        # The default ep group is experts_num/die_num.
        global_rank = get_world_group().rank_in_group
        all_to_all_group_size = world_size // experts_tp_size

        kwargs.update({
            "scales": None,  # Quantization coefficient
            "quant_mode": layer.quant_mode,  # 0: Non-quantization; 1: Static quantization; 2: Dynamic quantization
            "group_ep": layer.moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
            "ep_world_size": all_to_all_group_size,
            "ep_rank_id": global_rank // experts_tp_size,
            "group_tp": layer.moe_rs_group_name,
            "tp_world_size": experts_tp_size,
            "tp_rank_id": global_rank % experts_tp_size,
            "x_active_mask": mc2_mask,
        })

        output = torch_npu.npu_moe_distribute_dispatch_v2(**kwargs)
        expand_x, dynamic_scale, expand_idx, expert_token_nums, ep_recv_counts = output[0:5]

        group_list = expert_token_nums.to(torch.int64)

        if model_extra_config.operator_opt_config.use_omni_placement and is_route_expert:
            layer.planner.record_activation(layer.moe_layer_idx, group_list,
                                            support_multi_stream=model_extra_config.operator_opt_config.moe_multi_stream_tune and (
                                                not is_prefill))

        if shared_expert_rank_num > 0 and global_rank // experts_tp_size < shared_expert_rank_num:
            x = {"x_int8": expand_x, "pertoken_scale": dynamic_scale}
            hidden_states_experts = layer(x)
        else:
            # cal experts
            group_list = group_list[
                         :len(layer.w13_weight)]  # Adapt to redundant and non-redundant layers, #ENABLE_OMNI_PLANNER
            hidden_states_experts = moe_expert_quant_forward(layer, expand_x, group_list, act_dtype, dynamic_scale)

        # moeCombine
        kwargs = {
            "expand_x": hidden_states_experts,
            "expert_ids": topk_ids,  # [n*topk]
            "assist_info_for_combine": expand_idx,
            "expert_scales": topk_weights.to(torch.float32),  # weight [n*topk]
            "expert_shard_type": 0,
            "shared_expert_rank_num": shared_expert_rank_num,
            "moe_expert_num": max_num_deployed_expert,  # ENABLE_OMNI_PLANNER, 0 redundancy 256, 1 redundancy expert 320
            "global_bs": global_bs,  # 0 Default (all); you can set all tokens
        }
        tp_recv_counts = output[5]
        stage3_kwargs = {
            "ep_send_counts": ep_recv_counts,  # dispatch's send_counts
            "group_ep": layer.moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
            "ep_world_size": all_to_all_group_size,
            "ep_rank_id": global_rank // experts_tp_size,
            "tp_send_counts": tp_recv_counts,
            "group_tp": layer.moe_rs_group_name,
            "tp_world_size": experts_tp_size,
            "tp_rank_id": global_rank % experts_tp_size,
            "x_active_mask": mc2_mask,
        }
        kwargs.update(stage3_kwargs)

        hidden_states_route = torch_npu.npu_moe_distribute_combine_v2(**kwargs)
    else:
        raise ValueError("ep number should be greater than 1.")
    return hidden_states_route


def static_routing(hidden_states: torch.Tensor):
    batch_size = hidden_states.size(0)
    indices = np.arange(batch_size, dtype=np.int64)
    return indices % model_extra_config.parall_config.redundancy_shared_expert_num + get_ep_group().world_size - model_extra_config.parall_config.redundancy_shared_expert_num


def shared_expert_alltoall_ep(hidden_states: torch.Tensor, expert: torch.nn.Module, warm_up: bool):
    if warm_up:
        return None
    world_size = get_ep_group().world_size
    expert_assignments = static_routing(hidden_states)
    expert_assignments_ = torch.from_numpy(expert_assignments).to(hidden_states.device)
    send_counts = torch.bincount(expert_assignments_, minlength=world_size).to(hidden_states.device)
    recv_counts = torch.zeros_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)

    sorted_indices = expert_assignments.argsort()
    sorted_data = hidden_states[sorted_indices]

    recv_data = torch.empty(
        recv_counts.sum().item(), *hidden_states.shape[1:],
        dtype=hidden_states.dtype, device=hidden_states.device
    )
    dist.all_to_all_single(
        recv_data, sorted_data,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist()
    )

    if recv_data.size(0) > 0:
        output = expert(recv_data)
    else:
        output = torch.empty((0, hidden_states.size(1)), device=hidden_states.device, dtype=hidden_states.dtype)

    send_back_counts = recv_counts.clone()
    recv_back_counts = torch.zeros_like(send_back_counts)

    dist.all_to_all_single(recv_back_counts, send_back_counts)
    recv_back_data = torch.empty(
        recv_back_counts.sum().item(), *output.shape[1:],
        dtype=output.dtype, device=output.device
    )
    dist.all_to_all_single(
        recv_back_data, output,
        output_split_sizes=recv_back_counts.tolist(),
        input_split_sizes=send_back_counts.tolist()
    )

    inverse_indices = np.argsort(sorted_indices, kind='stable')
    return recv_back_data[inverse_indices]


def fused_experts_allgather_ep_a2(layer: torch.nn.Module,
                                       hidden_states: torch.Tensor,
                                       pertoken_scale: torch.Tensor,
                                       topk_weights: torch.Tensor,
                                       topk_ids: torch.Tensor,
                                       n_routed_experts: int,
                                       is_prefill: bool,
                                       max_num_deployed_expert_per_rank: int,  # ENABLE_OMNI_PLANNER
                                       smooth_scale: torch.Tensor
                                       ):
    expert_parallel_size = get_ep_group().world_size

    if expert_parallel_size > 1:
        batch_size, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        n_total_expert = n_routed_experts * get_ep_group().world_size

        experts_start_idx = get_world_group().rank_in_group * max_num_deployed_expert_per_rank  # ENABLE_OMNI_PLANNER
        experts_end_idx = experts_start_idx + n_routed_experts
        expert_range = [experts_start_idx, experts_end_idx]

        row_idx_type = 0 if model_extra_config.operator_opt_config.pd_seperate_prefill else 1
        sorted_tokens, expanded_x_idx, expert_tokens, dynamic_quant_scale = \
            torch_npu.npu_moe_init_routing_v2(hidden_states, topk_ids, scale=pertoken_scale, offset=None,
                                              active_num=topk_ids.numel(),
                                              expert_capacity=-1, expert_num=n_total_expert, drop_pad_mode=0,
                                              expert_tokens_num_type=1,
                                              expert_tokens_num_flag=True, quant_mode=-1,
                                              active_expert_range=expert_range, row_idx_type=row_idx_type)
        if is_prefill and model_extra_config.operator_opt_config.prefill_enable_long_seq:
            sorted_tokens = sorted_tokens[:expert_tokens.sum(), :]
            dynamic_quant_scale = dynamic_quant_scale[:expert_tokens.sum()]

        if is_prefill or not model_extra_config.operator_opt_config.enable_kv_rmsnorm_rope_cache:
            if not model_extra_config.operator_opt_config.pd_seperate_prefill:
                range1 = torch.arange(0, expanded_x_idx.shape[0], dtype=torch.int32, device="npu")
                range2 = range1 * torch.tensor(991, dtype=torch.int32, device="npu")
                mask = (range1 >= torch.sum(expert_tokens)).to(torch.int32)
                expanded_x_idx += range2 * mask
                expanded_x_idx = expanded_x_idx % expanded_x_idx.shape[0]
                expanded_x_idx = torch.clamp(expanded_x_idx, min=0, max=expanded_x_idx.shape[0] - 1)
                sorted_topk_weight = torch.index_select(topk_weights.reshape(-1), 0, expanded_x_idx)
                row_index = torch.floor(torch.div(expanded_x_idx, topk_ids.shape[-1])).to(torch.int64)
                share_input = torch.zeros((batch_size // expert_parallel_size, hidden_size), dtype=torch.bfloat16,
                                          device="npu")
        else:
            if model_extra_config.operator_opt_config.moe_multi_stream_tune:
                with tng.scope.npu_stream_switch('11'):
                    range1 = torch.arange(0, expanded_x_idx.shape[0], dtype=torch.int32, device="npu")
                    range2 = range1 * torch.tensor(991, dtype=torch.int32, device="npu")
                    mask = (range1 >= torch.sum(expert_tokens)).to(torch.int32)
                    expanded_x_idx += range2 * mask
                    expanded_x_idx = expanded_x_idx % expanded_x_idx.shape[0]
                    expanded_x_idx = torch.clamp(expanded_x_idx, min=0, max=expanded_x_idx.shape[0] - 1)
                    sorted_topk_weight = torch.index_select(topk_weights.reshape(-1), 0, expanded_x_idx)
                    row_index = torch.floor(torch.div(expanded_x_idx, topk_ids.shape[-1])).to(torch.int64)
                    share_input = torch.zeros((batch_size // expert_parallel_size, hidden_size), dtype=torch.bfloat16,
                                              device="npu")
            else:
                range1 = torch.arange(0, expanded_x_idx.shape[0], dtype=torch.int32, device="npu")
                range2 = range1 * torch.tensor(991, dtype=torch.int32, device="npu")
                mask = (range1 >= torch.sum(expert_tokens)).to(torch.int32)
                expanded_x_idx += range2 * mask
                expanded_x_idx = expanded_x_idx % expanded_x_idx.shape[0]
                expanded_x_idx = torch.clamp(expanded_x_idx, min=0, max=expanded_x_idx.shape[0] - 1)
                sorted_topk_weight = torch.index_select(topk_weights.reshape(-1), 0, expanded_x_idx)
                row_index = torch.floor(torch.div(expanded_x_idx, topk_ids.shape[-1])).to(torch.int64)
                share_input = torch.zeros((batch_size // expert_parallel_size, hidden_size), dtype=torch.bfloat16,
                                          device="npu")

        if model_extra_config.operator_opt_config.use_omni_placement:
            group_list = expert_tokens.to(torch.int64)
            layer.planner.record_activation(layer.moe_layer_idx, group_list,
                                            support_multi_stream=model_extra_config.operator_opt_config.moe_multi_stream_tune and (
                                                not is_prefill))

        if layer.weight_num_bits == 8:
            gate_up_proj = torch_npu.npu_grouped_matmul([sorted_tokens], [layer.w13_weight], bias=None, group_list=expert_tokens,
                                                        split_item=3, output_dtype=torch.int32, group_type=0,
                                                        group_list_type=1)[0]

            gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(gate_up_proj, weight_scale=layer.w13_weight_scale,
                                                                              activation_scale=dynamic_quant_scale,
                                                                              bias=None, quant_scale=smooth_scale,
                                                                              quant_offset=None, group_index=expert_tokens,
                                                                              activate_left=True, quant_mode=1)

            if model_extra_config.operator_opt_config.pd_seperate_prefill:
                out = torch_npu.npu_grouped_matmul([gate_up_proj], [layer.w2_weight], scale=[layer.w2_weight_scale],
                                                   per_token_scale=[pertoken_scale], bias=None,
                                                   group_list=expert_tokens, split_item=3, output_dtype=torch.bfloat16,
                                                   group_type=0, group_list_type=1)[0]
                output = torch_npu.npu_moe_finalize_routing(out.unsqueeze(1).to(torch.bfloat16),
                                                            None, None, None,
                                                            topk_weights.to(torch.bfloat16),
                                                            expanded_x_idx, topk_ids, drop_pad_mode=3)
            else:
                output = torch_npu.npu_grouped_matmul_finalize_routing(gate_up_proj, layer.w2_weight, scale=layer.w2_weight_scale, bias=None,
                                                                       pertoken_scale=pertoken_scale,
                                                                       group_list=expert_tokens, shared_input=share_input,
                                                                       logit=sorted_topk_weight, row_index=row_index,
                                                                       output_bs=batch_size, shared_input_weight=1.0,
                                                                       group_list_type=1,
                                                                       shared_input_offset=0)
        elif layer.weight_num_bits == 4:
            gate_up_proj = torch_npu.npu_grouped_matmul([sorted_tokens], [layer.w13_weight], bias=[layer.w13_weight_bias], scale=[layer.w13_weight_int4_scale],
                                                        offset=None, antiquant_scale=None, antiquant_offset=None,
                                                        per_token_scale=[dynamic_quant_scale],
                                                        group_list=expert_tokens,
                                                        activation_input=None, activation_quant_scale=None,
                                                        activation_quant_offset=None, split_item=3, group_type=0,
                                                        group_list_type=1, act_type=0, output_dtype=torch.bfloat16)[0]

            fake_scale = torch.ones(layer.w13_weight_int4_scale.shape, dtype=torch.float32, device="npu").view(-1, layer.w13_weight_int4_scale.shape[-1])
            dynamic_quant_scale = torch.ones(dynamic_quant_scale.shape, dtype=torch.float32, device="npu")
            scale_2 = torch.ones((n_routed_experts, layer.w13_weight_int4_scale.shape[-1] // 2), dtype=torch.float32, device="npu")
            gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(gate_up_proj, weight_scale=fake_scale,
                                                                              activation_scale=dynamic_quant_scale,
                                                                              bias=None, quant_scale=scale_2,
                                                                              quant_offset=None,
                                                                              group_index=expert_tokens,
                                                                              activate_left=True, quant_mode=1)

            output = torch_npu.npu_grouped_matmul_finalize_routing(gate_up_proj, layer.w2_weight,
                                                                   group_list=expert_tokens, scale=layer.w2_weight_int4_scale,
                                                                   bias=layer.w2_weight_bias,
                                                                   pertoken_scale=pertoken_scale,
                                                                   shared_input=share_input, logit=sorted_topk_weight,
                                                                   row_index=row_index,
                                                                   output_bs=batch_size, group_list_type=1).to(torch.bfloat16)

        if not is_prefill and (
                model_extra_config.operator_opt_config.enable_round_pipeline_comm or model_extra_config.operator_opt_config.enable_pipeline_comm):
            return output

        return output.to(torch.bfloat16)
