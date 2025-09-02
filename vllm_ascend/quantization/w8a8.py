#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

from typing import Any, Callable, Dict, Optional

import torch
import torch_npu
from vllm.attention.backends.abstract import AttentionType
from vllm.distributed.parallel_state import get_ep_group

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.moe.experts_selector import select_experts
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, is_310p


def quant_per_tensor(in_tensor: torch.Tensor,
                     input_scale: torch.Tensor,
                     input_offset: torch.Tensor,
                     function=False):
    return torch_npu.npu_quantize(in_tensor, input_scale, input_offset,
                                  torch.qint8, -1, function)


class AscendW8A8LinearMethod:
    """Linear method for Ascend W8A8.

    Args:
        w_sym: whether the linear weight is symmetrically quantized.
    """

    def __init__(self) -> None:
        # aclnn quant matmul requires to transpose matrix B, set to true by default.
        self.transpose_weight = not is_310p()

    @staticmethod
    def get_weight(
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> Dict[str, Any]:
        params_dict = {
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {}
        params_dict["input_scale"] = torch.empty(1, dtype=params_dtype)
        params_dict["input_offset"] = torch.empty(1, dtype=torch.int8)
        return params_dict

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        params_dict = {}
        params_dict["quant_bias"] = torch.empty(output_size, dtype=torch.int32)
        if params_dtype == torch.bfloat16:
            params_dict["deq_scale"] = torch.empty(output_size,
                                                   dtype=torch.float32)
        elif params_dtype == torch.float16:
            params_dict["deq_scale"] = torch.empty(output_size,
                                                   dtype=torch.int64)
        params_dict["weight_scale"] = torch.empty(output_size,
                                                  1,
                                                  dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size,
                                                   1,
                                                   dtype=params_dtype)
        return params_dict

    def get_pergroup_param(self, input_size: int, output_size: int,
                           params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        if x.dtype != torch.int8:
            x = quant_per_tensor(
                x,
                layer.aclnn_input_scale_reciprocal,
                layer.aclnn_input_offset,
            )
        quant_bias = layer.quant_bias if tp_rank == 0 else None
        if is_310p():
            # On 300I Duo platform, we need transpose again if
            # using nz. This transpose can be skipped in torchair.
            output = torch_npu.npu_quant_matmul(
                x,
                layer.weight.data.transpose(1, 0),
                layer.deq_scale,
                bias=quant_bias,
                output_dtype=layer.params_dtype,
            )
        else:
            output = torch_npu.npu_quant_matmul(
                x,
                layer.weight,
                layer.deq_scale,
                bias=quant_bias,
                output_dtype=layer.params_dtype,
            )
        return output

    def process_weights_after_loading(self, layer):
        expanding_factor = layer.weight.data.shape[1]
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor),
            requires_grad=False)
        layer.aclnn_input_scale_reciprocal = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor),
            requires_grad=False)
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor),
            requires_grad=False).to(layer.aclnn_input_scale.dtype)
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = torch_npu.npu_format_cast(layer.weight.data,
                                                      ACL_FORMAT_FRACTAL_NZ)
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        layer.weight_offset.data = torch.flatten(layer.weight_offset.data)


class AscendW8A8FusedMoEMethod:
    """FusedMoe method for Ascend W8A8.
    """

    def __init__(self):
        self.transpose_weight = True

    @staticmethod
    def get_weight(num_experts: int, intermediate_size_per_partition: int,
                   hidden_sizes: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(num_experts,
                                               2 *
                                               intermediate_size_per_partition,
                                               hidden_sizes,
                                               dtype=torch.int8,
                                               requires_grad=False)
        param_dict["w2_weight"] = torch.empty(num_experts,
                                              hidden_sizes,
                                              intermediate_size_per_partition,
                                              dtype=torch.int8,
                                              requires_grad=False)
        return param_dict

    @staticmethod
    def get_dynamic_quant_param(num_experts: int,
                                intermediate_size_per_partition: int,
                                hidden_sizes: int,
                                params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=torch.float32)
        param_dict["w13_weight_offset"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=torch.float16)
        param_dict["w2_weight_scale"] = torch.empty(num_experts,
                                                    hidden_sizes,
                                                    1,
                                                    dtype=torch.float32)
        param_dict["w2_weight_offset"] = torch.empty(num_experts,
                                                     hidden_sizes,
                                                     1,
                                                     dtype=torch.float16)
        param_dict["w2_deq_scale"] = torch.empty(num_experts,
                                                 hidden_sizes,
                                                 dtype=torch.float32)
        param_dict["w13_deq_scale"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            dtype=torch.float32)
        param_dict["w2_input_scale"] = torch.empty(num_experts,
                                                   1,
                                                   dtype=torch.float32)
        param_dict["w13_input_scale"] = torch.empty(num_experts,
                                                    1,
                                                    dtype=torch.float32)
        param_dict["w2_input_offset"] = torch.empty(num_experts,
                                                    1,
                                                    dtype=torch.int8)
        param_dict["w13_input_offset"] = torch.empty(num_experts,
                                                     1,
                                                     dtype=torch.int8)
        param_dict["quant_bias"] = torch.empty(num_experts,
                                               hidden_sizes,
                                               dtype=torch.int32)

        return param_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        shared_experts: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert router_logits.shape[
            1] == global_num_experts, "Number of global experts mismatch"

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
            global_num_experts=global_num_experts)

        if is_310p():
            return fused_experts_310p(hidden_states=x,
                                      w1=layer.w13_weight,
                                      w1_scale=layer.w13_weight_scale,
                                      w1_input_scale=layer.w13_input_scale,
                                      w2=layer.w2_weight,
                                      w2_scale=layer.w2_weight_scale,
                                      w2_input_scale=layer.w2_input_scale,
                                      topk_weights=topk_weights,
                                      topk_ids=topk_ids,
                                      top_k=top_k,
                                      global_num_experts=global_num_experts,
                                      expert_map=expert_map)
        return fused_experts(hidden_states=x,
                             w1=layer.w13_weight,
                             w1_scale=layer.w13_weight_scale,
                             w1_input_scale=layer.w13_input_scale,
                             w1_input_offset=layer.w13_input_offset,
                             w2=layer.w2_weight,
                             w2_scale=layer.w2_weight_scale,
                             w2_input_scale=layer.w2_input_scale,
                             w2_input_offset=layer.w2_input_offset,
                             topk_weights=topk_weights,
                             topk_ids=topk_ids,
                             top_k=top_k,
                             global_num_experts=global_num_experts,
                             expert_map=expert_map)

    def process_weights_after_loading(self, layer):
        if not is_310p():
            layer.w13_weight.data = layer.w13_weight.data.transpose(
                1, 2).contiguous()
            layer.w2_weight.data = layer.w2_weight.data.transpose(
                1, 2).contiguous()
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(
            layer.w13_weight_scale.data.shape[0], -1)

        layer.w13_weight_offset.data = layer.w13_weight_offset.data.view(
            layer.w13_weight_offset.data.shape[0], -1)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.view(
            layer.w2_weight_scale.data.shape[0], -1)
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.view(
            layer.w2_weight_offset.data.shape[0], -1)
        expanding_factor_w13 = layer.w13_weight.data.shape[1]
        expanding_factor_w2 = layer.w2_weight.data.shape[1]

        if is_310p():
            layer.w13_input_scale.data = torch.nn.Parameter(
                layer.w13_input_scale.data.max())
            layer.w2_input_scale.data = torch.nn.Parameter(
                layer.w2_input_scale.data.max())
        else:
            layer.w13_input_scale.data = torch.nn.Parameter(
                layer.w13_input_scale.data.repeat(1,
                                                  expanding_factor_w13)[0:1])
            layer.w2_input_scale.data = torch.nn.Parameter(
                layer.w2_input_scale.data.repeat(1, expanding_factor_w2)[0:1])

        layer.w13_input_offset.data = torch.nn.Parameter(
            layer.w13_input_scale.data.repeat(1, expanding_factor_w13)[0:1])
        layer.w2_input_offset.data = torch.nn.Parameter(
            layer.w2_input_scale.data.repeat(1, expanding_factor_w2)[0:1])

        # converting ACL_FORMAT_FRACTAL_NZ.
        # npu_quant_grouped_matmul_dequant in eager mode does not accept
        # ACL_FORMAT_FRACTAL_NZ.
        if not is_310p():
            layer.w13_weight.data = torch_npu.npu_format_cast(
                layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ).contiguous()
            layer.w2_weight.data = torch_npu.npu_format_cast(
                layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ).contiguous()


class AscendC8KVCacheMethod:

    def __init__(self) -> None:
        self.antiquant_scale_comb = None

    @staticmethod
    def create_weights(layer) -> None:
        param_dict = {}  # num_kv_heads * head_size
        param_dict["key_antiquant_scale"] = torch.empty(layer.num_kv_heads *
                                                        layer.head_size,
                                                        dtype=torch.float16,
                                                        requires_grad=False)
        param_dict["value_antiquant_scale"] = torch.empty(layer.num_kv_heads *
                                                          layer.head_size,
                                                          dtype=torch.float16,
                                                          requires_grad=False)
        for weight_name, weight_param in param_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            layer.register_parameter(weight_name, param)

    def process_weights_after_loading(self, layer):
        self.antiquant_scale_comb = torch.cat(
            (layer.key_antiquant_scale.data.unsqueeze(0),
             layer.value_antiquant_scale.data.unsqueeze(0)),
            dim=0).to(torch.float16).contiguous()

    def apply(self, layer, query, key, value, kv_cache, attn_metadata,
              attn_type, scale, output) -> torch.Tensor:
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.view(num_tokens, layer.num_heads * layer.head_size)
        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")

        # C8
        quant_key = quant_per_tensor(
            key.view(-1, layer.num_kv_heads * layer.head_size),
            layer.key_antiquant_scale.data.view(-1), None, True)
        quant_value = quant_per_tensor(
            value.view(-1, layer.num_kv_heads * layer.head_size),
            layer.value_antiquant_scale.data.view(-1), None, True)

        # View q k v to BSH.
        query = query.view(-1, layer.num_heads, layer.head_size)
        key = key.view(-1, layer.num_kv_heads, layer.head_size)
        value = value.view(-1, layer.num_kv_heads, layer.head_size)
        # TODO: Remove this contiguous in the future.
        value = value.contiguous()

        if kv_cache[0].numel() > 0:
            # if key_cache is None:
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping

            block_size = key_cache.shape[1]
            slots_indices = slots.reshape(-1, 1)
            block_indices = slots_indices // block_size
            slots_indices = slots_indices % block_size
            indices = torch.cat((block_indices, slots_indices), dim=1)

            # C8
            torch_npu.npu_scatter_nd_update_(key_cache, indices, quant_key)
            torch_npu.npu_scatter_nd_update_(value_cache, indices, quant_value)

        # V0-Style scheduler situation.
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            assert attn_metadata is not None
            assert attn_metadata.attn_mask is not None
            mask = attn_metadata.attn_mask
            torch_npu._npu_flash_attention(query=query,
                                           key=key,
                                           value=value,
                                           mask=mask,
                                           seq_len=attn_metadata.seq_lens,
                                           scale_value=scale,
                                           num_heads=layer.num_heads,
                                           num_kv_heads=layer.num_kv_heads,
                                           out=output.reshape(query.shape))

        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            raise NotImplementedError("kv cache int8 are not "
                                      "implemented for "
                                      "PrefillCacheHit")
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:  # changed attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            if hasattr(attn_metadata, "decode"):
                # torch_air
                decode_meta = attn_metadata.decode
                seq_lens = decode_meta.seq_lens_list
            else:
                seq_lens = attn_metadata.seq_lens
            block_size = key_cache.shape[1]
            query = query.view(num_tokens, 1, layer.num_heads *
                               layer.head_size).contiguous()  # changed

            # [num_blocks, block_size, N, D] --> [num_blocks, N, block_size, D]
            key = key_cache
            value = value_cache

            output = torch_npu.npu_incre_flash_attention(
                query,
                key,
                value,
                num_key_value_heads=layer.num_kv_heads,
                num_heads=layer.num_heads,
                actual_seq_lengths=seq_lens,
                scale_value=scale,
                input_layout='BSH',
                block_size=block_size,
                block_table=attn_metadata.block_tables,
                antiquant_scale=self.antiquant_scale_comb,
            )

        # Normal V1 situation.
        else:
            raise NotImplementedError("kv cache int8 are not "
                                      "implemented for "
                                      "other case")
        return output


def fused_experts_310p(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w1_input_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_input_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    expert_map: torch.Tensor = None,
) -> torch.Tensor:
    ep_size = get_ep_group().world_size
    local_num_experts = global_num_experts // ep_size
    local_num_group = top_k // ep_size

    bsz, _ = hidden_states.shape
    flatten_topk_ids = topk_ids.view(-1)
    sorted_topk_ids = torch.argsort(flatten_topk_ids.float())
    sorted_topk_ids = sorted_topk_ids.to(torch.int32)
    sorted_hidden_states = hidden_states.index_select(
        0, sorted_topk_ids // local_num_group)

    experts_id = torch.arange(0,
                              local_num_experts,
                              dtype=topk_ids.dtype,
                              device=topk_ids.device)
    num_tokens_per_expert = (flatten_topk_ids.unsqueeze(-1) == experts_id).to(
        torch.float32).sum(0)
    topk_scales = topk_weights.view(-1).index_select(
        0, sorted_topk_ids).unsqueeze(-1)
    group_list = num_tokens_per_expert.cumsum(dim=0).to(torch.int64)

    gate_up_out = torch_npu.npu_quant_grouped_matmul_dequant(
        x=sorted_hidden_states,
        quantized_weight=w1,
        weight_scale=w1_scale,
        group_list=group_list,
        x_scale=w1_input_scale,
        quant_mode="pertensor")

    gate_up_out = torch_npu.npu_swiglu(gate_up_out.to(torch.float32)).to(
        torch.float16)
    gate_up_out *= topk_scales

    down_out = torch_npu.npu_quant_grouped_matmul_dequant(
        x=gate_up_out,
        quantized_weight=w2,
        weight_scale=w2_scale,
        group_list=group_list,
        x_scale=w2_input_scale,
        quant_mode="pertensor")

    unsorted_topk_ids = torch.argsort(sorted_topk_ids.float()).to(torch.int32)
    unsorted_hidden_states = down_out.index_select(0, unsorted_topk_ids)
    final_hidden_states = unsorted_hidden_states.reshape(
        bsz, top_k // ep_size, -1).sum(1)

    return final_hidden_states


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w1_input_scale: torch.Tensor,
    w1_input_offset: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_input_scale: torch.Tensor,
    w2_input_offset: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    global_num_experts: int,
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

    original_dtype = hidden_states.dtype
    ep_size = get_ep_group().world_size
    local_num_experts = global_num_experts // ep_size
    w1_input_scale, _ = w1_input_scale.max(0)
    quant_sorted_hidden_states = quant_per_tensor(
        hidden_states,
        w1_input_scale,
        None,
        True,
    )
    if expert_map is not None:
        expanded_x, expanded_row_idx, expert_token_count, expanded_scale = torch_npu.npu_moe_init_routing_v2(
            quant_sorted_hidden_states,
            topk_ids,
            scale=None,
            active_num=topk_ids.numel(),
            expert_capacity=-1,
            expert_num=local_num_experts,
            drop_pad_mode=0,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            quant_mode=-1,
            active_expert_range=[0, local_num_experts],
            row_idx_type=0,
        )

    else:
        raise NotImplementedError(
            "The quantified version of MOE class models "
            "currently does not support tensor parallelism")
    if expanded_x.dtype != w1.dtype:
        w1_input_scale, _ = w1_input_scale.max(0)
        quant_sorted_hidden_states = quant_per_tensor(
            expanded_x,
            w1_input_scale,
            None,
            True,
        )
    else:
        quant_sorted_hidden_states = expanded_x
    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[quant_sorted_hidden_states],
        weight=[w1],
        scale=[w1_scale * w1_input_scale[0]],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_token_count,
        output_dtype=original_dtype,
    )[0]
    gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    if gate_up_out.dtype != w2.dtype:
        w2_input_scale, _ = w2_input_scale.max(0)
        quant_gate_up_out = quant_per_tensor(
            gate_up_out,
            w2_input_scale,
            None,
            True,
        )
    else:
        quant_gate_up_out = gate_up_out

    down_out = torch_npu.npu_grouped_matmul(
        x=[quant_gate_up_out],
        weight=[w2],
        scale=[w2_scale * w2_input_scale[0]],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_token_count,
        output_dtype=original_dtype,
    )[0]

    if expert_map is not None:
        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            down_out,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights.to(down_out.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
            drop_pad_mode=2,
        )
    else:
        raise NotImplementedError(
            "The quantified version of MOE class models "
            "currently does not support tensor parallelism")

    return final_hidden_states
