#
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
# Adapted from the vllm-ascend project to reuse its model components
#  for omni-infer integration.
#
# By using quantization case, this file is called before worker patch achieve,
# we need to import patch_utils here first to make sure the patch is applied.

# import vllm_ascend.patch.worker.patch_common.patch_utils  # type: ignore[import]  # isort: skip  # noqa
import os
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch
import torch_npu
from torch.library import Library

from vllm import utils
from vllm.utils import vllm_lib
from vllm.distributed.parallel_state import get_ep_group
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.parameter import PerTensorScaleParameter
from vllm.model_executor.utils import set_weight_attrs

from omni.adaptors.vllm.utils import NPU_W8A8_STATIC
from omni.models.pangu.pangu_pro_moe.fused_moe import select_experts
from omni.layers.attention.backend.attention import AscendAttentionState

def ascend_direct_register_custom_op(
        op_name: str,
        op_func: Callable,
        mutates_args: list[str],
        fake_impl: Optional[Callable] = None,
        target_lib: Optional[Library] = None,
        dispatch_key: str = "CUDA",
        tags: Tuple[torch.Tag, ...] = (),
):
    # In pytorch 2.5.1, torch.library.infer_schema require the input function to
    # have annotations supported by typing library. But in pytorch 2.7.0 which
    # vllm using, torch.library.infer_schema require the python builtin type. In
    # this case, we should revert built type to typing type for 2.5.1 backward
    # compatibility.
    for k, v in op_func.__annotations__.items():
        if v == list[int]:
            op_func.__annotations__[k] = List[int]
        if v == Optional[list[int]]:
            op_func.__annotations__[k] = Optional[List[int]]
        # TODO: add more type convert here if needed.
    import torch.library
    schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)

utils.direct_register_custom_op = ascend_direct_register_custom_op

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

def quant_per_tensor(in_tensor: torch.Tensor,
                     input_scale: torch.Tensor,
                     input_offset: torch.Tensor,
                     function=False):
    return torch_npu.npu_quantize(in_tensor, input_scale, input_offset,
                                  torch.qint8, -1, function)

@register_quantization_config(NPU_W8A8_STATIC)
class AscendQuantConfig_Pangu_Pro_Moe(QuantizationConfig):
    """Config class for Ascend
    
    This class is a general class that parse quantization configs
    that are supported on ascend hardware.
    """

    def __init__(self, quant_config: Dict[str, Any]):
        super().__init__()
        self.quant_description = quant_config

    def __repr__(self) -> str:
        return "AscendQuantConfig_Pangu_Pro_Moe:\n" + super().__repr__()

    @classmethod
    def get_name(cls) -> str:
        return NPU_W8A8_STATIC

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "Ascend hardware dose not support \"get_min_capability\" feature.")

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quant_model_description.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AscendQuantConfig_Pangu_Pro_Moe":
        return cls(config)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        if torch.npu.is_available():
            return 
        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if self.is_layer_skipped_ascend(prefix, self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            return AscendLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            if self.is_layer_skipped_ascend(prefix, self.packed_modules_mapping):
                return AscendUnquantizedFusedMoEMethod()
            return AscendFusedMoEMethod(self)
        return None

    def is_layer_skipped_ascend(
        self,
        prefix: str,
        fused_mapping: Mapping[str, List[str]] = MappingProxyType({})):
        # adapted from vllm.model_executor.layers.quantization.utils.quant_utils.is_layer_skipped
        proj_name = prefix.split(".")[-1]
        if proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name)
                for shard_proj_name in fused_mapping[proj_name]
            ]

            is_skipped = None
            for shard_prefix in shard_prefixes:
                is_shard_skipped = self.quant_description[shard_prefix + '.weight'] == "FLOAT"

                if is_skipped is None:
                    is_skipped = is_shard_skipped
                elif is_shard_skipped != is_skipped:
                    raise ValueError(
                        f"Detected some but not all shards of {prefix} "
                        "are quantized. All shards of fused layers "
                        "to have the same precision.")
        else:
            is_skipped = self.quant_description[prefix + '.weight'] == "FLOAT"

        assert is_skipped is not None
        return is_skipped

    def get_scaled_act_names(self) -> List[str]:
        return []



class AscendLinearMethod(LinearMethodBase):
    """Linear method for Ascend quantization.

    This class calls AscendQuantizer to search a specific quantization
    implementations supported on ascend hardware for linear methods.

    Args:
        quant_config: The Ascend quantization config.
    """

    def __init__(self, quant_config: AscendQuantConfig_Pangu_Pro_Moe) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        # ====== get_weight ======
        weight_dict = {
            "weight": torch.empty(output_size_per_partition, input_size_per_partition, dtype=torch.int8)
        }
        for weight_name, weight_param in weight_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})
            layer.register_parameter(weight_name, param)
            set_weight_attrs(param, extra_weight_attrs)

        # ====== get_pertensor_param ======
        pertensor_dict = {
            "input_scale": torch.empty(1, dtype=params_dtype),
            "input_offset": torch.empty(1, dtype=torch.int8)
        }
        for pertensor_name, pertensor_param in pertensor_dict.items():
            param = PerTensorScaleParameter(data=pertensor_param,
                                            weight_loader=weight_loader)
            # disable warning
            param.ignore_warning = True
            layer.register_parameter(pertensor_name, param)

        # ====== get_perchannel_param ======
        perchannel_dict = {
            "quant_bias": torch.empty(output_size_per_partition, dtype=torch.int32),
            "weight_scale": torch.empty(output_size_per_partition, 1, dtype=params_dtype),
            "weight_offset": torch.empty(output_size_per_partition, 1, dtype=params_dtype)
        }
        if params_dtype == torch.bfloat16:
            perchannel_dict["deq_scale"] = torch.empty(output_size_per_partition, dtype=torch.float32)
        elif params_dtype == torch.float16:
            perchannel_dict["deq_scale"] = torch.empty(output_size_per_partition, dtype=torch.int64)
        
        for perchannel_name, perchannel_param in perchannel_dict.items():
            param = torch.nn.Parameter(perchannel_param, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(perchannel_name, param)
            set_weight_attrs(param, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        expanding_factor = layer.weight.data.shape[1]
        layer.aclnn_input_scale = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor),
            requires_grad=False)
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor),
            requires_grad=False).to(layer.aclnn_input_scale.dtype)
        
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 29)
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        layer.weight_offset.data = torch.flatten(layer.weight_offset.data)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(layer, RowParallelLinear):
            tp_rank = get_tensor_model_parallel_rank()
        else:
            tp_rank = 0

        original_dtype = x.dtype
        if original_dtype != torch.int8:
            x = quant_per_tensor(x, layer.aclnn_input_scale, layer.aclnn_input_offset)

        quant_bias = layer.quant_bias if tp_rank == 0 else None
        output = torch_npu.npu_quant_matmul(
            x,
            layer.weight,
            layer.deq_scale,
            bias=quant_bias,
            output_dtype=original_dtype,
        )
        return output



class AscendKVCacheMethod(BaseKVCacheMethod):
    """KVCache method for Ascend quantization.

    This class calls AscendQuantizer to search a specific quantization
    implementations supported on ascend hardware for kvcache methods.

    Args:
        quant_config: The Ascend quantization config.
    """

    def __init__(self, quant_config: AscendQuantConfig_Pangu_Pro_Moe) -> None:
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module) -> None:
        # Different from linear method, there are no weight processing/slicing
        # steps for attention in vllm. So the whole process of create weights
        # is hidden into the specific quant method.
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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.antiquant_scale_comb = torch.cat(
            (layer.key_antiquant_scale.data.unsqueeze(0),
             layer.value_antiquant_scale.data.unsqueeze(0)),
            dim=0).to(torch.float16).contiguous()

    def apply(self, layer: torch.nn.Module, query: torch.Tensor,
              key: torch.Tensor, value: torch.Tensor, kv_cache, attn_metadata,
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
            # torch_air
            # decode_meta = attn_metadata.decode
            # seq_lens = decode_meta.seq_lens_list
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


class AscendFusedMoEMethod(FusedMoEMethodBase):
    """FusedMoE method for Ascend quantization.

    This class calls AscendQuantizer to search a specific quantization
    implementations supported on ascend hardware for kvcache methods.

    Args:
        quant_config: The Ascend quantization config.
    """

    def __init__(self, quant_config: AscendQuantConfig_Pangu_Pro_Moe):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        # ====== get_weight ======
        weight_param = {
            "w13_weight": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
                requires_grad=False
            ),
            "w2_weight": torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
                requires_grad=False
            )
        }
        for param_key, param_value in weight_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)

        # ====== get_dynamic_quant_param ======
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        dynamic_quant_param = {
            "w13_weight_scale": torch.empty(num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32),
            "w13_weight_offset": torch.empty(num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float16),
            "w2_weight_scale": torch.empty(num_experts, hidden_size, 1, dtype=torch.float32),
            "w2_weight_offset": torch.empty(num_experts, hidden_size, 1, dtype=torch.float16),
            "w2_deq_scale": torch.empty(num_experts, hidden_size, dtype=torch.float32),
            "w13_deq_scale": torch.empty(num_experts, 2 * intermediate_size_per_partition, dtype=torch.float32),
            "w2_input_scale": torch.empty(num_experts, 1, dtype=torch.float32),
            "w13_input_scale": torch.empty(num_experts, 1, dtype=torch.float32),
            "w2_input_offset": torch.empty(num_experts, 1, dtype=torch.int8),
            "w13_input_offset": torch.empty(num_experts, 1, dtype=torch.int8),
            "quant_bias": torch.empty(num_experts, hidden_size, dtype=torch.int32)
        }
        for param_key, param_value in dynamic_quant_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)

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
        assert router_logits.shape[1] == global_num_experts, "Number of global experts mismatch"

        # NOTE: now npu_moe_gating_top_k can only support `group_count=256` pattern
        if global_num_experts == 256:
            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                router_logits,
                k=top_k,
                bias=e_score_correction_bias,
                k_group=topk_group,
                group_count=num_expert_group,
                group_select_mode=1,
                renorm=0,
                norm_type=1,
                routed_scaling_factor=1,
                eps=float(1e-20)
            )
        else:
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
                global_num_experts=global_num_experts,
            )

        if os.environ.get("VLLM_ENABLE_MC2", '0') == "1" and not is_prefill:
            raise NotImplementedError("W8A8FusedMoe are not implemented for VLLM_ENABLE_MC2")

        return fused_experts(
            hidden_states=x,
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
            expert_map=expert_map
        )


    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # torch.npu.config.allow_internal_format = True
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(layer.w13_weight_scale.data.shape[0], -1).to(torch.float32)
        layer.w13_weight_offset.data = layer.w13_weight_offset.data.view(layer.w13_weight_offset.data.shape[0], -1).to(torch.float16)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.view(layer.w2_weight_scale.data.shape[0], -1).to(torch.float32)
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.view(layer.w2_weight_offset.data.shape[0], -1).to(torch.float16)
        expanding_factor_w13 = layer.w13_weight.data.shape[1]
        expanding_factor_w2 = layer.w2_weight.data.shape[1]
        layer.w13_input_scale.data = torch.nn.Parameter(layer.w13_input_scale.data.repeat(1, expanding_factor_w13)[0:1]).to(torch.float16)

        layer.w2_input_scale.data = torch.nn.Parameter(layer.w2_input_scale.data.repeat(1, expanding_factor_w2)[0:1]).to(torch.float16)
        layer.w13_input_offset.data = torch.nn.Parameter(layer.w13_input_scale.data.repeat(1, expanding_factor_w13)[0:1]).to(torch.int8)
        layer.w2_input_offset.data = torch.nn.Parameter(layer.w2_input_scale.data.repeat(1, expanding_factor_w2)[0:1]).to(torch.int8)

        # NZ
        # layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, 29).contiguous()
        # layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, 29).contiguous()
