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

import numpy as np
import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_forward_context import FusedMoEState
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe import unified_fused_experts_eager
from vllm_ascend.ops.moe.experts_selector import select_experts


class AscendW4A8DynamicLinearMethod:
    """Linear method for Ascend W4A8_DYNAMIC
    """

    def __init__(self):
        self.transpose_weight = True
        try:
            self.group_size = get_current_vllm_config(
            ).quant_config.quant_description.get("group_size", 256)
        except AttributeError:
            self.group_size = 256

    @staticmethod
    def get_weight(input_size: int, output_size: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    @staticmethod
    def get_perchannel_param(output_size: int,
                             params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    def get_pergroup_param(self, input_size: int, output_size: int,
                           params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size,
                                                  1,
                                                  dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size,
                                                   1,
                                                   dtype=params_dtype)
        params_dict["weight_scale_second"] = torch.empty(output_size,
                                                         input_size //
                                                         self.group_size,
                                                         dtype=params_dtype)
        params_dict["weight_offset_second"] = torch.empty(output_size,
                                                          input_size //
                                                          self.group_size,
                                                          dtype=params_dtype)
        return params_dict

    @staticmethod
    def process_scale_second(weight: torch.Tensor, scale: torch.Tensor,
                             per_group_scale: torch.Tensor):
        k, n = weight.shape
        group_num, n = per_group_scale.shape
        weight_high = weight.to(torch.float32).reshape(
            group_num, -1, n) * per_group_scale.reshape(group_num, 1, n)
        weight_high = weight_high.reshape(k, n)
        bias = 8 * (weight_high.to(torch.float32) * scale).sum(dim=0)
        antiquant_scale = (scale * per_group_scale).reshape(group_num, n)
        return antiquant_scale.npu(), bias

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = None,
    ) -> torch.Tensor:
        return torch_npu.npu_weight_quant_batchmatmul(
            x,
            layer.weight,
            antiquant_scale=layer.weight_scale_second.to(x.dtype),
            antiquant_group_size=self.group_size,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module):
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten().to(
            torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.flatten()
        layer.weight_scale_second.data, scale_bias = self.process_scale_second(
            layer.weight.data,
            layer.weight_scale.data,
            layer.weight_scale_second.data.transpose(0, 1).contiguous(),
        )
        param = torch.nn.Parameter(scale_bias, requires_grad=False)
        layer.register_parameter("weight_scale_bias", param)
        layer.weight.data = torch_npu.npu_convert_weight_to_int4pack(
            layer.weight.data.to(torch.int32))


class AscendW4A8DynamicFusedMoEMethod:
    """FusedMoe method for Ascend W4A8_DYNAMIC.
    """

    def __init__(self):
        self.transpose_weight = True

        self.ep_group = get_ep_group()

        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get(
            "group_size", 256)
        quant_version = vllm_config.quant_config.quant_description.get(
            "version", "0")
        # NOTE: new quantize weights: 2 int4 pack into int8
        self.new_quant_version = quant_version == "1.0.0"
        self.tp_size = 1 if vllm_config.parallel_config.enable_expert_parallel else self.ep_group.world_size
        if self.new_quant_version and self.tp_size > 16:
            raise ValueError(
                "The current weight does not support moe part tp>16.")

        try:
            device_group = get_mc2_group().device_group
            # TODO: Try local_rank = ep_group.rank_in_group
            local_rank = torch.distributed.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self.moe_all_to_all_group_name = backend.get_hccl_comm_name(
                local_rank)
        except AttributeError:
            self.moe_all_to_all_group_name = ""

    def get_weight(self, num_experts: int,
                   intermediate_size_per_partition: int, hidden_sizes: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        if self.new_quant_version:
            w13_output_size = intermediate_size_per_partition
            w2_output_size = hidden_sizes // 2
        else:
            w13_output_size = 2 * intermediate_size_per_partition
            w2_output_size = hidden_sizes

        param_dict["w13_weight"] = torch.empty(num_experts,
                                               w13_output_size,
                                               hidden_sizes,
                                               dtype=torch.int8)
        param_dict["w2_weight"] = torch.empty(num_experts,
                                              w2_output_size,
                                              intermediate_size_per_partition,
                                              dtype=torch.int8)
        return param_dict

    def get_dynamic_quant_param(self, num_experts: int,
                                intermediate_size_per_partition: int,
                                hidden_sizes: int,
                                params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=params_dtype)

        param_dict["w13_weight_offset"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=params_dtype)

        param_dict["w13_weight_scale_second"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes // self.group_size,
            dtype=params_dtype)

        param_dict["w13_weight_offset_second"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes // self.group_size,
            dtype=params_dtype)

        param_dict["w2_weight_scale"] = torch.empty(num_experts,
                                                    hidden_sizes,
                                                    1,
                                                    dtype=params_dtype)
        param_dict["w2_weight_offset"] = torch.empty(num_experts,
                                                     hidden_sizes,
                                                     1,
                                                     dtype=params_dtype)
        param_dict["w2_weight_scale_second"] = torch.empty(
            num_experts,
            hidden_sizes,
            intermediate_size_per_partition // self.group_size,
            dtype=params_dtype)
        param_dict["w2_weight_offset_second"] = torch.empty(
            num_experts,
            hidden_sizes,
            intermediate_size_per_partition // self.group_size,
            dtype=params_dtype)

        if self.new_quant_version:
            param_dict["w13_scale_bias"] = torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                1,
                dtype=torch.float32)
            param_dict["w2_scale_bias"] = torch.empty(num_experts,
                                                      hidden_sizes,
                                                      16 // self.tp_size,
                                                      dtype=torch.float32)

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
        enable_force_load_balance: bool = True,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        shared_experts: Optional[Any] = None,
        quantized_x_for_share: Optional[Any] = None,
        dynamic_scale_for_share: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert router_logits.shape[
            1] == global_num_experts, "Number of global experts mismatch"

        # NOTE: now npu_moe_gating_top_k can only support `group_count=256` pattern
        topk_weights, topk_ids, row_idx = select_experts(
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

        fused_moe_state = get_forward_context().fused_moe_state
        shared_gate_up, shared_dequant_scale = None, None
        if shared_experts is not None and fused_moe_state == FusedMoEState.MC2:
            share_up_out, _ = shared_experts.gate_up_proj(
                (quantized_x_for_share, dynamic_scale_for_share))
            shared_gate_up, shared_dequant_scale = share_up_out[
                0], share_up_out[1]

        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance:
            topk_ids = torch.randint_like(topk_ids, 0, global_num_experts)

        topk_weights = topk_weights.to(x.dtype)

        return unified_fused_experts_eager(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale_second,
            w2_scale=layer.w2_weight_scale_second,
            w1_scale_bias=layer.w13_scale_bias,
            w2_scale_bias=layer.w2_scale_bias,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            row_idx=row_idx,
            expert_map=expert_map,
            log2phy=log2phy,
            global_redundant_expert_num=global_redundant_expert_num,
            shared_experts=shared_experts,
            shared_gate_up=shared_gate_up,
            shared_dequant_scale=shared_dequant_scale,
            mc2_mask=kwargs.get("mc2_mask", None),
            with_quant=True)

    def process_scale(self, weight: torch.Tensor, scale, per_group_scale):
        group_num, k, n = weight.shape
        # the weight of the new version is reduced by half by pack n, so it needs to be restored
        if self.new_quant_version:
            n = n * 2
        per_group_scale = per_group_scale.reshape(group_num, -1, n)
        group_num, quantgroup_num, n = per_group_scale.shape
        bias = None
        if not self.new_quant_version:
            weight_high = weight.to(torch.float32).reshape([group_num, quantgroup_num, -1, n]) * \
                per_group_scale.reshape([group_num, quantgroup_num, 1, n])
            weight_high = weight_high.reshape([group_num, k, n])
            bias = 8 * (weight_high.to(torch.float32) * scale).sum(axis=1)
        scale_fp32 = (scale * per_group_scale).to(torch.float16).to(
            torch.float32)
        scale_fp32_np = scale_fp32.cpu().numpy()
        scale_fp32_np.dtype = np.uint32
        sscale_uint64 = np.zeros((group_num, quantgroup_num, n * 2),
                                 dtype=np.uint32)

        sscale_uint64[..., ::2] = scale_fp32_np

        sscale_uint64_buffer = np.frombuffer(sscale_uint64.tobytes(),
                                             dtype=np.int64).copy()
        sscale_uint64_tensor = torch.from_numpy(sscale_uint64_buffer).reshape(
            group_num, quantgroup_num, n)
        sscale_uint64_tensor = sscale_uint64_tensor.npu()
        return sscale_uint64_tensor, bias

    def update_bias(self, layer, w13_bias, w2_bias):
        if self.new_quant_version:
            layer.w13_scale_bias.data = layer.w13_scale_bias.data.transpose(
                1, 2).contiguous().sum(axis=1)
            layer.w2_scale_bias.data = layer.w2_scale_bias.data.transpose(
                1, 2).contiguous().sum(axis=1)
        else:
            w13_scale_bias = torch.nn.Parameter(w13_bias, requires_grad=False)
            layer.register_parameter("w13_scale_bias", w13_scale_bias)
            w2_scale_bias = torch.nn.Parameter(w2_bias, requires_grad=False)
            layer.register_parameter("w2_scale_bias", w2_scale_bias)

    def pack_to_int32(self, weight: torch.Tensor):
        if self.new_quant_version:
            group_num, k, n = weight.shape
            assert n % 4 == 0, "the last dim of weight needs to be divided by 4"
            packed_n = n // 4
            # pack 4 int8(int4*2) to int32, because in pytorch, we need to use int32 to represent int4
            packed_weight = torch.from_numpy(
                np.frombuffer(weight.cpu().numpy().tobytes(), dtype=np.int32))
            return packed_weight.reshape(group_num, k, packed_n).npu()
        else:
            return torch_npu.npu_quantize(weight.to(torch.float32),
                                          torch.tensor([1.]).npu(), None,
                                          torch.quint4x2, -1, False)

    def process_weights_after_loading(self, layer):
        if self.transpose_weight:
            layer.w13_weight.data = layer.w13_weight.data.transpose(
                1, 2).contiguous()
            layer.w2_weight.data = layer.w2_weight.data.transpose(
                1, 2).contiguous()
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(
            1, 2).contiguous()
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(
            1, 2).contiguous()
        layer.w13_weight_scale_second.data = layer.w13_weight_scale_second.data.transpose(
            1, 2).contiguous()
        layer.w2_weight_scale_second.data = layer.w2_weight_scale_second.data.transpose(
            1, 2).contiguous()

        layer.w13_weight_scale_second.data, w13_bias = self.process_scale(
            layer.w13_weight, layer.w13_weight_scale.data,
            layer.w13_weight_scale_second.data)
        layer.w2_weight_scale_second.data, w2_bias = self.process_scale(
            layer.w2_weight, layer.w2_weight_scale.data,
            layer.w2_weight_scale_second.data)

        self.update_bias(layer, w13_bias, w2_bias)

        layer.w13_weight.data = self.pack_to_int32(layer.w13_weight.data)
        layer.w2_weight.data = self.pack_to_int32(layer.w2_weight.data)
