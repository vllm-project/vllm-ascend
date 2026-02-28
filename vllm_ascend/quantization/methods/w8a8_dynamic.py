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

from collections.abc import Callable
from typing import Any

import torch
import torch_npu
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.flash_common3_context import get_flash_common3_context
from vllm_ascend.ops.fused_moe.experts_selector import select_experts, zero_experts_compute
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, maybe_trans_nz

from .base import AscendLinearScheme, AscendMoEScheme, QuantType
from .registry import register_scheme


def scale_from_float_to_int64(scale):
    """Convert float32 scale to int64 representation."""
    import numpy as np

    scale = torch.from_numpy(
        np.frombuffer(scale.cpu().to(torch.float32).numpy().tobytes(), dtype=np.int32).astype(np.int64)
    ).to(scale.device)
    return scale


@register_scheme("W8A8_DYNAMIC", "linear")
class AscendW8A8DynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W8A8_DYNAMIC.

    This scheme uses dynamic per-token quantization for activations
    and per-channel quantization for weights.
    """

    def __init__(self):
        pass

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}
        return params_dict

    def get_perchannel_param(
        self,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=pertoken_scale,
            bias=bias,
            output_dtype=x.dtype,
        )
        return output

    def process_weights_after_loading(self, layer):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        # cast quantized weight tensors in NZ format for higher inference speed
        layer.weight.data = maybe_trans_nz(layer.weight.data)
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.flatten()


@register_scheme("W8A8_DYNAMIC", "moe")
class AscendW8A8DynamicFusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for Ascend W8A8_DYNAMIC."""

    # Declare the quantization type for this scheme
    quant_type: QuantType = QuantType.W8A8

    def __init__(self):
        self.ep_group = get_ep_group()

        vllm_config = get_current_vllm_config()
        ascend_config = get_ascend_config()
        self.use_aclgraph = (
            vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not vllm_config.model_config.enforce_eager
        )
        self.multistream_overlap_gate = ascend_config.multistream_overlap_gate

        self.dynamic_eplb = ascend_config.eplb_config.dynamic_eplb
        self.in_dtype = vllm_config.model_config.dtype
        self.supports_eplb = True

        try:
            device_group = get_mc2_group().device_group
            # TODO: Try local_rank = ep_group.rank_in_group
            local_rank = torch.distributed.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        except AttributeError:
            self.moe_all_to_all_group_name = ""

    def get_weight(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes, dtype=torch.int8
        )
        param_dict["w2_weight"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition, dtype=torch.int8
        )
        return param_dict

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=params_dtype
        )
        param_dict["w13_weight_offset"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=params_dtype
        )
        param_dict["w2_weight_scale"] = torch.empty(num_experts, hidden_sizes, 1, dtype=params_dtype)
        param_dict["w2_weight_offset"] = torch.empty(num_experts, hidden_sizes, 1, dtype=params_dtype)
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
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        **kwargs,
    ) -> torch.Tensor:
        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)
        if zero_expert_num == 0 or zero_expert_type is None:
            assert router_logits.shape[1] == global_num_experts - global_redundant_expert_num, (
                "Number of global experts mismatch (excluding redundancy)"
            )

        if self.multistream_overlap_gate:
            fc3_context = get_flash_common3_context()
            assert fc3_context is not None
            topk_weights = fc3_context.topk_weights
            topk_ids = fc3_context.topk_ids
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
                routed_scaling_factor=routed_scaling_factor,
                e_score_correction_bias=e_score_correction_bias,
                global_num_experts=global_num_experts,
            )
        assert topk_ids is not None
        assert topk_weights is not None
        if zero_expert_num > 0 and zero_expert_type is not None:
            topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=global_num_experts,
                zero_expert_type=zero_expert_type,
                hidden_states=x,
            )
        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance:
            random_matrix = torch.rand(
                topk_ids.size(0), global_num_experts - global_redundant_expert_num, device=topk_ids.device
            )
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        assert topk_weights is not None
        topk_weights = topk_weights.to(self.in_dtype)

        moe_comm_method = get_forward_context().moe_comm_method
        if self.dynamic_eplb:
            w1 = layer.w13_weight_list
            w1_scale = layer.w13_weight_scale_fp32_list
            w2 = layer.w2_weight_list
            w2_scale = layer.w2_weight_scale_list
        else:
            w1 = [layer.w13_weight]
            w1_scale = [layer.w13_weight_scale_fp32]
            w2 = [layer.w2_weight]
            w2_scale = [layer.w2_weight_scale]

        fused_scale_flag = (
            get_forward_context().moe_comm_type == MoECommType.FUSED_MC2
            and envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 1
        )
        final_hidden_states = moe_comm_method.fused_experts(
            hidden_states=x,
            pertoken_scale=pertoken_scale,
            w1=w1,
            w1_scale=[layer.fused_w1_scale] if fused_scale_flag else w1_scale,
            w2=w2,
            w2_scale=[layer.fused_w2_scale] if fused_scale_flag else w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            use_int8_w8a8=True,
            expert_map=expert_map,
            log2phy=log2phy,
            dynamic_eplb=self.dynamic_eplb,
            mc2_mask=kwargs.get("mc2_mask"),
        )
        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states += zero_expert_result
        return final_hidden_states

    def process_weights_after_loading(self, layer):
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()
        # TODO(zzzzwwjj): Currently, `torch_npu.npu_grouped_matmul_swiglu_quant`
        # can only support weight nz.
        layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
        layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(layer.w13_weight_scale.data.shape[0], -1)
        layer.w13_weight_scale_fp32 = layer.w13_weight_scale.data.to(torch.float32)
        layer.w13_weight_offset.data = layer.w13_weight_offset.data.view(layer.w13_weight_offset.data.shape[0], -1)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.view(layer.w2_weight_scale.data.shape[0], -1)
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.view(layer.w2_weight_offset.data.shape[0], -1)

        layer.fused_w1_scale = scale_from_float_to_int64(layer.w13_weight_scale.data)
        layer.fused_w2_scale = scale_from_float_to_int64(layer.w2_weight_scale.data)

        if self.dynamic_eplb:
            layer.w13_weight_list = [weight.clone() for weight in layer.w13_weight.data.unbind(dim=0)]
            layer.w2_weight_list = [weight.clone() for weight in layer.w2_weight.data.unbind(dim=0)]
            layer.w13_weight_scale_fp32_list = [
                weight.clone() for weight in layer.w13_weight_scale_fp32.data.unbind(dim=0)
            ]
            layer.w2_weight_scale_list = [weight.clone() for weight in layer.w2_weight_scale.data.unbind(dim=0)]
            del layer.w13_weight
            del layer.w2_weight
            del layer.w13_weight_scale
            del layer.w13_weight_scale_fp32
            del layer.w2_weight_scale
            torch.npu.empty_cache()
