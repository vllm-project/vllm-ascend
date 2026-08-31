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

from typing import Any

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType, use_cann_megamoe
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts  # noqa: F401
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, maybe_trans_nz

from ..base import (
    AscendLinearScheme,
    AscendMoEScheme,
    QuantType,
    TPWeightGatherSpec,
)
from ..registry import register_scheme


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

    act_quant_type: torch.dtype = torch.int8
    tp_weight_gather_specs = (TPWeightGatherSpec("weight"),)
    tp_weight_output_gather_specs = (
        TPWeightGatherSpec("weight", gather_dim=1),
        TPWeightGatherSpec("weight_scale"),
        TPWeightGatherSpec("weight_offset"),
    )
    supports_tp_weight_switch = True

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
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=self.act_quant_type)
        need_unsqz = False
        if pertoken_scale.dim() == 2:
            need_unsqz = True
            quantized_x = quantized_x.squeeze(dim=1)
            pertoken_scale = pertoken_scale.squeeze(dim=1)
        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=pertoken_scale,
            bias=bias if self.act_quant_type == torch.int8 else None,
            output_dtype=x.dtype,
        )
        if need_unsqz:
            output = output.unsqueeze(dim=1)
        return output

    def process_weights_after_loading(self, layer):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        # cast quantized weight tensors in NZ format for higher inference speed
        if self.act_quant_type == torch.int8:
            layer.weight.data = maybe_trans_nz(layer.weight.data)
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.flatten()


@register_scheme("W8A8_DYNAMIC", "moe")
class AscendW8A8DynamicFusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for Ascend W8A8_DYNAMIC."""

    supports_eplb = True
    # Declare the quantization type for this scheme
    quant_type: QuantType = QuantType.W8A8

    def __init__(self):
        vllm_config = get_current_vllm_config()
        ascend_config = get_ascend_config()
        self.dynamic_eplb = False if vllm_config.use_v2_model_runner else ascend_config.eplb_config.dynamic_eplb
        self.use_expert_weight_list = self.dynamic_eplb or (
            vllm_config.use_v2_model_runner is True and vllm_config.parallel_config.enable_eplb is True
        )
        self.in_dtype = vllm_config.model_config.dtype
        try:
            device_group = get_mc2_group().device_group
            # TODO: Try local_rank = ep_group.rank_in_group
            local_rank = torch.distributed.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        except AttributeError:
            logger.warning_once(
                "[vllm-ascend/W8A8_DYNAMIC] MC2 group metadata unavailable, "
                "falling back to empty moe_all_to_all_group_name."
            )
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
        layer: "AscendRoutedExperts",  # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        lora_context = getattr(layer, "_ascend_moe_lora_context", None)
        assert topk_ids is not None
        assert topk_weights is not None
        topk_weights = topk_weights.to(self.in_dtype)

        activation = getattr(layer, "activation", "silu")
        act_name = getattr(activation, "value", activation)
        moe_comm_method = _EXTRA_CTX.moe_comm_method
        fused_scale_flag = (
            _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2
            and get_ascend_config().enable_fused_mc2 == 1
            and act_name != "swigluoai_uninterleave"
        )
        use_mega_moe_quant_path = fused_scale_flag and _EXTRA_CTX.use_mega_moe
        if self.use_expert_weight_list:
            w1 = layer.w13_weight_list
            w2 = layer.w2_weight_list
            if use_mega_moe_quant_path:
                # EPLB rearranges these lists in place. MegaMoE consumes the
                # original INT8/NZ weights and one flattened scale per expert.
                w1_scale = [scale.reshape(-1) for scale in layer.fused_w1_scale_list]
                w2_scale = [scale.reshape(-1) for scale in layer.fused_w2_scale_list]
                w1_scale_bias = None
                w2_scale_bias = None
            else:
                w1_scale = layer.fused_w1_scale_list if fused_scale_flag else layer.w13_weight_scale_fp32_list
                w2_scale = layer.fused_w2_scale_list if fused_scale_flag else layer.w2_weight_scale_list
                w1_scale_bias = layer.fused_w1_scale_bias if fused_scale_flag else None
                w2_scale_bias = layer.fused_w2_scale_bias if fused_scale_flag else None

        elif use_mega_moe_quant_path:
            w1 = layer.cann_mega_moe_w13_weight_list
            w1_scale = layer.cann_mega_moe_fused_w1_scale_list
            w2 = layer.cann_mega_moe_w2_weight_list
            w2_scale = layer.cann_mega_moe_fused_w2_scale_list
            w1_scale_bias = None
            w2_scale_bias = None

        else:
            w1 = [layer.w13_weight]
            w1_scale = [layer.fused_w1_scale] if fused_scale_flag else [layer.w13_weight_scale_fp32]
            w2 = [layer.w2_weight]
            w2_scale = [layer.fused_w2_scale] if fused_scale_flag else [layer.w2_weight_scale]
            w1_scale_bias = layer.fused_w1_scale_bias if fused_scale_flag else None
            w2_scale_bias = layer.fused_w2_scale_bias if fused_scale_flag else None

        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=w1,
                w2=w2,
                quant_type=self.quant_type,
                dynamic_eplb=self.use_expert_weight_list,
                expert_map=layer.ascend_expert_map,
                global_redundant_expert_num=layer.global_redundant_expert_num,
                mc2_mask=layer.ascend_mc2_mask,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                pertoken_scale=layer.ascend_pertoken_scale,
                activation=activation,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                lora_context=lora_context,
            )
        )
        return final_hidden_states

    @staticmethod
    def get_eplb_weight_views(layer: torch.nn.Module) -> list:
        if hasattr(layer, "w13_weight_list"):
            weights = [
                layer.w13_weight_list,
                layer.w2_weight_list,
                layer.w13_weight_scale_fp32_list,
                layer.w2_weight_scale_list,
            ]
            fused_w1_scale = getattr(layer, "fused_w1_scale_list", None)
            fused_w2_scale = getattr(layer, "fused_w2_scale_list", None)
            if (fused_w1_scale is None) != (fused_w2_scale is None):
                raise RuntimeError(
                    "FUSED_MC2 EPLB requires fused_w1_scale_list and fused_w2_scale_list "
                    "to be present or absent together."
                )
            if fused_w1_scale is not None and fused_w2_scale is not None:
                weights.extend([fused_w1_scale, fused_w2_scale])
            return weights

        weights = [
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_fp32,
            layer.w2_weight_scale,
        ]
        fused_w1_scale = getattr(layer, "fused_w1_scale", None)
        fused_w2_scale = getattr(layer, "fused_w2_scale", None)
        if (fused_w1_scale is None) != (fused_w2_scale is None):
            raise RuntimeError(
                "FUSED_MC2 EPLB requires fused_w1_scale and fused_w2_scale to be present or absent together."
            )
        if fused_w1_scale is not None and fused_w2_scale is not None:
            num_local_experts = layer.w13_weight.shape[0]
            weights.extend(
                [
                    fused_w1_scale.view(num_local_experts, -1),
                    fused_w2_scale.view(num_local_experts, -1),
                ]
            )
        return weights

    def process_weights_after_loading(self, layer):
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()
        # TODO(zzzzwwjj): Currently, `torch_npu.npu_grouped_matmul_swiglu_quant`
        # can only support weight nz.
        if self.quant_type == QuantType.W8A8:
            layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
            layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(layer.w13_weight_scale.data.shape[0], -1)
        layer.w13_weight_scale_fp32 = layer.w13_weight_scale.data.to(torch.float32)
        layer.w13_weight_offset.data = layer.w13_weight_offset.data.view(layer.w13_weight_offset.data.shape[0], -1)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.view(layer.w2_weight_scale.data.shape[0], -1)
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.view(layer.w2_weight_offset.data.shape[0], -1)

        if get_ascend_config().enable_fused_mc2 == 1:
            layer.fused_w1_scale = scale_from_float_to_int64(layer.w13_weight_scale.data)
            layer.fused_w2_scale = scale_from_float_to_int64(layer.w2_weight_scale.data)
            layer.fused_w1_scale_bias = [torch.tensor([], dtype=torch.float32)]
            layer.fused_w2_scale_bias = [torch.tensor([], dtype=torch.float32)]

        if self.use_expert_weight_list:
            layer.w13_weight_list = [weight.clone() for weight in layer.w13_weight.data.unbind(dim=0)]
            layer.w2_weight_list = [weight.clone() for weight in layer.w2_weight.data.unbind(dim=0)]
            layer.w13_weight_scale_fp32_list = [
                weight.clone() for weight in layer.w13_weight_scale_fp32.data.unbind(dim=0)
            ]
            layer.w2_weight_scale_list = [weight.clone() for weight in layer.w2_weight_scale.data.unbind(dim=0)]
            if get_ascend_config().enable_fused_mc2 == 1:
                layer.fused_w1_scale_list = [
                    weight.clone()
                    for weight in layer.fused_w1_scale.view(len(layer.w13_weight_list), -1).data.unbind(dim=0)
                ]
                layer.fused_w2_scale_list = [
                    weight.clone()
                    for weight in layer.fused_w2_scale.view(len(layer.w2_weight_list), -1).data.unbind(dim=0)
                ]
            del layer.w13_weight
            del layer.w2_weight
            del layer.w13_weight_scale
            del layer.w13_weight_scale_fp32
            del layer.w2_weight_scale
            if get_ascend_config().enable_fused_mc2 == 1:
                del layer.fused_w1_scale
                del layer.fused_w2_scale
            torch.npu.empty_cache()

        elif use_cann_megamoe(get_current_vllm_config()):
            layer.cann_mega_moe_w13_weight_list = list(layer.w13_weight.data.unbind(dim=0))
            layer.cann_mega_moe_w2_weight_list = list(layer.w2_weight.data.unbind(dim=0))
            layer.cann_mega_moe_fused_w1_scale_list = list(
                layer.fused_w1_scale.view(layer.w13_weight.shape[0], -1).data.unbind(dim=0)
            )
            layer.cann_mega_moe_fused_w2_scale_list = list(
                layer.fused_w2_scale.view(layer.w2_weight.shape[0], -1).data.unbind(dim=0)
            )
