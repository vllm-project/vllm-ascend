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
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, _MEGA_MOE_SUPPORTED, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, enable_dsa_cp, maybe_trans_nz

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

    act_quant_type: torch.dtype = torch.int8

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

        chunk_size = getattr(layer, "_chunk_size", 0)
        if isinstance(chunk_size, int) and chunk_size > 0:
            bias_1 = bias[:chunk_size] if bias is not None else None
            bias_2 = bias[chunk_size:] if bias is not None else None
            output = torch.cat(
                [
                    torch_npu.npu_quant_matmul(
                        quantized_x,
                        layer.weight_1,
                        layer.weight_1_scale,
                        pertoken_scale=pertoken_scale,
                        bias=bias_1,
                        output_dtype=x.dtype,
                    ),
                    torch_npu.npu_quant_matmul(
                        quantized_x,
                        layer.weight_2,
                        layer.weight_2_scale,
                        pertoken_scale=pertoken_scale,
                        bias=bias_2,
                        output_dtype=x.dtype,
                    ),
                ],
                dim=-1,
            )
        else:
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
        if "wq_b" in getattr(layer, "prefix", "") and layer.weight.shape[1] >= 65536 and enable_dsa_cp():
            # TODO(jianzs): Remove this workaround after
            # `torch_npu.npu_quant_matmul` supports large weight dimensions.
            chunk_size = layer.weight.shape[1] // 2
            assert chunk_size < 65536, "Even after chunking, the weight dimension is still larger than 65536."
            layer._chunk_size = chunk_size
            layer.weight_1 = maybe_trans_nz(layer.weight.data[:, :chunk_size].contiguous())
            layer.weight_2 = maybe_trans_nz(layer.weight.data[:, chunk_size:].contiguous())
            layer.weight_1_scale = layer.weight_scale.data[:chunk_size].flatten().contiguous()
            layer.weight_2_scale = layer.weight_scale.data[chunk_size:].flatten().contiguous()
            layer.weight_1_scale_fp32 = layer.weight_1_scale.to(torch.float32)
            layer.weight_2_scale_fp32 = layer.weight_2_scale.to(torch.float32)
            layer.weight_1_offset = layer.weight_offset.data[:chunk_size].flatten().contiguous()
            layer.weight_2_offset = layer.weight_offset.data[chunk_size:].flatten().contiguous()
            del layer.weight
            del layer.weight_scale
            del layer.weight_offset
        else:
            # cast quantized weight tensors in NZ format for higher inference speed
            if self.act_quant_type == torch.int8:
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
        vllm_config = get_current_vllm_config()
        ascend_config = get_ascend_config()
        self.use_aclgraph = (
            vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not vllm_config.model_config.enforce_eager
        )
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
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
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
        if self.dynamic_eplb:
            w1 = layer.w13_weight_list
            w1_scale = layer.fused_w1_scale_list if fused_scale_flag else layer.w13_weight_scale_fp32_list
            w2 = layer.w2_weight_list
            w2_scale = layer.fused_w2_scale_list if fused_scale_flag else layer.w2_weight_scale_list
            w1_scale_bias = [torch.tensor([], dtype=torch.float32)] if fused_scale_flag else None
            w2_scale_bias = [torch.tensor([], dtype=torch.float32)] if fused_scale_flag else None

        elif fused_scale_flag and _MEGA_MOE_SUPPORTED:
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
            w1_scale_bias = [torch.tensor([], dtype=torch.float32)] if fused_scale_flag else None
            w2_scale_bias = [torch.tensor([], dtype=torch.float32)] if fused_scale_flag else None

        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=w1,
                w2=w2,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=getattr(layer, "ascend_expert_map", None),
                global_redundant_expert_num=getattr(layer, "global_redundant_expert_num", 0),
                mc2_mask=getattr(layer, "_ascend_mc2_mask", None),
                apply_router_weight_on_input=getattr(layer, "apply_router_weight_on_input", False),
                log2phy=getattr(layer, "log2phy", None),
                pertoken_scale=getattr(layer, "_ascend_pertoken_scale", None),
                activation=activation,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                swiglu_limit=layer.swiglu_limit,
                swiglu_alpha=getattr(layer, "swiglu_alpha", 1.0),
                swiglu_beta=getattr(layer, "swiglu_beta", 0.0),
            )
        )
        return final_hidden_states

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

        if self.dynamic_eplb:
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

        elif get_ascend_config().enable_fused_mc2 == 1 and _MEGA_MOE_SUPPORTED:
            layer.cann_mega_moe_w13_weight_list = list(layer.w13_weight.data.unbind(dim=0))
            layer.cann_mega_moe_w2_weight_list = list(layer.w2_weight.data.unbind(dim=0))
            layer.cann_mega_moe_fused_w1_scale_list = list(
                layer.fused_w1_scale.view(layer.w13_weight.shape[0], -1).data.unbind(dim=0)
            )
            layer.cann_mega_moe_fused_w2_scale_list = list(
                layer.fused_w2_scale.view(layer.w2_weight.shape[0], -1).data.unbind(dim=0)
            )
