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
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights, build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.moe_utils import cumsum_group_list, maybe_normalize_mxfp_scale_layout
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts  # noqa: F401
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, FP8_METHOD, dispose_tensor

from ..base import (
    AscendLinearScheme,
    AscendMoEScheme,
    QuantType,
    WeightSwitchGatherSpec,
)
from ..registry import register_scheme

# CANN uses 36 to select FP8 E4M3FN output for situ_mx_quant.
SITU_MX_DST_TYPE_E4M3FN = 36


@register_scheme("W4A8_MXFP", "linear")
class AscendW4A8MXFPDynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W4A8_MXFP (Microscaling) quantization."""

    weight_switch_gather_specs = (
        WeightSwitchGatherSpec("weight"),
        WeightSwitchGatherSpec("weight_scale"),
    )
    weight_switch_output_gather_specs = (
        WeightSwitchGatherSpec("weight", gather_dim=1),
        WeightSwitchGatherSpec("weight_scale", gather_dim=1),
    )
    supports_weight_switch = True

    def __init__(self):
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)

    @staticmethod
    def get_weight(input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size // 2, dtype=torch.uint8)}
        return params_dict

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, input_size // self.group_size, dtype=torch.uint8)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        if isinstance(x, tuple):
            quantized_x, dynamic_scale = x
            output_dtype = torch.bfloat16
        else:
            quantized_x, dynamic_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
            output_dtype = x.dtype

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=dynamic_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=output_dtype,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            group_sizes=[0, 0, self.group_size],
        )

        return output

    def process_weights_after_loading(self, layer):
        """Cast the weight to NZ format and reshape its scale for NPU inference.

        Records the original shapes and marks the layer transformed so
        ``restore_weights_for_rl_loading`` can reverse it before an RL reload.
        """
        if getattr(layer, "_mxfp4_transformed", False):
            return
        if not hasattr(layer, "_mxfp4_original_shapes"):
            layer._mxfp4_original_shapes = {
                "weight": tuple(layer.weight.data.shape),
                "weight_scale": tuple(layer.weight_scale.data.shape),
            }
        layer.weight.data = torch_npu.npu_format_cast(
            layer.weight.data,
            ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2,
        )
        layer.weight.data = layer.weight.data.transpose(-1, -2)
        n, k = layer.weight_scale.shape
        layer.weight_scale.data = layer.weight_scale.data.reshape(n, k // 2, 2).transpose(-3, -2)
        layer._mxfp4_transformed = True

    def restore_weights_for_rl_loading(self, layer):
        """Undo the NZ/scale transform so the weight loader can reload ND weights.

        Reverses the transpose, casts the weight back to ND (format 2), and
        restores the scale's original shape.
        """
        if not getattr(layer, "_mxfp4_transformed", False):
            return
        layer.weight.data = layer.weight.data.transpose(-1, -2)
        layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 2)
        orig_scale_shape = layer._mxfp4_original_shapes["weight_scale"]
        layer.weight_scale.data = layer.weight_scale.data.transpose(-3, -2).reshape(orig_scale_shape)
        layer._mxfp4_transformed = False


@register_scheme("W4A8_MXFP", "moe")
class AscendW4A8MXFPDynamicFusedMoEMethod(AscendMoEScheme):
    """FusedMoe method for Ascend W4A8_DYNAMIC."""

    supports_eplb = True
    quant_type: QuantType = QuantType.W4A8MXFP
    act_quant_type: torch.dtype = torch.float8_e4m3fn
    fused_activations = frozenset({"silu", "situ"})

    def __init__(self):
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)
        ascend_config = get_ascend_config()
        self.dynamic_eplb = False if vllm_config.use_v2_model_runner else ascend_config.eplb_config.dynamic_eplb

    @staticmethod
    def get_weight(
        num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // 2, dtype=torch.uint8
        )
        param_dict["w2_weight"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // 2, dtype=torch.uint8
        )
        return param_dict

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // self.group_size, dtype=torch.uint8
        )

        param_dict["w2_weight_scale"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=torch.uint8
        )
        return param_dict

    def apply(
        self,
        layer: "AscendRoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        moe_comm_method = _EXTRA_CTX.moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                layer=layer,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=layer.ascend_expert_map,
                global_redundant_expert_num=layer.global_redundant_expert_num,
                mc2_mask=layer.ascend_mc2_mask,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                pertoken_scale=layer.ascend_pertoken_scale,
                activation=layer.activation,
                mxfp_act_quant_type=torch.float8_e4m3fn,
                mxfp_weight_quant_type=torch_npu.float4_e2m1fn_x2,
                mxfp_scale_dtype=torch_npu.float8_e8m0fnu,
                mxfp_per_token_scale_dtype=torch_npu.float8_e8m0fnu,
                mxfp_use_bf16=(x.dtype in [torch.bfloat16, torch.float8_e4m3fn]),
            ),
            quant_method=self,
        )

    def get_fused_mc2_weights(self, layer: torch.nn.Module) -> MoEWeights:
        return MoEWeights(
            w1=layer.w13_weight_list,
            w2=layer.w2_weight_list,
            w1_scale=layer.w13_weight_scale_list,
            w2_scale=layer.w2_weight_scale_list,
            w1_scale_bias=None,
            w2_scale_bias=None,
        )

    @staticmethod
    def get_eplb_weight_views(layer: torch.nn.Module) -> list:
        return [
            layer.w13_weight_list,
            layer.w2_weight_list,
            layer.w13_weight_scale_list,
            layer.w2_weight_scale_list,
        ]

    def process_weights_after_loading(self, layer):
        self._process_moe_weights_after_loading(layer)

    def _process_moe_weights_after_loading(self, layer, reinterpret_as_uint8: bool = False) -> None:
        """Build the unified per-expert NZ weight and ND scale lists."""
        if getattr(layer, "_mxfp4_transformed", False):
            return
        if not hasattr(layer, "_mxfp4_original_shapes"):
            layer._mxfp4_original_shapes = {
                "w13_weight": tuple(layer.w13_weight.data.shape),
                "w13_weight_scale": tuple(layer.w13_weight_scale.data.shape),
                "w2_weight": tuple(layer.w2_weight.data.shape),
                "w2_weight_scale": tuple(layer.w2_weight_scale.data.shape),
            }

        max_experts_per_card = 128
        local_num_experts = getattr(layer, "local_num_experts", layer.w13_weight.shape[0])
        assert local_num_experts <= max_experts_per_card, (
            f"W4A8 MXFP requires local_num_experts <= {max_experts_per_card} "
            "(CANN tensorList limit for per-expert NZ lists), "
            f"got {local_num_experts}. "
            "Consider increasing data-parallel-size to reduce experts per card."
        )

        if not hasattr(layer, "_mxfp4_moe_parameters"):
            layer._mxfp4_moe_parameters = {
                name: getattr(layer, name)
                for name in (
                    "w13_weight",
                    "w2_weight",
                    "w13_weight_scale",
                    "w2_weight_scale",
                )
            }

        for weight_name in ("w13_weight", "w2_weight"):
            weight = getattr(layer, weight_name)
            weight_data = weight.data.view(torch.uint8) if reinterpret_as_uint8 else weight.data
            input_dtype = torch_npu.float4_e2m1fn_x2
            use_native_fp4 = weight_name == "w13_weight" and getattr(layer, "activation", None) in (
                MoEActivation.SITU,
                "situ",
            )

            weight_list = []
            for expert in weight_data.unbind(dim=0):
                expert = expert.clone()
                if use_native_fp4:
                    expert = expert.view(torch.float4_e2m1fn_x2)
                    input_dtype = torch.float4_e2m1fn_x2
                if expert.device.type == "npu":
                    expert = torch_npu.npu_format_cast(
                        expert,
                        ACL_FORMAT_FRACTAL_NZ,
                        customize_dtype=torch.float8_e4m3fn,
                        input_dtype=input_dtype,
                    )
                # Preserve the transposed NZ view expected by WeightNz GMM.
                # Transposing before format-casting materializes an NZ tensor
                # without transpose metadata, which CANN rejects for FP8 x FP4.
                weight_list.append(expert.transpose(0, 1))
            self._update_expert_list(layer, f"{weight_name}_list", weight_list)

            scale_name = f"{weight_name}_scale"
            scale = getattr(layer, scale_name)
            scale_data = scale.data.view(torch.uint8) if reinterpret_as_uint8 else scale.data
            g_num, n_size, k_size = scale_data.shape
            scale_nd = scale_data.reshape(g_num, n_size, k_size // 2, 2).transpose(1, 2).contiguous()
            scale_list = [expert.clone() for expert in scale_nd.unbind(dim=0)]
            self._update_expert_list(layer, f"{scale_name}_list", scale_list)

        for tensor_name, parameter in layer._mxfp4_moe_parameters.items():
            dispose_tensor(parameter)
            delattr(layer, tensor_name)

        layer._mxfp4_transformed = True
        torch.npu.empty_cache()

    @staticmethod
    def _update_expert_list(layer: torch.nn.Module, name: str, values: list[torch.Tensor]) -> None:
        """Install expert tensors or refresh graph-stable buffers in place."""
        current = getattr(layer, name, None)
        if current is None:
            setattr(layer, name, values)
            return
        if len(current) != len(values):
            raise ValueError(f"{name} changed expert count across weight reloads")
        for destination, source in zip(current, values):
            if destination.shape != source.shape or destination.dtype != source.dtype:
                raise ValueError(f"{name} changed tensor schema across weight reloads")
            if destination.device.type == "npu":
                destination_format = int(torch_npu.get_npu_format(destination))
                source_format = int(torch_npu.get_npu_format(source))
                if destination_format != source_format:
                    source = torch_npu.npu_format_cast(
                        source,
                        destination_format,
                        customize_dtype=destination.dtype,
                    )
                if destination_format != int(torch_npu.Format.ND):
                    torch_npu.copy_memory_(destination, source)
                    continue
            destination.copy_(source)

    def restore_weights_for_rl_loading(self, layer):
        """Undo the NZ/scale transform so the weight loader can reload ND weights.

        Reverses the transpose, casts the expert weights back to ND (format 2),
        and restores the scales' original shapes.
        """
        if not getattr(layer, "_mxfp4_transformed", False):
            return
        orig_shapes = layer._mxfp4_original_shapes
        parameters = layer._mxfp4_moe_parameters

        for weight_name in ("w13_weight", "w2_weight"):
            scale_name = f"{weight_name}_scale"
            weight_list = getattr(layer, f"{weight_name}_list")
            scale_list = getattr(layer, f"{scale_name}_list")
            weight_nd = [
                torch_npu.npu_format_cast(weight, torch_npu.Format.ND)
                if weight.device.type == "npu"
                else weight
                for weight in weight_list
            ]
            restored_weight = torch.stack(weight_nd).transpose(1, 2).contiguous()
            restored_scale = (
                torch.stack(scale_list).transpose(1, 2).reshape(orig_shapes[scale_name]).contiguous()
            )

            weight_parameter = parameters[weight_name]
            scale_parameter = parameters[scale_name]
            if restored_weight.dtype != weight_parameter.dtype:
                restored_weight = restored_weight.view(weight_parameter.dtype)
            if restored_scale.dtype != scale_parameter.dtype:
                restored_scale = restored_scale.view(scale_parameter.dtype)
            weight_parameter.data = restored_weight.reshape(orig_shapes[weight_name])
            scale_parameter.data = restored_scale
            setattr(layer, weight_name, weight_parameter)
            setattr(layer, scale_name, scale_parameter)

        layer._mxfp4_transformed = False

    @staticmethod
    def _get_weights(layer: torch.nn.Module, name: str) -> list[torch.Tensor]:
        return getattr(layer, f"{name}_list")

    def apply_gmm1_act_quant(self, mlp_compute_input: MoEMlpComputeInput):
        hidden_states = mlp_compute_input.hidden_states
        hidden_states, pertoken_scale = self._quant_hidden_states(hidden_states, mlp_compute_input.dynamic_scale)
        layer = mlp_compute_input.layer
        assert layer is not None
        if (
            mlp_compute_input.activation == MoEActivation.SITU
            and mlp_compute_input.group_list_type in (0, 1)
            and self.group_size == 32
            and (mlp_compute_input.activation_situ_linear_beta or 0.0) > 0.0
        ):
            hidden_states, out_scale, _ = DeviceOperator.npu_grouped_matmul_situ_quant(
                x=hidden_states,
                weight=self._get_weights(layer, "w13_weight"),
                weight_scale=self._get_weights(layer, "w13_weight_scale"),
                x_scale=pertoken_scale,
                group_list=mlp_compute_input.group_list,
                group_list_type=mlp_compute_input.group_list_type,
                beta=(
                    1.0 if mlp_compute_input.activation_situ_beta is None else mlp_compute_input.activation_situ_beta
                ),
                linear_beta=mlp_compute_input.activation_situ_linear_beta or 0.0,
                mxfp_quant_dtype=self.quant_type,
            )
            dispose_tensor(mlp_compute_input.hidden_states)
            return hidden_states, maybe_normalize_mxfp_scale_layout(out_scale)
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=self._get_weights(layer, "w13_weight"),
            scale=None,
            antiquant_scale=self._get_weights(layer, "w13_weight_scale"),
            scale_dtype=None,
            per_token_scale=[pertoken_scale],
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=2,
            group_type=0,
            group_list=mlp_compute_input.group_list,
            group_list_type=mlp_compute_input.group_list_type,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            output_dtype=torch.bfloat16,
        )[0]
        dispose_tensor(mlp_compute_input.hidden_states)
        if mlp_compute_input.activation == MoEActivation.SITU:
            # SituAndMul: run the dequantized gmm1 first, then fuse the situ
            # activation with MXFP output quantization (Kimi K3 SITU activation).

            hidden_states, swiglu_out_scale = torch.ops._C_ascend.situ_mx_quant(
                x=hidden_states,
                beta=1.0 if mlp_compute_input.activation_situ_beta is None else mlp_compute_input.activation_situ_beta,
                linear_beta=mlp_compute_input.activation_situ_linear_beta or 0.0,
                activate_left=True,
                dst_type=SITU_MX_DST_TYPE_E4M3FN,
            )
            return hidden_states, maybe_normalize_mxfp_scale_layout(swiglu_out_scale)

        # The `group_index` input for the `npu_swiglu_group_quant` operator
        # currently only supports the `count` type. In the current version, the
        # `npu_swiglu_group_quant` operator performs a summation calculation on
        # `group_index`. Therefore, under the cumsum type, the last value is directly taken
        # and Avoid executing two small operators.
        if mlp_compute_input.group_list_type == 0:
            group_index = mlp_compute_input.group_list[-1:]
        else:
            group_index = cumsum_group_list(mlp_compute_input.group_list, mlp_compute_input.group_list_type, 1)
        hidden_states, out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
            hidden_states,
            topk_weight=None,
            group_index=group_index,
            dst_type=torch.float8_e4m3fn,
            quant_mode=2,
            clamp_value=mlp_compute_input.swiglu_limit,
        )
        return hidden_states, maybe_normalize_mxfp_scale_layout(out_scale)

    def apply_gmm1(self, mlp_compute_input: MoEMlpComputeInput):
        hidden_states = mlp_compute_input.hidden_states
        hidden_states, pertoken_scale = self._quant_hidden_states(hidden_states, mlp_compute_input.dynamic_scale)
        layer = mlp_compute_input.layer
        assert layer is not None
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=self._get_weights(layer, "w13_weight"),
            scale=self._get_weights(layer, "w13_weight_scale"),
            per_token_scale=[pertoken_scale],
            bias=None,
            split_item=2,
            group_type=0,
            group_list=mlp_compute_input.group_list,
            group_list_type=mlp_compute_input.group_list_type,
            output_dtype=torch.bfloat16,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        )[0]
        dispose_tensor(mlp_compute_input.hidden_states)
        return hidden_states

    def apply_act_quant(self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor):
        hidden_states, dynamic_scale = torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=torch.float8_e4m3fn)
        return hidden_states, maybe_normalize_mxfp_scale_layout(dynamic_scale)

    def apply_gmm2(self, mlp_compute_input: MoEMlpComputeInput, hidden_states, act_out_scale):
        layer = mlp_compute_input.layer
        assert layer is not None
        input_dtype = mlp_compute_input.hidden_states.dtype
        use_bf16 = input_dtype in [torch.bfloat16, torch.float8_e4m3fn]
        output_dtype = (
            input_dtype
            if input_dtype in [torch.bfloat16, torch.float16]
            else (torch.bfloat16 if use_bf16 else torch.float16)
        )
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=self._get_weights(layer, "w2_weight"),
            scale=None,
            antiquant_scale=self._get_weights(layer, "w2_weight_scale"),
            bias=None,
            per_token_scale=[act_out_scale],
            split_item=2,
            group_list_type=mlp_compute_input.group_list_type,
            group_type=0,
            group_list=mlp_compute_input.group_list,
            output_dtype=output_dtype,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
        )[0]


@register_scheme(FP8_METHOD, "ds_w4a8_moe")
class AscendW4A8MXFPDSDynamicFusedMoEMethod(AscendW4A8MXFPDynamicFusedMoEMethod):
    """FusedMoe method for DS original w4a8 mxfp quantization."""

    model_dtype = None
    quant_type: QuantType = QuantType.W4A8MXFP

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes // self.group_size,
            dtype=torch.float8_e8m0fnu,
        )

        param_dict["w2_weight_scale"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=torch.float8_e8m0fnu
        )
        return param_dict

    def process_weights_after_loading(self, layer):
        self._process_moe_weights_after_loading(layer, reinterpret_as_uint8=True)
