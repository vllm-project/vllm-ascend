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
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType, use_cann_megamoe
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights, build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.moe_utils import (
    _custom_gmm_swiglu_enabled,
    _prepare_dequant_swiglu_weight_scale,
    cumsum_group_list,
)
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts  # noqa: F401
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, dispose_tensor, maybe_trans_nz

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
    act_quant_type: torch.dtype = torch.int8
    fused_activations = frozenset({"silu", "swigluoai_uninterleave"})

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
        moe_comm_method = _EXTRA_CTX.moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                layer=layer,
                quant_type=self.quant_type,
                dynamic_eplb=self.use_expert_weight_list,
                expert_map=layer.ascend_expert_map,
                global_redundant_expert_num=layer.global_redundant_expert_num,
                mc2_mask=layer.ascend_mc2_mask,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                pertoken_scale=layer.ascend_pertoken_scale,
                activation=activation,
                lora_context=lora_context,
            ),
            quant_method=self,
        )

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

    def _get_mlp_weights(self, layer: torch.nn.Module) -> tuple:
        """Return (w1, w1_scale, w2, w2_scale) in the standard MLP layout."""
        if self.use_expert_weight_list:
            return (
                layer.w13_weight_list,
                layer.w13_weight_scale_fp32_list,
                layer.w2_weight_list,
                layer.w2_weight_scale_list,
            )
        return (
            [layer.w13_weight],
            [layer.w13_weight_scale_fp32],
            [layer.w2_weight],
            [layer.w2_weight_scale],
        )

    def get_fused_mc2_weights(self, layer: torch.nn.Module) -> MoEWeights:
        """Normalized weight payload for the FUSED_MC2 comm path."""
        activation = getattr(layer, "activation", "silu")
        act_name = getattr(activation, "value", activation)
        fused_scale_flag = (
            _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2
            and get_ascend_config().enable_fused_mc2 == 1
            and act_name != "swigluoai_uninterleave"
        )
        use_mega_moe = fused_scale_flag and _EXTRA_CTX.use_mega_moe
        if self.use_expert_weight_list:
            if use_mega_moe:
                return MoEWeights(
                    w1=layer.w13_weight_list,
                    w2=layer.w2_weight_list,
                    w1_scale=[scale.reshape(-1) for scale in layer.fused_w1_scale_list],
                    w2_scale=[scale.reshape(-1) for scale in layer.fused_w2_scale_list],
                    w1_scale_bias=None,
                    w2_scale_bias=None,
                )
            else:
                return MoEWeights(
                    w1=layer.w13_weight_list,
                    w2=layer.w2_weight_list,
                    w1_scale=layer.fused_w1_scale_list if fused_scale_flag else layer.w13_weight_scale_fp32_list,
                    w2_scale=layer.fused_w2_scale_list if fused_scale_flag else layer.w2_weight_scale_list,
                    w1_scale_bias=layer.fused_w1_scale_bias if fused_scale_flag else None,
                    w2_scale_bias=layer.fused_w2_scale_bias if fused_scale_flag else None,
                )
        elif use_mega_moe:
            return MoEWeights(
                w1=layer.cann_mega_moe_w13_weight_list,
                w2=layer.cann_mega_moe_w2_weight_list,
                w1_scale=layer.cann_mega_moe_fused_w1_scale_list,
                w2_scale=layer.cann_mega_moe_fused_w2_scale_list,
                w1_scale_bias=None,
                w2_scale_bias=None,
            )
        return MoEWeights(
            w1=[layer.w13_weight],
            w2=[layer.w2_weight],
            w1_scale=[layer.fused_w1_scale] if fused_scale_flag else [layer.w13_weight_scale_fp32],
            w2_scale=[layer.fused_w2_scale] if fused_scale_flag else [layer.w2_weight_scale],
            w1_scale_bias=layer.fused_w1_scale_bias if fused_scale_flag else None,
            w2_scale_bias=layer.fused_w2_scale_bias if fused_scale_flag else None,
        )

    def get_mlp_weights(self, layer: torch.nn.Module) -> MoEWeights:
        """Standard MLP-layout weights used by the quantized MoE LoRA backend."""
        w1, w1_scale, w2, w2_scale = self._get_mlp_weights(layer)
        return MoEWeights(
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )

    def apply_gmm1_act_quant(self, mlp_compute_input: MoEMlpComputeInput):
        hidden_states = mlp_compute_input.hidden_states
        hidden_states, pertoken_scale = self._quant_hidden_states(hidden_states, mlp_compute_input.dynamic_scale)
        layer = mlp_compute_input.layer
        w1, w1_scale, _, _ = self._get_mlp_weights(layer)
        activation = mlp_compute_input.activation
        fusion = mlp_compute_input.fusion
        dynamic_eplb = mlp_compute_input.dynamic_eplb
        is_swigluoai_uninterleave = activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE
        is_mc2 = _EXTRA_CTX.moe_comm_type == MoECommType.MC2

        if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb, activation):
            # eplb branch
            hidden_states, swiglu_out_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                x=hidden_states,
                weight=w1,
                weight_scale=w1_scale,
                x_scale=pertoken_scale,
                group_list=cumsum_group_list(mlp_compute_input.group_list, mlp_compute_input.group_list_type, 0),
                swiglu_limit=mlp_compute_input.swiglu_limit,
            )
        elif fusion and not is_swigluoai_uninterleave:
            hidden_states, swiglu_out_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz(
                x=hidden_states,
                weight=w1[0],
                group_list=cumsum_group_list(mlp_compute_input.group_list, mlp_compute_input.group_list_type, 0),
                weight_scale=w1_scale[0],
                x_scale=pertoken_scale,
                bias=None,
                swiglu_limit=mlp_compute_input.swiglu_limit,
            )
        # The following 2 branches are prepared to those who choose to not use grouped_matmul_swiglu_quant, it will be
        # deleted after grouped_matmul_swiglu_quant is the default option. For now, when fusion_ops_gmmswigluquant is
        # set to false, grouped_matmul_swiglu_quant won't be used.
        elif is_mc2:
            # For those who choose to not use grouped_matmul_swiglu_quant and in decode stage,
            # use gmm1 + dequant_swiglu_quant instead. This path will double the memory of activation tensor.
            # This branch will be deleted when grouped_matmul_swiglu_quant supports swigluoai_uninterleave
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=w1,
                split_item=3,
                group_list_type=mlp_compute_input.group_list_type,
                group_type=0,
                group_list=mlp_compute_input.group_list,
                output_dtype=torch.int32,
            )[0]
            dequant_swiglu_kwargs = {
                "x": hidden_states,
                "weight_scale": _prepare_dequant_swiglu_weight_scale(w1_scale, is_swigluoai_uninterleave),
                "activation_scale": pertoken_scale,
                "bias": None,
                "quant_scale": None,
                "quant_offset": None,
                "group_index": cumsum_group_list(mlp_compute_input.group_list, mlp_compute_input.group_list_type, 1),
                "activate_left": True,
                "quant_mode": 1,
            }
            if is_swigluoai_uninterleave:
                dequant_swiglu_kwargs.update(
                    {
                        "swiglu_mode": 1,
                        "clamp_limit": mlp_compute_input.swiglu_limit,
                        "glu_alpha": mlp_compute_input.swiglu_alpha,
                        "glu_bias": mlp_compute_input.swiglu_beta,
                    }
                )
            hidden_states, swiglu_out_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(**dequant_swiglu_kwargs)
        else:
            # For those who choose to not use grouped_matmul_swiglu_quant and in prefill stage,
            # use gmm1 + activation + quant instead.
            hidden_states = self.apply_gmm1(mlp_compute_input)
            if is_swigluoai_uninterleave:
                hidden_states = torch_npu.npu_clipped_swiglu(
                    hidden_states,
                    interleaved=False,
                    alpha=mlp_compute_input.swiglu_alpha,
                    limit=mlp_compute_input.swiglu_limit,
                    bias=mlp_compute_input.swiglu_beta,
                )
                hidden_states, swiglu_out_scale = self.apply_act_quant(mlp_compute_input, hidden_states)
            elif HAS_TRITON:
                from vllm_ascend.ops.triton.activation.swiglu_quant import swiglu_quant

                hidden_states, swiglu_out_scale = swiglu_quant(
                    hidden_states,
                    group_list=mlp_compute_input.group_list,
                    group_list_type=mlp_compute_input.group_list_type,
                )
            else:
                hidden_states = torch_npu.npu_swiglu(hidden_states)
                hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(hidden_states)

        dispose_tensor(mlp_compute_input.hidden_states)
        return hidden_states, swiglu_out_scale

    def apply_gmm1(self, mlp_compute_input: MoEMlpComputeInput):
        hidden_states = mlp_compute_input.hidden_states
        hidden_states, pertoken_scale = self._quant_hidden_states(hidden_states, mlp_compute_input.dynamic_scale)
        layer = mlp_compute_input.layer
        w1, w1_scale, _, w2_scale = self._get_mlp_weights(layer)
        w2_scale_dtype = w2_scale[0].dtype
        scale = [w1_scale[0].to(w2_scale_dtype)]
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w1 if isinstance(w1, list) else [w1],
            scale=scale,
            bias=None,
            per_token_scale=[pertoken_scale],
            split_item=2,
            group_type=0,
            group_list=mlp_compute_input.group_list,
            group_list_type=mlp_compute_input.group_list_type,
            output_dtype=w2_scale_dtype,
        )[0]
        dispose_tensor(mlp_compute_input.hidden_states)
        return hidden_states

    def apply_act_quant(self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor):
        return torch_npu.npu_dynamic_quant(hidden_states, dst_type=self.act_quant_type)

    def apply_gmm2(self, mlp_compute_input: MoEMlpComputeInput, hidden_states, act_out_scale):
        _, _, w2, w2_scale = self._get_mlp_weights(mlp_compute_input.layer)
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w2,
            scale=w2_scale,
            bias=None,
            per_token_scale=[act_out_scale],
            split_item=2,
            group_list_type=mlp_compute_input.group_list_type,
            group_type=0,
            group_list=mlp_compute_input.group_list,
            output_dtype=w2_scale[0].dtype,
        )[0]
