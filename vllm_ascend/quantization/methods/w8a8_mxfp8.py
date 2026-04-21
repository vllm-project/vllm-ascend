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

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.device.mxfp_compat import (
    FLOAT8_E8M0FNU_DTYPE,
    ensure_mxfp8_linear_available,
    ensure_mxfp8_moe_available,
)
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input

from .base import AscendLinearScheme, AscendMoEScheme, QuantType
from .registry import register_scheme


@register_scheme("W8A8_MXFP8", "linear")
class AscendW8A8MXFP8DynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W8A8_MXFP8 (Microscaling FP8) quantization.

    This scheme uses microscaling FP8 quantization with per-group scales.
    The activation is dynamically quantized to FP8 (E4M3FN format) with
    microscaling, and weights are stored in FP8 format with per-group scales.
    """

    model_dtype = None

    def __init__(self):
        ensure_mxfp8_linear_available("W8A8_MXFP8 linear quantization")
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn)}
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
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        # reshape x for Qwen VL models
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])
        quantized_x, dynamic_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        pertoken_scale = dynamic_scale
        output_dtype = x.dtype
        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            pertoken_scale=pertoken_scale,
            pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            bias=bias,
            output_dtype=output_dtype,
            group_sizes=[1, 1, self.group_size],
        )
        # reshape output for Qwen VL models
        if len(original_shape) > 2:
            output = output.view(*original_shape[:-1], -1)

        return output

    def process_weights_after_loading(self, layer):
        """Process weights after loading for MXFP8 inference.

        This method transforms weights for NPU MXFP8 computation:
        - weight: (output_size, input_size) -> (input_size, output_size)
        - weight_scale: (n_dim, k_dim) -> (k_dim//2, n_dim, 2)

        For RL training scenarios where weights need to be reloaded multiple times,
        this method stores original shapes and can be called multiple times safely.
        Use restore_weights_for_rl_loading() before weight reload, then call this
        method again after loading.
        """

        # Check if already transformed to avoid double transformation
        if getattr(layer, "_mxfp8_transformed", False):
            return

        # Store original shapes for RL weight reloading
        # Only store on first call (when shapes are in original format)
        if not hasattr(layer, "_mxfp8_original_shapes"):
            layer._mxfp8_original_shapes = {
                "weight": tuple(layer.weight.data.shape),
                "weight_scale": tuple(layer.weight_scale.data.shape),
            }

        n_dim, k_dim = layer.weight_scale.data.shape
        layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)
        layer.weight.data = layer.weight.data.transpose(0, 1)
        layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1)

        # Mark as transformed
        layer._mxfp8_transformed = True

    def restore_weights_for_rl_loading(self, layer):
        """Restore weights to original shapes for RL weight reloading.

        This method must be called BEFORE model.load_weights() in RL training
        loops to restore the tensors to their original shapes that the weight
        loader expects.

        After weight loading, call process_weights_after_loading() again to
        re-apply the MXFP8 transformations.

        Shape transformations reversed:
        - weight: (input_size, output_size) -> (output_size, input_size)
        - weight_scale: (k_dim//2, n_dim, 2) -> (n_dim, k_dim)
        """

        if not getattr(layer, "_mxfp8_transformed", False):
            # Not transformed, nothing to restore
            return

        if not hasattr(layer, "_mxfp8_original_shapes"):
            raise RuntimeError(
                "Cannot restore weights: original shapes not recorded. "
                "This should not happen if process_weights_after_loading was called first."
            )

        orig_shapes = layer._mxfp8_original_shapes
        orig_scale_shape = orig_shapes["weight_scale"]

        # Restore weight: (input_size, output_size) -> (output_size, input_size)
        target_weight = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = layer.weight.data.transpose(0, 1)
        layer.weight.data.copy_(target_weight)

        # Restore weight_scale: (k_dim//2, n_dim, 2) -> (n_dim, k_dim)
        # Current shape: (k_dim//2, n_dim, 2)
        # Target shape: (n_dim, k_dim)
        target_scale = layer.weight_scale.data.transpose(0, 1).reshape(orig_scale_shape).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1).view(orig_scale_shape)
        layer.weight_scale.data.copy_(target_scale)

        # Mark as not transformed (ready for weight loading)
        layer._mxfp8_transformed = False


@register_scheme("W8A8_MXFP8", "moe")
class AscendW8A8MXFP8DynamicFusedMoEMethod(AscendMoEScheme):
    """FusedMoe method for Ascend W8A8_DYNAMIC."""

    model_dtype = None
    quant_type: QuantType = QuantType.MXFP8

    def __init__(self):
        ensure_mxfp8_moe_available("W8A8_MXFP8 MoE quantization")
        self.ep_group = get_ep_group()

        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)
        ascend_config = get_ascend_config()
        self.use_aclgraph = (
            vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not vllm_config.model_config.enforce_eager
        )
        self.dynamic_eplb = ascend_config.eplb_config.dynamic_eplb

    @staticmethod
    def get_weight(
        num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes, dtype=torch.float8_e4m3fn
        )
        param_dict["w2_weight"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition, dtype=torch.float8_e4m3fn
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
        enable_force_load_balance: bool = True,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = global_num_experts - global_redundant_expert_num
        assert router_logits.shape[1] == expected, "Number of global experts mismatch (excluding redundancy)"
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

        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance:
            random_matrix = torch.rand(
                topk_ids.size(0), global_num_experts - global_redundant_expert_num, device=topk_ids.device
            )
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                mxfp_act_quant_type=torch.float8_e4m3fn,
                mxfp_weight_quant_type=torch.float8_e4m3fn,
                mxfp_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                mxfp_per_token_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                mxfp_use_bf16=(x.dtype == torch.bfloat16),
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
            )
        )

    def process_weights_after_loading(self, layer):
        """Process weights after loading for MXFP8 inference.

        This method transforms weights for NPU MXFP8 computation:
        - w13_weight: (g_num, n_size, k_size) -> (g_num, k_size, n_size)
        - w2_weight: (g_num, n_size, k_size) -> (g_num, k_size, n_size)
        - w13_weight_scale: (g_num, n_size, k_size) -> (g_num, k_size//2, n_size, 2)
        - w2_weight_scale: (g_num, n_size, k_size) -> (g_num, k_size//2, n_size, 2)

        For RL training scenarios where weights need to be reloaded multiple times,
        this method stores original shapes and can be called multiple times safely.
        Use restore_weights_for_rl_loading() before weight reload, then call this
        method again after loading.
        """

        # Check if already transformed to avoid double transformation
        if getattr(layer, "_mxfp8_transformed", False):
            return

        # Store original shapes for RL weight reloading
        # Only store on first call (when shapes are in original format)
        if not hasattr(layer, "_mxfp8_original_shapes"):
            layer._mxfp8_original_shapes = {
                "w13_weight": tuple(layer.w13_weight.data.shape),
                "w13_weight_scale": tuple(layer.w13_weight_scale.data.shape),
                "w2_weight": tuple(layer.w2_weight.data.shape),
                "w2_weight_scale": tuple(layer.w2_weight_scale.data.shape),
            }

        g_num, n_size, k_size = layer.w13_weight_scale.shape
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.reshape(g_num, n_size, k_size // 2, 2)
        g_num, n_size, k_size = layer.w2_weight_scale.shape
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.reshape(g_num, n_size, k_size // 2, 2)
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(1, 2)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(1, 2)

        # Mark as transformed
        layer._mxfp8_transformed = True

    def restore_weights_for_rl_loading(self, layer):
        """Restore weights to original shapes for RL weight reloading.

        This method must be called BEFORE model.load_weights() in RL training
        loops to restore the tensors to their original shapes that the weight
        loader expects.

        After weight loading, call process_weights_after_loading() again to
        re-apply the MXFP8 transformations.

        Shape transformations reversed:
        - w13_weight: (g_num, k_size, n_size) -> (g_num, n_size, k_size)
        - w2_weight: (g_num, k_size, n_size) -> (g_num, n_size, k_size)
        - w13_weight_scale: (g_num, k_size//2, n_size, 2) -> (g_num, n_size, k_size)
        - w2_weight_scale: (g_num, k_size//2, n_size, 2) -> (g_num, n_size, k_size)
        """

        if not getattr(layer, "_mxfp8_transformed", False):
            # Not transformed, nothing to restore
            return

        if not hasattr(layer, "_mxfp8_original_shapes"):
            raise RuntimeError(
                "Cannot restore weights: original shapes not recorded. "
                "This should not happen if process_weights_after_loading was called first."
            )

        orig_shapes = layer._mxfp8_original_shapes

        def _restore(weight_key: str, scale_key: str):
            """Helper to restore a single MoE weight and its scale using safe memory copies."""
            # --- 1. Restore Weight ---
            weight_tensor = getattr(layer, weight_key)
            target_weight = weight_tensor.data.transpose(1, 2).contiguous()
            weight_tensor.data = weight_tensor.data.transpose(1, 2)
            weight_tensor.data.copy_(target_weight)

            # --- 2. Restore Weight Scale ---
            scale_tensor = getattr(layer, scale_key)
            orig_scale_shape = orig_shapes[scale_key]

            target_scale = scale_tensor.data.transpose(1, 2).reshape(orig_scale_shape).contiguous()
            scale_tensor.data = scale_tensor.data.transpose(1, 2).view(orig_scale_shape)
            scale_tensor.data.copy_(target_scale)

        _restore("w13_weight", "w13_weight_scale")
        _restore("w2_weight", "w2_weight_scale")

        # Mark as not transformed (ready for weight loading)
        layer._mxfp8_transformed = False
