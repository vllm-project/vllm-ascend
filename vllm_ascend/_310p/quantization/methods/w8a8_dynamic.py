#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from typing import Any

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights, build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.quantization.methods.base import AscendMoEScheme, QuantType
from vllm_ascend.utils import maybe_trans_nz

from .registry import register_scheme
from .w8a8_base import AscendW8A8Linear310pScheme


@register_scheme("W8A8_DYNAMIC", "moe")
class AscendW8A8DynamicFusedMoEMethod310(AscendMoEScheme):
    """310P-only FusedMoE method for Ascend W8A8_DYNAMIC.

    Notes:
      - This scheme is discovered via 310P local registry.
    """

    # Declare the quantization type for this scheme
    quant_type: QuantType = QuantType.W8A8
    # Activation quant dtype used by the MLP gmm hooks.
    act_quant_type: torch.dtype = torch.int8
    # 310P gmm1+swiglu+quant is fused inside npu_quant_grouped_matmul_dequant
    # + npu_swiglu, so silu is handled by ``apply_gmm1_act_quant``.
    fused_activations = frozenset({"silu"})

    def __init__(self):
        self.ep_group = get_ep_group()
        vllm_config = get_current_vllm_config()
        self.in_dtype = vllm_config.model_config.dtype

    def get_weight(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        # Fused gate_up_proj (column parallel)
        param_dict["w13_weight"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes, dtype=torch.int8
        )
        # down_proj (row parallel)
        param_dict["w2_weight"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition, dtype=torch.int8
        )
        return param_dict

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
        )
        param_dict["w13_weight_offset"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=params_dtype
        )
        param_dict["w2_weight_scale"] = torch.empty(num_experts, hidden_sizes, 1, dtype=torch.float32)
        param_dict["w2_weight_offset"] = torch.empty(num_experts, hidden_sizes, 1, dtype=params_dtype)
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
        topk_weights = topk_weights.to(self.in_dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method

        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                layer=layer,
                quant_type=self.quant_type,
                dynamic_eplb=False,
                expert_map=layer.ascend_expert_map,
                global_redundant_expert_num=layer.global_redundant_expert_num,
                mc2_mask=layer.ascend_mc2_mask,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                pertoken_scale=layer.ascend_pertoken_scale,
                activation=getattr(layer, "activation", "silu"),
            ),
            quant_method=self,
        )
        return final_hidden_states

    def _get_group_list(self, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
        """Return the cumulative-sum group_list expected by 310P kernels."""
        group_list = mlp_compute_input.group_list
        if mlp_compute_input.group_list_type == 1:
            # Convert group_list to cumulative sum format if group_list is count format
            group_list = torch.cumsum(group_list, dim=0)
        return group_list

    def _get_mlp_weights(self, layer: torch.nn.Module) -> tuple:
        """Return (w1, w1_scale, w2, w2_scale) in the standard MLP layout."""
        return (
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
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

    def get_fused_mc2_weights(self, layer: torch.nn.Module) -> MoEWeights:
        """Normalized weight payload for the FUSED_MC2 comm path.

        310P only registers the ALLGATHER comm method (no FUSED_MC2 / mega-moe
        kernels), so the standard MLP layout is returned as-is.
        """
        return self.get_mlp_weights(layer)

    def apply_gmm1_act_quant(self, mlp_compute_input: MoEMlpComputeInput):
        """Fused gmm1 (quant + dequant) + swiglu via 310P kernels.

        ``npu_quant_grouped_matmul_dequant`` quantizes the hidden states to
        int8 internally (``quant_mode="pertoken"``) and dequantizes the gmm1
        output, so the activation scale is not needed by ``apply_gmm2``.
        """
        layer = mlp_compute_input.layer
        assert layer is not None
        w1, w1_scale, _, _ = self._get_mlp_weights(layer)
        hidden_states = torch_npu.npu_quant_grouped_matmul_dequant(
            x=mlp_compute_input.hidden_states,
            quantized_weight=w1,
            weight_scale=w1_scale,
            group_list=self._get_group_list(mlp_compute_input),
            quant_mode="pertoken",
        )
        hidden_states = torch_npu.npu_swiglu(hidden_states)
        return hidden_states, None

    def apply_gmm1(self, mlp_compute_input: MoEMlpComputeInput):
        """gmm1 (quant + dequant) only, returns the pre-activation output."""
        layer = mlp_compute_input.layer
        assert layer is not None
        w1, w1_scale, _, _ = self._get_mlp_weights(layer)
        return torch_npu.npu_quant_grouped_matmul_dequant(
            x=mlp_compute_input.hidden_states,
            quantized_weight=w1,
            weight_scale=w1_scale,
            group_list=self._get_group_list(mlp_compute_input),
            quant_mode="pertoken",
        )

    def apply_act_quant(self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor):
        # The 310P gmm2 kernel re-quantizes the activation internally with
        # ``quant_mode="pertoken"``, so no explicit quantization is needed.
        return hidden_states, None

    def apply_gmm2(self, mlp_compute_input: MoEMlpComputeInput, hidden_states, act_out_scale):
        """down projection (gmm2, quant + dequant)."""
        layer = mlp_compute_input.layer
        assert layer is not None
        _, _, w2, w2_scale = self._get_mlp_weights(layer)
        return torch_npu.npu_quant_grouped_matmul_dequant(
            x=hidden_states,
            quantized_weight=w2,
            weight_scale=w2_scale,
            group_list=self._get_group_list(mlp_compute_input),
            quant_mode="pertoken",
        )

    def process_weights_after_loading(self, layer):
        layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)
        layer.w2_weight.data = maybe_trans_nz(layer.w2_weight.data)
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(layer.w13_weight_scale.data.shape[0], -1)
        layer.w13_weight_offset.data = layer.w13_weight_offset.data.view(layer.w13_weight_offset.data.shape[0], -1)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.view(layer.w2_weight_scale.data.shape[0], -1)
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.view(layer.w2_weight_offset.data.shape[0], -1)


@register_scheme("W8A8_DYNAMIC", "linear")
class AscendW8A8DynamicLinearMethod310(AscendW8A8Linear310pScheme):
    """310P-only W8A8 dynamic linear scheme.

    Notes:
      - This scheme is discovered via 310P local registry.
    """

    def get_perchannel_param(
        self,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        params["weight_scale"] = torch.empty(output_size, 1, dtype=torch.float32)
        params["weight_offset"] = torch.empty(output_size, 1, dtype=torch.float32)
        return params

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        # NOTE(310P):
        # - There is an accuracy issue currently, which is expected to be fixed in the next version.
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        need_unsqz = False
        if pertoken_scale.dim() == 2:
            need_unsqz = True
            quantized_x = quantized_x.squeeze(dim=1)
            pertoken_scale = pertoken_scale.squeeze(dim=1)

        # NOTE(310P):
        # - Currently, W8A8 dynamic quantization supports only symmetric quantization.
        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight.data,
            layer.weight_scale,
            pertoken_scale=pertoken_scale,
            bias=bias,
            output_dtype=x.dtype,
        )
        if need_unsqz:
            output = output.unsqueeze(dim=1)
        return output

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # cast quantized weight tensors in NZ format for higher inference speed
        layer.weight.data = maybe_trans_nz(layer.weight.data).transpose(0, 1)
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_offset.data = layer.weight_offset.data.flatten()
