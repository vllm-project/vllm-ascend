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
"""Ascend W8A16_FP8 quantization helpers and fused MoE method."""

from typing import Any

import torch
import torch_npu
from vllm.config import get_current_vllm_config

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts  # noqa: F401
from vllm_ascend.utils import dispose_tensor

from ..base import AscendMoEScheme, QuantType
from ..registry import register_scheme


@register_scheme("W8A16_FP8", "moe")
class AscendW8A16FP8FusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for Ascend W8A16_FP8 (weight-only FP8 quantization).

    Key characteristics:
    - Weights are stored in torch.float8_e4m3fn, activations remain bf16/fp16.
    - Dequant is fused into npu_grouped_matmul via antiquant_scale; float
      formats are symmetric around 0, so no antiquant_offset is passed.
    - Per-channel scales use the activation dtype (bfloat16/float16) with
      shape [num_experts, output_channels].

    After :meth:`process_weights_after_loading`, ``apply`` consumes:

    - ``w13_weight``: ``torch.float8_e4m3fn``,
      ``[num_experts, hidden_sizes, 2 * moe_intermediate_size]``.
    - ``w2_weight``: ``torch.float8_e4m3fn``,
      ``[num_experts, moe_intermediate_size, hidden_sizes]``.
    - ``w13_weight_scale``: activation dtype,
      ``[num_experts, 2 * moe_intermediate_size]``.
    - ``w2_weight_scale``: activation dtype, ``[num_experts, hidden_sizes]``.
    """

    supports_eplb = False

    quant_type: QuantType = QuantType.W8A16FP
    act_quant_type: torch.dtype | None = None
    # W8A16FP keeps activations unquantized and has no fused gmm1+act+quant
    # kernel: the action method runs gmm1 -> activation -> gmm2.
    fused_activations = frozenset()

    def __init__(self) -> None:
        vllm_config = get_current_vllm_config()
        self.dynamic_eplb = False if vllm_config.use_v2_model_runner else get_ascend_config().eplb_config.dynamic_eplb

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Create weight tensors for w13 (gate+up) and w2 (down) projections.

        W8A16FP uses native float8_e4m3fn with the full K dimension.
        """
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes,
            dtype=torch.float8_e4m3fn,
        )
        param_dict["w2_weight"] = torch.empty(
            num_experts,
            hidden_sizes,
            intermediate_size_per_partition,
            dtype=torch.float8_e4m3fn,
        )
        return param_dict

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Create per-channel weight scale tensors.

        For W8A16FP per-channel dequant, the scale shape is [E, N] (one scale
        per output channel per expert) and the dtype matches the activation
        dtype (bfloat16/float16).
        """
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            dtype=params_dtype,
        )
        param_dict["w2_weight_scale"] = torch.empty(
            num_experts,
            hidden_sizes,
            dtype=params_dtype,
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
            ),
            quant_method=self,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Prepare weights for NPU grouped matmul.

        Only transpose + contiguous for per-channel antiquant_scale alignment.
        """
        # w13_weight: [E, 2N, K] -> [E, K, 2N] (transpose for per-channel dequant)
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        # w2_weight: [E, K, N] -> [E, N, K]
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)

        layer.w13_weight_scale.data = layer.w13_weight_scale.data.contiguous()
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.contiguous()

    def apply_gmm1(self, mlp_compute_input: MoEMlpComputeInput):
        layer = mlp_compute_input.layer
        assert layer is not None
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[mlp_compute_input.hidden_states],
            weight=[layer.w13_weight],
            antiquant_scale=[layer.w13_weight_scale],
            split_item=2,
            group_list_type=mlp_compute_input.group_list_type,
            group_type=0,
            group_list=mlp_compute_input.group_list,
            output_dtype=layer.w2_weight_scale.dtype,
        )[0]
        dispose_tensor(mlp_compute_input.hidden_states)
        return hidden_states

    def apply_act_quant(self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor):
        # A16 activation: no (re)quantization.
        return hidden_states, None

    def apply_gmm2(self, mlp_compute_input: MoEMlpComputeInput, hidden_states, act_out_scale):
        layer = mlp_compute_input.layer
        assert layer is not None
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            antiquant_scale=[layer.w2_weight_scale],
            split_item=2,
            group_list_type=mlp_compute_input.group_list_type,
            group_type=0,
            group_list=mlp_compute_input.group_list,
            output_dtype=layer.w2_weight_scale.dtype,
        )[0]
