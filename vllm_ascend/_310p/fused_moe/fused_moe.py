#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import torch
import torch_npu
from vllm.model_executor.layers.fused_moe import SharedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights, build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.moe_comm_method import _MoECommMethods
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import maybe_trans_nz

from .moe_comm_method import AllGatherCommImpl310


class AscendUnquantizedFusedMoEMethod310(UnquantizedFusedMoEMethod):
    """Unquantized MoE method with 310P-specific kernels.

    The MLP stage is orchestrated by ``apply_moe_mlp`` through the same
    gmm1 / act_quant / gmm2 hooks as the quantized schemes. This class
    duck-types the hook interface instead of inheriting ``AscendMoEScheme``
    because it must extend the upstream ``UnquantizedFusedMoEMethod``.
    """

    quant_type = QuantType.NONE

    def __init__(self, moe: FusedMoEConfig = None):
        super().__init__(moe=moe)

    @property
    def is_monolithic(self) -> bool:
        return False

    def maybe_make_prepare_finalize(self, routing_tables=None):
        # Ascend 310P uses its own MoE communication and forward_impl path.
        # Do not let upstream modular-kernel initialization replace it.
        return None

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)

        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2).contiguous()
        w13_data = maybe_trans_nz(w13_data)
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)

        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2).contiguous()
        w2_data = maybe_trans_nz(w2_data)
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

    def supports_fused_activation(self, activation) -> bool:
        return False

    def get_mlp_weights(self, layer):
        """Standard MLP-layout weights, returned as a ``(w1, w2)`` tuple."""
        return layer.w13_weight, layer.w2_weight

    def get_fused_mc2_weights(self, layer) -> MoEWeights:
        """Normalized weight payload for the FUSED_MC2 comm path.

        310P only registers the ALLGATHER comm method, so the standard MLP
        layout is returned as-is.
        """
        w1, w2 = self.get_mlp_weights(layer)
        return MoEWeights(w1=w1, w2=w2)

    def apply_gmm1(self, mlp_compute_input: MoEMlpComputeInput):
        """gate/up projection (gmm1), returns the pre-activation output."""
        layer = mlp_compute_input.layer
        assert layer is not None
        w1, _ = self.get_mlp_weights(layer)
        # 310P weights are pre-transposed to (E, hidden, inter) + NZ in
        # ``process_weights_after_loading``, so no ``need_trans`` handling is
        # required here (310P never sets ``need_trans``).
        return torch_npu.npu_grouped_matmul(
            x=[mlp_compute_input.hidden_states],
            weight=[w1],
            split_item=2,
            group_list_type=mlp_compute_input.group_list_type,
            group_type=0,
            group_list=mlp_compute_input.group_list,
        )[0]

    def apply_act_quant(self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor):
        # Apply router weights between activation and down-proj, then keep the
        # activation unquantized.
        if mlp_compute_input.topk_scales is not None:
            hidden_states = hidden_states * mlp_compute_input.topk_scales
        return hidden_states, None

    def apply_gmm2(self, mlp_compute_input: MoEMlpComputeInput, hidden_states, act_out_scale):
        """down projection (gmm2)."""
        layer = mlp_compute_input.layer
        assert layer is not None
        _, w2 = self.get_mlp_weights(layer)
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2],
            split_item=2,
            group_list_type=mlp_compute_input.group_list_type,
            group_type=0,
            group_list=mlp_compute_input.group_list,
        )[0]

    def apply(
        self,
        layer: "AscendRoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                layer=layer,
                quant_type=QuantType.NONE,
                dynamic_eplb=False,
                expert_map=layer.ascend_expert_map,
                global_redundant_expert_num=layer.global_redundant_expert_num,
                mc2_mask=layer.ascend_mc2_mask,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                pertoken_scale=layer.ascend_pertoken_scale,
                activation=layer.activation,
            ),
            quant_method=self,
        )
        return final_hidden_states


class AscendRoutedExperts310(AscendRoutedExperts):
    def __init__(self, *args, tid2eid=None, n_shared_experts: int = 0, **kwargs):
        super().__init__(*args, tid2eid=tid2eid, n_shared_experts=n_shared_experts, **kwargs)
        if self.quant_config is None:
            # Preserve the pre-refactor BF16 lifecycle: let upstream create
            # weights first, then install the Ascend execution method.
            self._replace_quant_method(
                AscendUnquantizedFusedMoEMethod310(
                    self.moe_config,
                )
            )


class AscendMoERunner310(AscendMoERunner):
    def __init__(
        self,
        layer_name,
        moe_config,
        router,
        routed_experts,
        enable_dbo=False,
        gate=None,
        shared_experts=None,
        shared_expert_gate=None,
        routed_input_transform=None,
        routed_output_transform=None,
        routed_scaling_factor=1,
        tid2eid=None,
        n_shared_experts: int = 0,
    ):
        super().__init__(
            layer_name=layer_name,
            moe_config=moe_config,
            router=router,
            routed_experts=routed_experts,
            enable_dbo=enable_dbo,
            gate=gate,
            shared_experts=shared_experts,
            shared_expert_gate=shared_expert_gate,
            routed_input_transform=routed_input_transform,
            routed_output_transform=routed_output_transform,
            routed_scaling_factor=routed_scaling_factor,
        )

        ascend_shared_experts = getattr(self, "ascend_shared_experts", None)
        if ascend_shared_experts is not None:
            ascend_shared_experts.multistream_overlap = False
        _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl310(self.moe_config)
