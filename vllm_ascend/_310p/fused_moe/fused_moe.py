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
from collections.abc import Callable

import torch
import torch.nn.functional as F
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import FusedMoE, UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.experts_selector import zero_experts_compute
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult, _MoECommMethods
from vllm_ascend.quantization.methods.base import QuantType

from .experts_selector import select_experts
from .moe_comm_method import AllGatherCommImpl310


class AscendUnquantizedFusedMoEMethod310(UnquantizedFusedMoEMethod):
    def __init__(self, moe: FusedMoEConfig = None):
        super().__init__(moe=moe)

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)

        # Fused gate_up_proj (column parallel)
        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)
        # down_proj (row parallel)
        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2).contiguous()
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: torch.Tensor | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)

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

        if zero_expert_num > 0 and zero_expert_type is not None:
            topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=global_num_experts,
                zero_expert_type=zero_expert_type,
                hidden_states=x,
            )

        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = get_forward_context().moe_comm_method
        final_hidden_states = moe_comm_method.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states += zero_expert_result
        return final_hidden_states


class AscendFusedMoE310(FusedMoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.global_num_experts = kwargs["num_experts"]

        if self.quant_config is None:
            self.quant_method = AscendUnquantizedFusedMoEMethod310(self.moe_config)
        else:
            self.quant_method = self.quant_config.get_quant_method(self, self.layer_name)

        assert self.quant_method is not None

        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        self.moe_config.ep_group = get_ep_group()
        self.moe_config.supports_eplb = False

        # init moe
        self.global_expert_map = None
        self.local_expert_map = None
        if self.moe_config.ep_size > 1:
            self.global_expert_map, self.local_expert_map = self.init_experts_map(self.moe_config)
        self.local_num_experts = (
            torch.sum(self.local_expert_map != -1).item()
            if self.local_expert_map is not None
            else self.global_num_experts
        )

        self.moe_config.num_experts = self.global_num_experts
        self.moe_config.num_local_experts = self.local_num_experts
        self.moe_config.global_redundant_expert_num = 0

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": self.hidden_size,
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            "params_dtype": self.params_dtype,
            "weight_loader": self.weight_loader,
        }

        self.quant_method.create_weights(layer=self, **moe_quant_params)
        self.quant_type = self.get_quant_type()

        _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl310(self.moe_config)

    def init_experts_map(self, moe_config):
        """
        Initialize expert mapping for MoE (Mixture of Experts) model.

        This function creates mappings between global expert indices and local expert indices
        for each rank in the expert parallel group. It divides the total experts among
        different ranks and creates both global and local expert maps that are used
        during MoE computation to determine which experts are handled by which rank.

        Args:
            moe_config: Configuration object containing MoE parameters including
                       number of experts, expert parallel size, and expert parallel rank.

        Returns:
            tuple: A tuple containing:
                   - global_expert_map: Stack of expert maps for all ranks
                   - local_expert_map: Expert map for the current rank (transferred to NPU)
        """
        n_experts = moe_config.num_experts
        ep_size = moe_config.ep_size
        all_experts = torch.arange(n_experts, dtype=torch.int32)
        experts_groups = all_experts.chunk(ep_size)
        global_expert_map = []
        local_expert_map = None
        for rankid in range(ep_size):
            expert_map = torch.full((n_experts,), -1, dtype=torch.int32)
            local_experts = experts_groups[rankid]
            expert_map[local_experts] = torch.arange(local_experts.shape[0], dtype=torch.int32)
            global_expert_map.append(expert_map)
            if rankid == moe_config.ep_rank:
                local_expert_map = expert_map.npu()
        return torch.stack(global_expert_map), local_expert_map

    def get_quant_type(self) -> QuantType:
        quant_method = self.quant_method
        if not hasattr(quant_method, "quant_method") or quant_method.quant_method is None:
            return QuantType.NONE

        method = quant_method.quant_method
        quant_type = getattr(method, "quant_type", QuantType.NONE)
        if quant_type not in [QuantType.NONE, QuantType.W8A8]:
            raise RuntimeError("Only Unquant and W8A8 is supported.")
        return quant_type

    def forward_impl(  # type: ignore[override]
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        assert self.quant_method is not None
        assert self.routed_scaling_factor == 1.0, "routed_scaling_factor != 1.0 is not supported."
        forward_context = get_forward_context()

        hidden_states, router_logits, _, context_metadata = forward_context.moe_comm_method.prepare(
            hidden_states=hidden_states, router_logits=router_logits, quant_type=self.quant_type
        )

        # Matrix multiply.
        fused_experts_results: FusedExpertsResult = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            use_grouped_topk=self.use_grouped_topk,
            top_k=self.top_k,
            router_logits=router_logits,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            global_num_experts=self.global_num_experts,
            expert_map=self.local_expert_map,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )

        routed_out = forward_context.moe_comm_method.finalize(
            hidden_states=fused_experts_results.routed_out,
            reduce_results=self.reduce_results,
            context_metadata=context_metadata,
        )

        return routed_out


class AscendSharedFusedMoE310(SharedFusedMoE, AscendFusedMoE310):
    def __init__(
        self,
        shared_experts: torch.nn.Module,
        gate: torch.nn.Module | None = None,
        use_overlapped: bool = True,
        routed_input_transform: torch.nn.Module | None = None,
        **kwargs,
    ):
        AscendFusedMoE310.__init__(self, **kwargs)
        self._routed_input_transform = routed_input_transform
        self._shared_experts = shared_experts
        self.use_overlapped = use_overlapped
        self.shared_expert_stream = None
        self._gate = gate

    def _shared_experts_part1(self, hidden_states: torch.Tensor):
        """Execute gate_up projection for shared experts."""
        shared_gate_up, _ = self._shared_experts.gate_up_proj(hidden_states)  # type: ignore
        return shared_gate_up

    def _shared_experts_part2(self, hidden_states: torch.Tensor, shared_gate_up: torch.Tensor):
        """Execute activation and down projection for shared experts."""
        shared_act = self._shared_experts.act_fn(shared_gate_up)  # type: ignore
        shared_out, _ = self._shared_experts.down_proj(shared_act)  # type: ignore

        # Qwen3-Next specific gating mechanism.
        if hasattr(self._shared_experts, "expert_gate") and self._shared_experts.expert_gate is not None:
            gate_out, _ = self._shared_experts.expert_gate(hidden_states)  # type: ignore
            shared_out = F.sigmoid(gate_out) * shared_out
        return shared_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._shared_experts is None:
            fused_out = AscendFusedMoE310.forward(
                self,
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            shared_out = None
            return shared_out, fused_out
        shared_out, fused_out = AscendFusedMoE310.forward(
            self,
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        return shared_out, fused_out

    def _forward_shared_experts(self, hidden_states: torch.Tensor):
        if self._shared_experts is None:
            return None
        part1_out = self._shared_experts_part1(hidden_states)
        shared_out = self._shared_experts_part2(hidden_states, part1_out)
        return shared_out

    def forward_impl(  # type: ignore[override]
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ):
        routed_out = AscendFusedMoE310.forward_impl(
            self,
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        if self._shared_experts is None:
            return routed_out
        shared_out = self._forward_shared_experts(hidden_states)
        return shared_out, routed_out
