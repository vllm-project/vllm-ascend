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
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe.experts_selector import zero_experts_compute
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner, _clear_provisional_routed_expert_parameters
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    AllGatherCommImpl,
    FusedExpertsResult,
    _MoECommMethods,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import maybe_trans_nz

from .experts_selector import select_experts
from .moe_comm_method import AllGatherCommImpl310


class AscendUnquantizedFusedMoEMethod310(UnquantizedFusedMoEMethod):
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

        # Fused gate_up_proj (column parallel)
        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2).contiguous()
        w13_data = maybe_trans_nz(w13_data)
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)
        # down_proj (row parallel)
        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2).contiguous()
        w2_data = maybe_trans_nz(w2_data)
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
        num_experts: int = -1,
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
            global_num_experts=num_experts,
        )

        if zero_expert_num > 0 and zero_expert_type is not None:
            topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=num_experts,
                zero_expert_type=zero_expert_type,
                hidden_states=x,
            )

        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                quant_type=QuantType.NONE,
                dynamic_eplb=False,
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input,
            ),
        )
        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states += zero_expert_result
        return final_hidden_states


# Upstream vLLM PR #41184 made FusedMoE a factory and moved MoE weight
# ownership to RoutedExperts. 310P follows the same runner/factory extension point
# while injecting its own routed-experts owner for 310P-specific quantization and
# communication.
class AscendMoERunner310(AscendMoERunner):
    @property
    def is_internal_router(self) -> bool:
        # vLLM PR #41184 makes models read this property from MoERunner.
        # 310P has no internal fp32 gate path, so the model
        # must apply the gate and pass real router logits into MoE.
        return False


class AscendRoutedExperts310(RoutedExperts):
    # vLLM PR #41184 moved the MoE weight owner from FusedMoE to
    # RoutedExperts. This is the 310P MoE weight owner: it carries the same
    # 310P initialization and NPU MoE forward path on the routed-experts object.
    def __init__(
        self,
        layer_name: str,
        params_dtype: torch.dtype,
        moe_config: FusedMoEConfig,
        quant_config,
        expert_map_manager,
        expert_mapping=None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        swiglu_limit: float | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        *,
        original_num_experts: int | None = None,
        shared_experts: torch.nn.Module | None = None,
        routed_input_transform: torch.nn.Module | None = None,
        **kwargs,
    ):
        # Upstream vLLM PR #41184 requires RoutedExperts to own weight names.
        # Initialize it first, then replace the provisional upstream weights
        # with the 310P-specific Ascend weights below.
        RoutedExperts.__init__(
            self,
            layer_name=layer_name,
            params_dtype=params_dtype,
            moe_config=moe_config,
            quant_config=quant_config,
            expert_map_manager=expert_map_manager,
            expert_mapping=expert_mapping,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            swiglu_limit=swiglu_limit,
            e_score_correction_bias=e_score_correction_bias,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        self.vllm_config = get_current_vllm_config()
        self._routed_input_transform = routed_input_transform
        self._shared_experts = shared_experts
        self.global_num_experts = original_num_experts or self.moe_config.num_logical_experts
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.apply_router_weight_on_input = apply_router_weight_on_input

        if quant_config is None:
            self.quant_method = AscendUnquantizedFusedMoEMethod310(self.moe_config)
        else:
            self.quant_method = quant_config.get_quant_method(self, self.layer_name)

        assert self.quant_method is not None
        self.base_quant_method = self.quant_method

        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        self.moe_config.ep_group = get_ep_group()
        self.moe_config.supports_eplb = False

        self.global_expert_map = None
        self.local_expert_map = None
        if self.moe_config.ep_size > 1:
            raise RuntimeError("Expert Parallel is not supported on 310P. Please remove --enable-expert-parallel.")
        self.local_num_experts = self.global_num_experts

        self.moe_config.num_experts = self.global_num_experts
        self.moe_config.num_local_experts = self.local_num_experts
        self.moe_config.global_redundant_expert_num = 0
        self.expert_map_manager.global_num_experts = self.global_num_experts
        self.expert_map_manager._local_num_experts = self.local_num_experts
        self.expert_map_manager._expert_map = None

        # vLLM PR #41184's RoutedExperts.__init__ eagerly creates upstream
        # weights before the 310P quant method is installed. Reuse the shared
        # helper to drop those provisional expert weights while preserving
        # e_score_correction_bias (a routing parameter, not a recreatable weight).
        _clear_provisional_routed_expert_parameters(self)

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": self.hidden_size,
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            "params_dtype": self.params_dtype,
            "weight_loader": self.weight_loader,
            "global_num_experts": self.global_num_experts,
        }
        self.quant_method.create_weights(layer=self, **moe_quant_params)
        self.quant_type = self.get_quant_type()

        _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl310(self.moe_config)

    @property
    def is_internal_router(self) -> bool:
        # 310P Ascend path expects router logits from the model forward path.
        return False

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

        prepare_output = _EXTRA_CTX.moe_comm_method.prepare(
            hidden_states=hidden_states, router_logits=router_logits, quant_type=self.quant_type
        )
        hidden_states = prepare_output.hidden_states
        router_logits = prepare_output.router_logits
        pertoken_scale = prepare_output.pertoken_scale
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape

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
            num_experts=self.global_num_experts,
            expert_map=self.local_expert_map,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            pertoken_scale=pertoken_scale,
        )

        routed_out = _EXTRA_CTX.moe_comm_method.finalize(
            hidden_states=fused_experts_results.routed_out,
            reduce_results=isinstance(_EXTRA_CTX.moe_comm_method, AllGatherCommImpl),
            padded_hidden_states_shape=padded_hidden_states_shape,
        )

        return routed_out

    def _forward_shared_experts(self, hidden_states: torch.Tensor):
        if self._shared_experts is None:
            return None
        return self._shared_experts(hidden_states)

    def shared_forward_impl(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None = None,
    ):
        # vLLM PR #41184 makes MoERunner pass a separate shared-expert input.
        # Use it only for the shared path so routed 310P MoE still
        # sees the model-provided router logits.
        shared_hidden_states = hidden_states if shared_experts_input is None else shared_experts_input
        routed_out = self.forward_impl(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        if self._shared_experts is None:
            return routed_out
        shared_out = self._forward_shared_experts(shared_hidden_states)
        return shared_out, routed_out

    def ensure_moe_quant_config_init(self):
        return self._ensure_moe_quant_config_init()


class AscendFusedMoE310(torch.nn.Module):  # type: ignore[no-redef]
    # Upstream vLLM PR #41184 made FusedMoE a factory. 310P cannot subclass it
    # directly: __new__ builds the 310P runner/routed-experts through the
    # upstream extension point, and instances of this class are never created.
    def __new__(cls, *args, **kwargs):
        if cls is AscendFusedMoE310:
            return _create_ascend_fused_moe_runner_310(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "AscendFusedMoE310 is a factory shell for the current vLLM "
            "FusedMoE factory. Create MoE layers through the 310P factory "
            "so runner_cls/routed_experts_cls can inject the Ascend implementation."
        )


def _create_ascend_fused_moe_runner_310(*args, **kwargs):
    # Upstream vLLM PR #41184 exposes runner_cls/routed_experts_cls as the
    # runner/factory extension point. Use that point for 310P too, but inject the
    # 310P-specific routed owner so quantization and communication do not fall
    # back to the generic Ascend implementation.
    kwargs = dict(kwargs)
    routed_experts_args = dict(kwargs.pop("routed_experts_args", {}) or {})
    routed_experts_args.update(
        {
            "original_num_experts": kwargs.get("num_experts"),
            "shared_experts": kwargs.get("shared_experts"),
            "routed_input_transform": kwargs.get("routed_input_transform"),
        }
    )
    kwargs.setdefault("runner_cls", AscendMoERunner310)
    kwargs.setdefault("routed_experts_cls", AscendRoutedExperts310)
    kwargs["routed_experts_args"] = routed_experts_args
    return FusedMoE(*args, **kwargs)
