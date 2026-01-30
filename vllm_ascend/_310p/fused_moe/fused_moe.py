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
from typing import Callable, Optional

import torch
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod, get_compressed_expert_map)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.eplb.core.eplb_utils import init_eplb_config
from vllm_ascend.ops.fused_moe.experts_selector import (select_experts,
                                                        zero_experts_compute)
from vllm_ascend.ops.fused_moe.moe_comm_method import (FusedExpertsResult,
                                                       setup_moe_comm_method)
from vllm_ascend.ops.fused_moe.prepare_finalize import QuantType
from .experts_selector import select_experts, zero_experts_compute


class AscendUnquantizedFusedMoEMethod310(UnquantizedFusedMoEMethod):

    def __init__(self, moe: FusedMoEConfig = None):
        super().__init__(moe=moe)

    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod,
              self).process_weights_after_loading(layer)

        # Fused gate_up_proj (column parallel)
        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(
            1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)
        # down_proj (row parallel)
        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(
            1, 2).contiguous()
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              use_grouped_topk: bool,
              top_k: int,
              router_logits: torch.Tensor,
              renormalize: bool,
              topk_group: Optional[int] = None,
              num_expert_group: Optional[int] = None,
              custom_routing_function: Optional[Callable] = None,
              scoring_func: str = "softmax",
              routed_scaling_factor: float = 1.0,
              e_score_correction_bias: Optional[torch.Tensor] = None,
              global_num_experts: int = -1,
              expert_map: Optional[torch.Tensor] = None,
              apply_router_weight_on_input: bool = False,
              enable_force_load_balance: bool = False,
              log2phy: torch.Tensor = None,
              **kwargs) -> torch.Tensor:
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
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=global_num_experts)

        if zero_expert_num > 0 and zero_expert_type is not None:
            topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=global_num_experts,
                zero_expert_type=zero_expert_type,
                hidden_states=x,
            )

        topk_weights = topk_weights.to(x.dtype)
        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0),
                                       global_num_experts,
                                       device=topk_ids.device)
            topk_ids = torch.argsort(
                random_matrix, dim=1)[:, :topk_ids.size(1)].to(topk_ids.dtype)

        moe_comm_method = get_forward_context().moe_comm_method
        final_hidden_states = moe_comm_method.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            dynamic_eplb=self.dynamic_eplb,
            log2phy=log2phy,
            mc2_mask=kwargs.get("mc2_mask", None))
        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states += zero_expert_result
        return final_hidden_states


class AscendFusedMoE310(FusedMoE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        num_experts = kwargs["num_experts"]
        self._expert_map = None
        self.log2phy = None

        if self.quant_config is None:
            self.quant_method = AscendUnquantizedFusedMoEMethod310(
                self.moe_config)
        else:
            self.quant_method = self.quant_config.get_quant_method(
                self, self.layer_name)

        assert self.quant_method is not None

        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        self.moe_config.ep_group = get_ep_group()
        self.moe_config.supports_eplb = self.quant_method.supports_eplb
        ascend_config = get_ascend_config()

        # init moe
        eplb_config = ascend_config.eplb_config
        self.global_expert_map, self._expert_map, self.log2phy, self.global_redundant_expert_num = init_eplb_config(
            eplb_config, self.moe_instance_id, self.moe_config)
        self.global_num_experts = num_experts + self.global_redundant_expert_num
        self.local_num_experts = torch.sum(
            self._expert_map != -1).item() if self._expert_map is not None else self.global_num_experts
        if self._expert_map is not None:
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.", self.ep_rank, self.ep_size, self.local_num_experts,
                self.global_num_experts,
                get_compressed_expert_map(self._expert_map))

        self.moe_config.num_experts = self.global_num_experts
        self.moe_config.num_local_experts = self.local_num_experts
        self.moe_config.global_redundant_expert_num = self.global_redundant_expert_num

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": self.hidden_size,
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            "params_dtype": self.params_dtype,
            "weight_loader": self.weight_loader,
        }

        self.quant_method.create_weights(layer=self, **moe_quant_params)
        self.quant_type = self.get_quant_type()

        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
        if self.enable_shared_expert_dp:
            raise RuntimeError("shared_expert_dp is not supported for 310P.")
        setup_moe_comm_method(self.moe_config)

    def get_quant_type(self) -> QuantType:
        quant_method = self.quant_method
        if not hasattr(quant_method,
                       "quant_method") or quant_method.quant_method is None:
            return QuantType.NONE

        method = quant_method.quant_method
        if hasattr(method, "quant_type"):
            from vllm_ascend.quantization.methods.base import \
                QuantType as SchemeQuantType
            scheme_quant_type = method.quant_type
            if scheme_quant_type == SchemeQuantType.W8A8:
                raise RuntimeError("W8A8 is not supported currently.")
        return QuantType.NONE

    def update_expert_map(self, new_expert_map):
        self._expert_map = new_expert_map

    def get_log2phy_map(self):
        return self.log2phy

    def clear_moe_load(self):
        if self.moe_load is not None:
            self.moe_load.zero_()

    def maybe_all_reduce_tensor_model_parallel(
            self, final_hidden_states: torch.Tensor):
        return torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(
            final_hidden_states)

    def forward_impl(  # type: ignore[override]
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor
        ) -> torch.Tensor:

        assert self.quant_method is not None
        # Load balancing for token distribution among experts in dummy_run
        # TODO: The community only considers load balancing when DP > 1.
        # This approach may overlook some extreme scenarios.
        enable_force_load_balance = forward_context.in_profile_run

        forward_context = get_forward_context()
        hidden_states, router_logits, mc2_mask, context_metadata = forward_context.moe_comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=forward_context.sp_enabled,
            enable_shared_expert_dp=self.enable_shared_expert_dp,
            quant_type=self.quant_type)

        if isinstance(hidden_states, tuple):
            hidden_states, pertoken_scale = hidden_states
        else:
            pertoken_scale = None

        # Matrix multiply.
        fused_experts_results: FusedExpertsResult = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            pertoken_scale=pertoken_scale,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self._expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            enable_force_load_balance=enable_force_load_balance,
            log2phy=self.log2phy,
            global_redundant_expert_num=self.global_redundant_expert_num,
            mc2_mask=mc2_mask)

        routed_out = forward_context.moe_comm_method.finalize(
            hidden_states=fused_experts_results.routed_out,
            reduce_results=self.reduce_results,
            context_metadata=context_metadata)

        return routed_out
