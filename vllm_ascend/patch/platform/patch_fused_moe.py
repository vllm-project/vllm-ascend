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

# Patch vllm's FusedMoE factory to use AscendMoERunner by default.
#
# vllm's FusedMoE is a factory function (not a class). deepseek_v2 and other
# models do `from vllm.model_executor.layers.fused_moe import FusedMoE` and
# call it directly, so we must patch the binding in the package __init__ as
# well as the layer module before any model is imported.
#
# Import order in worker.__init__:
#   1. adapt_patch()  ->  this file runs  ->  FusedMoE patched
#   2. from vllm_ascend import ops
#   3. model loading  ->  deepseek_v2 imported  ->  gets patched FusedMoE  ✓

from inspect import signature

import torch
import vllm.model_executor.layers.fused_moe as _fused_moe_pkg
import vllm.model_executor.layers.fused_moe.layer as _fused_moe_layer
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.fused_moe.experts_selector import (
    select_experts as _ascend_select_experts,
)
from vllm_ascend.utils import is_310p

# Capture the real original before fused_moe.py's module-level code runs.
_original_FusedMoE = _fused_moe_layer.FusedMoE
_PATCH_MARKER = "_vllm_ascend_router_patch"
_ROUTER_COMPUTE_PARAMETERS = (
    "self",
    "hidden_states",
    "router_logits",
    "indices_type",
    "input_ids",
)

_DefaultAscendRoutedExperts: type[RoutedExperts] | None
if is_310p():
    from vllm_ascend._310p.fused_moe.fused_moe import AscendMoERunner310 as _DefaultAscendMoERunner

    _DefaultAscendRoutedExperts = None
else:
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner as _DefaultAscendMoERunner
    from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts

    _DefaultAscendRoutedExperts = AscendRoutedExperts


def _ascend_FusedMoE(*args, runner_cls=None, runner_args=None, routed_experts_cls=None, **kwargs):
    if runner_cls is None:
        runner_cls = _DefaultAscendMoERunner
    # RoutedExperts allocates its parameters before AscendMoERunner is
    # constructed. Propagate Ascend EPLB capacity into the upstream factory so
    # redundant expert slots are present when weights are created and loaded.
    eplb_config = get_ascend_config().eplb_config
    if eplb_config.dynamic_eplb or eplb_config.expert_map_path is not None:
        configured_redundancy = eplb_config.num_redundant_experts
        upstream_redundancy = kwargs.get("num_redundant_experts", 0)
        if configured_redundancy and upstream_redundancy not in (0, configured_redundancy):
            raise ValueError(
                f"Conflicting EPLB redundant expert counts: vLLM={upstream_redundancy}, Ascend={configured_redundancy}."
            )
        kwargs["enable_eplb"] = True
        kwargs["num_redundant_experts"] = configured_redundancy or upstream_redundancy
    if routed_experts_cls is None:
        routed_experts_cls = _DefaultAscendRoutedExperts
    # 'hash' is a DeepSeek V4 flag already consumed before FusedMoE is called;
    # 'tid2eid' is Ascend-specific and must reach AscendMoERunner via runner_args.
    kwargs.pop("hash", None)
    tid2eid = kwargs.pop("tid2eid", None)
    if tid2eid is not None:
        runner_args = dict(runner_args) if runner_args is not None else {}
        runner_args["tid2eid"] = tid2eid
    return _original_FusedMoE(
        *args,
        runner_cls=runner_cls,
        runner_args=runner_args,
        routed_experts_cls=routed_experts_cls,
        **kwargs,
    )


def _fused_topk_compute_routing(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    indices_type: torch.dtype | None,
    *,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _ascend_select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=self.top_k,
        use_grouped_topk=False,
        renormalize=self.renormalize,
        scoring_func=self.scoring_func,
        indices_type=indices_type,
    )


def _grouped_topk_compute_routing(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    indices_type: torch.dtype | None,
    *,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_experts = router_logits.shape[-1]
    use_grouped_topk = num_experts > self.num_expert_group and num_experts % self.num_expert_group == 0
    topk_weights, topk_ids = _ascend_select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=self.top_k,
        use_grouped_topk=use_grouped_topk,
        renormalize=self.renormalize,
        topk_group=self.topk_group if use_grouped_topk else None,
        num_expert_group=self.num_expert_group if use_grouped_topk else None,
        scoring_func=self.scoring_func,
        routed_scaling_factor=1.0,
        e_score_correction_bias=self.e_score_correction_bias,
        indices_type=indices_type,
    )
    if self.routed_scaling_factor != 1.0:
        topk_weights = topk_weights * self.routed_scaling_factor
    return topk_weights, topk_ids


def _fused_topk_bias_compute_routing(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    indices_type: torch.dtype | None,
    *,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_weights, topk_ids = _ascend_select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=self.top_k,
        use_grouped_topk=False,
        renormalize=self.renormalize,
        scoring_func=self.scoring_func,
        routed_scaling_factor=1.0,
        e_score_correction_bias=self.e_score_correction_bias,
        indices_type=indices_type,
        input_ids=input_ids,
        tid2eid=self._hash_indices_table,
    )
    if self.routed_scaling_factor != 1.0:
        topk_weights = topk_weights * self.routed_scaling_factor

    if self.num_fused_shared_experts > 0:
        num_tokens = topk_ids.shape[0]
        num_shared_experts = self.num_fused_shared_experts
        shared_ids = torch.arange(
            self.global_num_experts,
            self.global_num_experts + num_shared_experts,
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        ).expand(num_tokens, num_shared_experts)
        shared_weights = torch.full(
            (num_tokens, num_shared_experts),
            self.shared_expert_weight,
            dtype=topk_weights.dtype,
            device=topk_weights.device,
        )
        topk_ids = torch.cat([topk_ids, shared_ids], dim=-1)
        topk_weights = torch.cat([topk_weights, shared_weights], dim=-1)

    return topk_weights, topk_ids


def _patch_router_compute() -> None:
    router_patches = (
        (FusedTopKRouter, _fused_topk_compute_routing),
        (GroupedTopKRouter, _grouped_topk_compute_routing),
        (FusedTopKBiasRouter, _fused_topk_bias_compute_routing),
    )
    for router_cls, ascend_compute_routing in router_patches:
        original_compute_routing = router_cls._compute_routing
        if getattr(original_compute_routing, _PATCH_MARKER, False):
            continue
        if tuple(signature(original_compute_routing).parameters) != _ROUTER_COMPUTE_PARAMETERS:
            raise RuntimeError(
                f"Unsupported vLLM MoE router contract: {router_cls.__name__}._compute_routing signature changed."
            )
        setattr(ascend_compute_routing, _PATCH_MARKER, True)
        router_cls._compute_routing = ascend_compute_routing


_fused_moe_layer.FusedMoE = _ascend_FusedMoE
_fused_moe_pkg.FusedMoE = _ascend_FusedMoE
_patch_router_compute()
