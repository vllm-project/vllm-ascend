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

from collections.abc import Callable
from typing import Any

import torch
import vllm.model_executor.layers.fused_moe as _fused_moe_pkg
import vllm.model_executor.layers.fused_moe.layer as _fused_moe_layer
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.fused_moe.router.router_factory import create_ascend_fused_moe_router
from vllm_ascend.utils import is_310p

# Capture the real original before fused_moe.py's module-level code runs.
_original_FusedMoE = _fused_moe_layer.FusedMoE
_DefaultAscendMoERunner: Any
_DefaultAscendRoutedExperts: Any
_IS_310P = is_310p()

if _IS_310P:
    from vllm_ascend._310p.fused_moe.fused_moe import AscendMoERunner310, AscendRoutedExperts310

    _DefaultAscendMoERunner = AscendMoERunner310
    _DefaultAscendRoutedExperts = AscendRoutedExperts310
else:
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts

    _DefaultAscendMoERunner = AscendMoERunner
    _DefaultAscendRoutedExperts = AscendRoutedExperts


def _ascend_FusedMoE(
    num_experts: int,
    top_k: int,
    *args,
    renormalize: bool = True,
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    custom_routing_function: Callable | None = None,
    router: FusedMoERouter | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    enable_eplb: bool = False,
    num_redundant_experts: int = 0,
    n_shared_experts: int | None = None,
    apply_routed_scale_to_output: bool = False,
    zero_expert_type: str | None = None,
    hash_indices_table: torch.Tensor | None = None,
    runner_cls: Any | None = None,
    runner_args: dict[str, Any] | None = None,
    routed_experts_cls: Any | None = None,
    routed_experts_args: dict[str, Any] | None = None,
    hash: Any | None = None,
    tid2eid: torch.Tensor | None = None,
    **kwargs,
):
    if runner_cls is None:
        runner_cls = _DefaultAscendMoERunner
    if routed_experts_cls is None:
        routed_experts_cls = _DefaultAscendRoutedExperts
    # RoutedExperts allocates its parameters before AscendMoERunner is
    # constructed. Propagate Ascend EPLB capacity into the upstream factory so
    # redundant expert slots are present when weights are created and loaded.
    eplb_config = get_ascend_config().eplb_config
    if eplb_config.dynamic_eplb or eplb_config.expert_map_path is not None:
        configured_redundancy = eplb_config.num_redundant_experts
        upstream_redundancy = num_redundant_experts
        if configured_redundancy and upstream_redundancy not in (0, configured_redundancy):
            raise ValueError(
                f"Conflicting EPLB redundant expert counts: vLLM={upstream_redundancy}, Ascend={configured_redundancy}."
            )
        enable_eplb = True
        num_redundant_experts = configured_redundancy or upstream_redundancy
    if router is None:
        router = create_ascend_fused_moe_router(
            top_k=top_k,
            global_num_experts=num_experts + num_redundant_experts,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor if not apply_routed_scale_to_output else 1.0,
            e_score_correction_bias=e_score_correction_bias,
            zero_expert_type=zero_expert_type,
            num_logical_experts=num_experts,
            hash_indices_table=hash_indices_table,
            tid2eid=tid2eid,
        )
    routed_experts_args = dict(routed_experts_args) if routed_experts_args is not None else {}
    routed_experts_args["router"] = router
    routed_experts_args["n_shared_experts"] = n_shared_experts
    if tid2eid is not None:
        routed_experts_args["tid2eid"] = tid2eid
    return _original_FusedMoE(
        *args,
        num_experts=num_experts,
        top_k=top_k,
        renormalize=renormalize,
        use_grouped_topk=use_grouped_topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        custom_routing_function=custom_routing_function,
        router=router,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        enable_eplb=enable_eplb,
        num_redundant_experts=num_redundant_experts,
        n_shared_experts=n_shared_experts,
        apply_routed_scale_to_output=apply_routed_scale_to_output,
        zero_expert_type=zero_expert_type,
        hash_indices_table=hash_indices_table,
        runner_cls=runner_cls,
        runner_args=runner_args,
        routed_experts_cls=routed_experts_cls,
        routed_experts_args=routed_experts_args,
        **kwargs,
    )


_fused_moe_layer.FusedMoE = _ascend_FusedMoE
_fused_moe_pkg.FusedMoE = _ascend_FusedMoE
