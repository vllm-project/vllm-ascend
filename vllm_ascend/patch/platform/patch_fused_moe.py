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

# Patch vllm's MoE factory to use AscendMoERunner by default.
#
# vLLM v0.26.0 exports the factory as FusedMoE, while vLLM main renamed it
# to FusedMoEFactory. We patch the exact binding for the active version lane
# in both the package __init__ and layer module before any model is imported.
#
# Import order in worker.__init__:
#   1. adapt_patch()  ->  this file runs  ->  MoE factory patched
#   2. from vllm_ascend import ops
#   3. model loading  ->  deepseek_v2 imported  ->  gets patched factory  ✓

from collections.abc import Callable
from inspect import signature
from types import MethodType
from typing import Any

import torch
import vllm.model_executor.layers.fused_moe as _fused_moe_pkg
import vllm.model_executor.layers.fused_moe.layer as _fused_moe_layer
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.eplb.state import AscendEplbLayerState
from vllm_ascend.ops.fused_moe.router.router_factory import create_ascend_fused_moe_router

_EPLB_ROUTER_ADAPTED = "_vllm_ascend_eplb_router_adapted"

# Capture the real original before fused_moe.py's module-level code runs.
_original_FusedMoE = _fused_moe_layer.FusedMoEFactory


def _ascend_apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
    eplb_state = self.eplb_state
    if eplb_state is None:
        return topk_ids
    self._validate_eplb_state()
    expert_replica_routing_table = eplb_state.expert_replica_routing_table
    if expert_replica_routing_table is None:
        raise RuntimeError("Ascend EPLB expert replica routing table is not initialized.")
    assert eplb_state.expert_load_view is not None
    assert eplb_state.should_record_tensor is not None
    assert eplb_state.num_unpadded_tokens_tensors is not None
    return torch.ops.vllm.ascend_eplb_map_to_physical_and_record(
        topk_ids,
        expert_replica_routing_table,
        eplb_state.expert_load_view,
        eplb_state.should_record_tensor,
        eplb_state.num_unpadded_tokens_tensors[dbo_current_ubatch_id()],
    )


def _adapt_eplb_router(router, enable_eplb: bool) -> None:
    """Attach only the Ascend EPLB state and mapping operation."""
    if not enable_eplb or getattr(router, _EPLB_ROUTER_ADAPTED, False):
        return

    if router.eplb_state is None:
        router.eplb_state = AscendEplbLayerState()
    elif not isinstance(router.eplb_state, AscendEplbLayerState):
        router.eplb_state = AscendEplbLayerState.from_upstream(router.eplb_state)

    original_apply_mapping = type(router)._apply_eplb_mapping
    if tuple(signature(original_apply_mapping).parameters) != ("self", "topk_ids"):
        raise RuntimeError(
            f"Unsupported vLLM MoE router contract: {type(router).__name__}._apply_eplb_mapping signature changed."
        )
    router._apply_eplb_mapping = MethodType(_ascend_apply_eplb_mapping, router)
    setattr(router, _EPLB_ROUTER_ADAPTED, True)


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
    hash_indices_table: torch.Tensor | None = None,
    runner_cls: Any | None = None,
    runner_args: dict[str, Any] | None = None,
    routed_experts_cls: Any | None = None,
    routed_experts_args: dict[str, Any] | None = None,
    hash: Any | None = None,
    tid2eid: torch.Tensor | None = None,
    **kwargs,
):
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
    # Keep the upstream hash table as the source of truth and share it with
    # the legacy Ascend quant-method path until that path also routes solely
    # through the Router.
    hash_indices_table_for_legacy_path = hash_indices_table if hash_indices_table is not None else tid2eid
    enable_router_eplb = enable_eplb and get_current_vllm_config().use_v2_model_runner
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
            num_logical_experts=num_experts,
            hash_indices_table=hash_indices_table,
            tid2eid=hash_indices_table_for_legacy_path,
            eplb_state=AscendEplbLayerState() if enable_router_eplb else None,
        )
    routed_experts_args = dict(routed_experts_args) if routed_experts_args is not None else {}
    routed_experts_args["n_shared_experts"] = n_shared_experts
    if hash_indices_table_for_legacy_path is not None:
        routed_experts_args["tid2eid"] = hash_indices_table_for_legacy_path
    runner = _original_FusedMoE(
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
        hash_indices_table=hash_indices_table,
        runner_cls=runner_cls,
        runner_args=runner_args,
        routed_experts_cls=routed_experts_cls,
        routed_experts_args=routed_experts_args,
        **kwargs,
    )
    _adapt_eplb_router(runner.router, enable_router_eplb)
    return runner


_fused_moe_layer.FusedMoEFactory = _ascend_FusedMoE
_fused_moe_pkg.FusedMoEFactory = _ascend_FusedMoE
