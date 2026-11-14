# SPDX-License-Identifier: Apache-2.0
"""Main-owned MegaMoe communication storage for serialized target/draft calls."""

from collections.abc import Callable
from importlib import import_module
from inspect import signature
from types import SimpleNamespace
from typing import Any


def is_draft_moe_config(moe_config: Any, vllm_config: Any) -> bool:
    """Identify a draft communicator even when MRv2 constructs it under target config.

    MRv2 eager DSpark forwards retain the target's context flag. The loaded
    target/draft model descriptors still distinguish their logical expert
    counts; redundant target slots must not turn a target into a draft.
    """
    is_draft_runner = getattr(vllm_config.model_config, "runner_type", None) == "draft"
    speculative = getattr(vllm_config, "speculative_config", None)
    draft_config = getattr(speculative, "draft_model_config", None)
    if draft_config is None:
        return is_draft_runner
    target_experts = int(vllm_config.model_config.get_num_experts())
    draft_experts = int(draft_config.get_num_experts())
    # Equal expert layouts already share a cached MoECommMethod, including
    # its lazy initialization when it was last constructed by the draft.
    if target_experts == draft_experts:
        return False
    logical_experts = int(moe_config.num_experts) - int(getattr(moe_config, "global_redundant_expert_num", 0))
    return is_draft_runner or logical_experts == draft_experts


def get_main_owned_buffer(
    scope: Any,
    factory: Callable,
    *,
    is_draft: bool,
    group: Any,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    **kwargs,
):
    """Allocate once for main; give draft its own metadata over that storage.

    The MC2 group coordinator owns the allocation for its lifetime. Target
    and draft must execute serially. A draft-first request or insufficient
    capacity fails before a second native communication context is created.
    """
    bound = signature(factory).bind(group, num_experts, num_max_tokens_per_rank, num_topk, **kwargs)
    bound.apply_defaults()
    params = bound.arguments
    owner = getattr(scope, "_main_mega_moe_symm_buffer", None)
    if owner is None:
        if is_draft:
            raise RuntimeError("Main must initialize MegaMoe before the draft")
        owner = factory(*bound.args, **bound.kwargs)
        scope._main_mega_moe_symm_buffer = owner
        return owner

    if group is not owner.group:
        raise RuntimeError("Main and draft MegaMoe process groups differ")
    for name in (
        "hidden",
        "dispatch_quant_mode",
        "dispatch_quant_out_dtype",
        "combine_quant_mode",
        "comm_alg",
        "topk_weights_type",
    ):
        if params[name] != getattr(owner, name):
            raise RuntimeError(f"Incompatible shared MegaMoe setting: {name}")
    if num_max_tokens_per_rank > owner.num_max_tokens_per_rank:
        raise RuntimeError("Draft token capacity exceeds the main allocation")
    module = import_module(factory.__module__)
    required = module._get_mega_moe_ccl_buffer_size(
        owner.ep_world_size,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        params["hidden"],
        params["max_recv_token_num"],
        params["dispatch_quant_mode"],
        params["dispatch_quant_out_dtype"],
        params["combine_quant_mode"],
        params["comm_alg"],
        params["topk_weights_type"],
    )
    if required > owner.ccl_buffer_size:
        raise RuntimeError(f"Requested MegaMoe capacity {required} exceeds main's {owner.ccl_buffer_size} bytes")
    view = SimpleNamespace(**vars(owner))
    for name, value in params.items():
        setattr(view, name, value)
    assert view.context is owner.context
    assert view._ctx_manager is owner._ctx_manager
    return view
