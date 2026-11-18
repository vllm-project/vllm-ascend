# SPDX-License-Identifier: Apache-2.0
"""Kimi decode MoE fusion scopes and optimization of the owning ACL graph."""

from contextlib import contextmanager

import torch
from vllm.forward_context import get_forward_context
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import _is_kimi_moe_model
from vllm_ascend.compilation.updatable_graph import get_capture_resource
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

_CAPTURE_SCOPES_KEY = "kimi_moe_superkernel_scopes"


def use_kimi_moe_superkernel(vllm_config, ascend_config) -> bool:
    """Reuse 0.30's switch; unsupported paths retain its generic graph policy.

    Include the target-derived draft config here so callers can disable both
    named scopes and the whole-model fallback for that draft.
    """
    return (
        ascend_config.ascend_compilation_config.enable_super_kernel
        and ascend_config.multistream_overlap_shared_expert
        and _is_kimi_moe_model(vllm_config)
        and get_current_hardware_profile().supports(HardwareCapability.CANN_MEGAMOE_MXFP)
    )


@contextmanager
def collect_moe_superkernel_scopes(forward_context, enabled: bool):
    """Track scopes per capture, including nested/piecewise capture isolation."""
    scopes: set[str] = set()
    if not enabled:
        yield scopes
        return
    kwargs = forward_context.additional_kwargs
    previous = kwargs.get(_CAPTURE_SCOPES_KEY)
    kwargs[_CAPTURE_SCOPES_KEY] = scopes
    try:
        yield scopes
    finally:
        if previous is None:
            kwargs.pop(_CAPTURE_SCOPES_KEY, None)
        else:
            kwargs[_CAPTURE_SCOPES_KEY] = previous


@contextmanager
def moe_superkernel_scope(name: str | None, enabled: bool):
    """None excludes shared communication from the enclosing named scope."""
    if not enabled:
        yield
        return
    torch.npu.super_kernel_scope_begin(name)
    try:
        if name is not None:
            scopes = get_forward_context().additional_kwargs.get(_CAPTURE_SCOPES_KEY)
            if scopes is None:
                # V2 creates the forward context inside the graph capture.
                # Its owning UpdatableGraph outlives that context; keep the
                # names with that graph, using its existing capture resources.
                scopes = get_capture_resource(_CAPTURE_SCOPES_KEY, set)
            scopes.add(name)
        yield
    finally:
        torch.npu.super_kernel_scope_end(name)


def moe_superkernel_options() -> dict:
    return {"dcci_disable_on_kernel": [".*"], "auto_op_parallel": 1}


def optimize_moe_superkernel(graph, scopes: set[str] | None = None) -> None:
    # Without named scopes CANN may fuse the whole graph. In particular, skip
    # draft, prefill and fallback captures that did not enter the decode ladder.
    if scopes is None:
        scopes = graph.get_capture_resource(_CAPTURE_SCOPES_KEY, set)
    if not scopes:
        return
    graph.super_kernel_optimize(optimize_options=moe_superkernel_options())
    logger.info(
        "Kimi MoE SuperKernel optimization requested for ACL graph with %d decode scopes; "
        "verify fusion in sk_meta/profiling.",
        len(scopes),
    )
