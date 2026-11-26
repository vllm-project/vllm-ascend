# SPDX-License-Identifier: Apache-2.0
"""Automatic mask writes restricted to proven graph-external host scopes.

The scope is enabled only by input preparation and FULL replay host entrypoints.
Capture/forward entrypoints do not enable it. No NPU capture-state query is used.
"""

import asyncio
import threading
from contextvars import ContextVar
from functools import wraps

_HOST_MASK_WRITES = ContextVar("hy4_graph_external_mask_writes", default=None)


def _scope_owner():
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.get_ident(), task


def mask_write_host_scope(function):
    """Allow mask fusion only for the HY4 Decode model state on this caller."""

    @wraps(function)
    def scoped(owner, *args, **kwargs):
        runner = getattr(owner, "model_runner", owner)
        state = getattr(runner, "model_state", None)
        enabled = getattr(state, "_decode_optimizations", False)
        token = _HOST_MASK_WRITES.set(_scope_owner() if enabled else None)
        try:
            return function(owner, *args, **kwargs)
        finally:
            _HOST_MASK_WRITES.reset(token)

    return scoped


def _enabled(capturing=False):
    return not capturing and _HOST_MASK_WRITES.get() is not None and _HOST_MASK_WRITES.get() == _scope_owner()


def try_write_prefix_mask(mask, actual, prefix_value, *, capturing=False):
    """Fuse two fills at the existing write point, retaining the original view."""
    if not _enabled(capturing):
        return False
    import torch

    if (
        mask.ndim != 1
        or mask.dtype != torch.bool
        or mask.device.type != "npu"
        or not isinstance(actual, int)
        or not 0 <= actual <= mask.numel()
        or mask.stride(0) <= 0
    ):
        return False
    if mask.numel():
        from vllm_ascend.ops.triton.target_prep_mask_kernels import _prefix_mask_write_kernel

        _prefix_mask_write_kernel[((mask.numel() + 255) // 256,)](
            mask,
            mask.stride(0),
            mask.numel(),
            actual,
            bool(prefix_value),
            256,
        )
    return True


def try_prepare_positions_with_padding(
    idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens, padding, actual_tokens, padded_tokens
):
    """Replace existing pos/seq kernel and the later two fills with one launch.

    The caller must skip its later prepare_padding_mask only when this returns
    True in this exact invocation. No persistent ready flag is kept.
    """
    if not _enabled():
        return False
    import torch

    tensors = (idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens, padding)
    if (
        padding.dtype != torch.bool
        or not isinstance(actual_tokens, int)
        or not isinstance(padded_tokens, int)
        or not 0 <= actual_tokens <= padded_tokens <= padding.numel()
        or query_start_loc.numel() < idx_mapping.numel() + 1
        or idx_mapping.numel() > seq_lens.numel()
        or padded_tokens > pos.numel()
        or any(t.ndim != 1 or not t.is_contiguous() or t.device != padding.device for t in tensors)
        or padding.device.type != "npu"
    ):
        return False
    from vllm_ascend.ops.triton.target_prep_mask_kernels import _prepare_pos_seq_lens_padding_kernel

    _prepare_pos_seq_lens_padding_kernel[(idx_mapping.numel() + 1,)](
        pos,
        seq_lens,
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        padding,
        seq_lens.numel(),
        padded_tokens,
        actual_tokens,
        1024,
    )
    return True
