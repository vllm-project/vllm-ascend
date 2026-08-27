# SPDX-License-Identifier: Apache-2.0
"""Patch FIA to skip capturing branch in PIECEWISE mode."""
from __future__ import annotations

import torch

from vllm.config import CUDAGraphMode

from .globals import logger


def _patch_attention() -> bool:
    """Patch full-attention (FIA) to skip the capturing branch in PIECEWISE mode.

    In PIECEWISE mode, full-attention ops are splitting ops — they run eagerly
    between graph pieces.  But ``_EXTRA_CTX.capturing`` is True during the
    entire capture process, so ``forward_fused_infer_attention`` enters its
    capturing branch and calls ``full_graph_fia`` → ``graph_task_group_begin``,
    which fails because the stream is not in capture status.

    Fix: temporarily set ``capturing=False`` for the duration of the full-
    attention forward pass when in PIECEWISE mode, so it uses the eager code
    path (correct for splitting ops).
    """
    try:
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
        from vllm_ascend.ascend_forward_context import (
            _EXTRA_CTX,
            get_forward_context,
        )
    except Exception as e:
        logger.warning("D-Cut: cannot import AscendAttentionBackendImpl: %s", e)
        return False

    patch_marker = "_dcut_piecewise_fia_patched"
    if getattr(AscendAttentionBackendImpl, patch_marker, False):
        return True

    _orig_ffia = AscendAttentionBackendImpl.forward_fused_infer_attention

    def _forward_fused_infer_attention(
        self, query, key, value, attn_metadata, output, kv_cache=None
    ):
        if _EXTRA_CTX.capturing or torch.compiler.is_compiling():
            ctx = get_forward_context()
            mode = getattr(ctx, "cudagraph_runtime_mode", None)
            if mode == CUDAGraphMode.PIECEWISE:
                # Full attention is a splitting op in PIECEWISE mode.
                # Temporarily disable capturing so the original method
                # uses the eager code path instead of full_graph_fia.
                orig_capturing = ctx.capturing
                ctx.capturing = False
                try:
                    return _orig_ffia(
                        self, query, key, value, attn_metadata, output, kv_cache
                    )
                finally:
                    ctx.capturing = orig_capturing
        return _orig_ffia(
            self, query, key, value, attn_metadata, output, kv_cache
        )

    AscendAttentionBackendImpl.forward_fused_infer_attention = (
        _forward_fused_infer_attention
    )
    setattr(AscendAttentionBackendImpl, patch_marker, True)
    logger.warning(
        "D-Cut: patched forward_fused_infer_attention to skip capturing "
        "branch in PIECEWISE mode (full attention is a splitting op)."
    )
    return True
