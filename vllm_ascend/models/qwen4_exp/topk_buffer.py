# SPDX-License-Identifier: Apache-2.0
"""Construction-scoped scratch storage for sequential target QSA layers."""

from contextlib import contextmanager
from contextvars import ContextVar

# Context-local construction state, reset even if model construction fails.
# Draft models are constructed outside this scope and retain per-layer indices.
_target_topk_buffers: ContextVar[dict | None] = ContextVar("qwen4_target_topk_buffers", default=None)


@contextmanager
def share_target_topk_buffers(enabled: bool):
    token = _target_topk_buffers.set({} if enabled else None)
    try:
        yield
    finally:
        _target_topk_buffers.reset(token)


def share_topk_buffer(attention) -> None:
    buffers = _target_topk_buffers.get()
    if buffers is None:
        return
    buffer = attention.topk_indices_buffer
    # Keep distinct layouts/devices separate, including heterogeneous layers.
    key = (tuple(buffer.shape), buffer.dtype, buffer.device)
    shared = buffers.setdefault(key, buffer)
    # Assignment preserves the upstream non-persistent buffer registration.
    attention.topk_indices_buffer = shared
