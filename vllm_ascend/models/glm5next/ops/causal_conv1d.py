# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend causal conv1d entry points for the GLM-5.3-Flash KDA layers.

The upstream CUDA kernels reference ``tl.extra.cuda.gdc_wait``, which Ascend
Triton does not provide -- the AST visitor raises even when ``launch_pdl`` is
False. Both entry points are therefore routed to Ascend implementations, and the
kwargs the upstream signatures grew for CUDA-side cache management are dropped
here rather than at every call site.

``causal_conv1d_update`` prefers the NPU Triton kernel, which has no host sync
and accepts the spec-decode arguments directly. The PyTorch fallback calls
``.item()`` per request, so ACL graph capture stalls at decode-FULL when the
kernel is unavailable; ``has_npu_triton_conv1d_update()`` lets the caller report
that up front instead of hanging during capture.
"""

import torch

from vllm_ascend.ops.causal_conv1d import (
    causal_conv1d_fn as _torch_causal_conv1d_fn,
)
from vllm_ascend.ops.causal_conv1d import (
    causal_conv1d_update as _torch_causal_conv1d_update,
)

try:
    from vllm_ascend.ops.triton.mamba.causal_conv1d import (  # type: ignore[attr-defined]
        causal_conv1d_update_npu as _npu_triton_conv1d_update,
    )

    _HAS_NPU_TRITON_CONV1D_UPDATE = True
except ImportError:
    _npu_triton_conv1d_update = None
    _HAS_NPU_TRITON_CONV1D_UPDATE = False

# Cache-management and validation kwargs the upstream CUDA signatures accept but
# the Ascend implementations neither need nor understand.
_UNSUPPORTED_KWARGS = (
    "null_block_id",
    "block_idx_first_scheduled_token",
    "block_idx_last_scheduled_token",
    "initial_state_idx",
    "num_computed_tokens",
    "block_size_to_align",
    "validate_data",
    "metadata",
)

_UPDATE_FALLBACK_UNSUPPORTED_KWARGS = (*_UNSUPPORTED_KWARGS, "max_query_len", "out")


def has_npu_triton_conv1d_update() -> bool:
    """Whether the host-sync-free NPU Triton update kernel is available."""
    return _HAS_NPU_TRITON_CONV1D_UPDATE


def causal_conv1d_fn(*args, **kwargs) -> torch.Tensor:
    for key in _UNSUPPORTED_KWARGS:
        kwargs.pop(key, None)
    return _torch_causal_conv1d_fn(*args, **kwargs)


def causal_conv1d_update(*args, **kwargs) -> torch.Tensor:
    if _npu_triton_conv1d_update is not None:
        for key in _UNSUPPORTED_KWARGS:
            kwargs.pop(key, None)
        return _npu_triton_conv1d_update(*args, **kwargs)

    for key in _UPDATE_FALLBACK_UNSUPPORTED_KWARGS:
        kwargs.pop(key, None)
    return _torch_causal_conv1d_update(*args, **kwargs)
