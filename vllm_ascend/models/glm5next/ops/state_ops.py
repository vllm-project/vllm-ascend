# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend recurrent-state gather/scatter for the GLM-5.3-Flash KDA layers."""

import math

import torch
import torch_npu

from vllm_ascend.ops.triton.mamba.state_ops import (
    gather_initial_states as gather_initial_states,
)


def scatter_states(state: torch.Tensor, src: torch.Tensor, indices: torch.Tensor) -> None:
    """Write unique selected state rows without a contiguous copy of the pool."""
    if state.device.type != "npu":
        state.index_copy_(0, indices.to(torch.int64), src)
        return
    if indices.numel() == 0 or math.prod(state.shape[1:]) == 0:
        return
    assert state.ndim >= 2 and src.ndim == state.ndim
    assert indices.ndim == 1 and indices.device == state.device
    assert src.shape == (indices.numel(), *state.shape[1:])
    assert indices.dtype in (torch.int32, torch.int64)
    assert state[0].is_contiguous() and src[0].is_contiguous()
    # Treat each contiguous state as one cache row. These views preserve the
    # page stride and storage offset without copying the padded state pool.
    row_size = math.prod(state.shape[1:])
    torch_npu.npu_scatter_nd_update_(
        state.view(state.shape[0], row_size),
        indices.unsqueeze(-1),
        src.view(indices.numel(), row_size),
    )
