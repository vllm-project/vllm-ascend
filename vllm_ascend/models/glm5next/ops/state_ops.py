# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend recurrent-state gather/scatter for the GLM-5.3-Flash KDA layers.

The upstream helpers in ``vllm.model_executor.layers.mamba.ops`` assert
``state.is_cuda`` and launch Triton kernels that reference
``tl.extra.cuda.gdc_wait``, neither of which holds on Ascend. Both are plain
index ops, so they are expressed in torch here. Keeping the masking on device
(rather than branching on ``has_initial_state``) avoids a host sync and leaves
the sequence ACL-graph safe.
"""

import torch


def gather_initial_states(
    state: torch.Tensor,
    indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """Read the cache rows at ``indices``, zeroing sequences that start fresh."""
    idx = indices.to(torch.int64) * has_initial_state.to(torch.int64)
    out = state.index_select(0, idx)
    keep = has_initial_state.view([-1] + [1] * (state.dim() - 1)).to(out.dtype)
    return out * keep


def scatter_states(
    state: torch.Tensor,
    src: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    """Scatter ``src`` rows into ``state`` at ``indices`` (in place).

    Equivalent to ``state[indices] = src``. Cache slots are unique per sequence,
    so the write needs no atomics. ``gather_initial_states`` is the read-side
    counterpart.
    """
    state.index_copy_(0, indices.to(torch.int64), src)
