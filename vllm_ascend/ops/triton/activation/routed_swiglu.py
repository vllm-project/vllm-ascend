# SPDX-License-Identifier: Apache-2.0
"""Clipped SwiGLU for the valid prefix of a BF16 MC2 receive buffer."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

# The smallest receive buffer measured in the A3 DSpark operator benchmark.
_MIN_CAPACITY = 5760
_MAX_EXPERTS = 64
_MAX_TILE_WIDTH = 4096


def can_use_routed_swiglu(x: torch.Tensor, groups: torch.Tensor, group_list_type: int) -> bool:
    return (
        x.device.type == "npu"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[0] >= _MIN_CAPACITY
        and 0 < x.shape[1] <= 2 * _MAX_TILE_WIDTH
        and x.shape[1] % 32 == 0  # Both BF16 halves start on a 32-byte boundary.
        and x.is_contiguous()
        and group_list_type in (0, 1)
        and groups.ndim == 1
        and 0 < groups.numel() <= _MAX_EXPERTS
        and groups.dtype in (torch.int32, torch.int64)
        and groups.device == x.device
        and groups.is_contiguous()
    )


@triton.jit
def _routed_swiglu_kernel(
    x,
    groups,
    out,
    limit: tl.constexpr,
    capacity: tl.constexpr,
    width: tl.constexpr,
    experts: tl.constexpr,
    group_block: tl.constexpr,
    cumulative: tl.constexpr,
    columns: tl.constexpr,
):
    if cumulative:
        rows = tl.load(groups + experts - 1).to(tl.int32)
    else:
        expert = tl.arange(0, group_block)
        rows = tl.sum(tl.load(groups + expert, expert < experts, other=0).to(tl.int32), axis=0)
    col = tl.arange(0, columns)
    for row in range(tl.program_id(0), tl.minimum(rows, capacity), tl.num_programs(0)):
        offset = row.to(tl.int64) * width
        gate = tl.load(x + 2 * offset + col, col < width, other=0).to(tl.float32)
        up = tl.load(x + 2 * offset + width + col, col < width, other=0).to(tl.float32)
        # Match the BF16 rounding of the in-place clamps before npu_swiglu.
        gate = tl.minimum(gate, limit).to(tl.bfloat16).to(tl.float32)
        up = tl.minimum(tl.maximum(up, -limit), limit).to(tl.bfloat16).to(tl.float32)
        activated = gate / (1.0 + tl.exp(-gate))
        tl.store(out + offset + col, activated * up, col < width)


def routed_swiglu(x: torch.Tensor, groups: torch.Tensor, group_list_type: int, limit: float) -> torch.Tensor:
    capacity, packed_width = x.shape
    width = packed_width // 2
    # GMM2 consumes the same group list. The unused capacity tail has no consumers.
    out = x.new_empty((capacity, width))
    _routed_swiglu_kernel[(get_vectorcore_num(),)](
        x,
        groups,
        out,
        limit,
        capacity,
        width,
        groups.numel(),
        triton.next_power_of_2(groups.numel()),
        group_list_type == 0,
        triton.next_power_of_2(width),
    )
    return out
