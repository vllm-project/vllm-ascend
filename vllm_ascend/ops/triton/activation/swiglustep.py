#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""Triton kernel for SwigluStepAndMul activation.

Math (identical to vllm's SwigluStepAndMul.forward_native):
    gate = x[..., :N],  up = x[..., N:]      # x is row-major [..., 2N]
    out  = silu(gate).clamp(max=limit) * up.clamp(min=-limit, max=limit)
    where silu(g) = g * sigmoid(g).

Order matters: silu BEFORE clamp on the gate half (SwigluStep), as opposed to
clamp-then-silu (SwigluOAI).

Launch pattern: 1D grid of size (num_vectorcore,), each program loops over
M / num_vectorcore rows. Keeping program count == core count minimizes host
launch overhead compared with a 2D BLOCK_M x BLOCK_N grid (which spawns
O(M * N / tile) programs on large shapes). Same pattern as swiglu_quant.py /
rope.py.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import (
    extract_slice,
    get_vectorcore_num,
    init_device_properties_triton,
)


@triton.jit
def _swiglustep_kernel(
    x_ptr,
    out_ptr,
    M,  # total rows (runtime value, not constexpr)
    TOTAL_COLS: tl.constexpr,  # 2N
    HALF_COLS: tl.constexpr,  # N
    LIMIT: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    # even split of rows across cores; tail core handles the remainder
    block_size = (M - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= M:
        return
    row_end = tl.minimum((pid + 1) * block_size, M)

    col_offsets = tl.arange(0, TOTAL_COLS)
    out_offsets = tl.arange(0, HALF_COLS)

    for row_idx in range(row_begin, row_end):
        # load one full row [2N], split gate/up via extract_slice, compute in fp32
        x_row = tl.load(x_ptr + row_idx * TOTAL_COLS + col_offsets)
        gate = extract_slice(x_row, offsets=(0,), sizes=(HALF_COLS,), strides=(1,)).to(tl.float32)
        up = extract_slice(x_row, offsets=(HALF_COLS,), sizes=(HALF_COLS,), strides=(1,)).to(tl.float32)

        # silu(gate) then clamp upper bound only
        s = gate * tl.sigmoid(gate)
        s = tl.minimum(s, LIMIT)
        # up clamp both sides
        up = tl.minimum(up, LIMIT)
        up = tl.maximum(-LIMIT, up)
        out = s * up

        tl.store(
            out_ptr + row_idx * HALF_COLS + out_offsets,
            out.to(x_ptr.dtype.element_ty),
        )


def swiglustep_forward_triton(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
    """Fused SwigluStep: silu(gate).clamp(max=limit) * up.clamp(±limit).

    Args:
        x: row-major tensor of shape [..., 2N] (dtype: float16 / bfloat16).
        limit: clamp bound (scalar). Step-3.7 uses 7.0 for expert layers.
    Returns:
        tensor of shape [..., N], same dtype as x.
    """
    assert x.shape[-1] % 2 == 0, f"swiglustep: last dim must be 2N (even), got {x.shape[-1]}"
    if not x.is_contiguous():
        x = x.contiguous()

    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    M, total_cols = x_2d.shape
    half_cols = total_cols // 2

    # NPU vector core requires UB load/store to be 32-byte aligned. The store
    # of `half_cols` (=N) elements per row is the tightest constraint, so
    # N * element_size must be a multiple of 32 (N % 16 == 0 for bf16/fp16).
    # Real MoE shapes (Step-3.7 N=1280) satisfy this; non-aligned tiny shapes
    # are rejected up front instead of crashing the AIC.
    align_elems = 32 // x.element_size()
    assert half_cols % align_elems == 0, (
        f"swiglustep: N (={half_cols}) must be a multiple of {align_elems} for 32-byte UB alignment on NPU vector core"
    )

    out_2d = torch.empty(M, half_cols, dtype=x.dtype, device=x.device)

    # idempotent: worker.py calls it once at serve time; standalone scripts need it
    init_device_properties_triton()
    num_vectorcore = get_vectorcore_num()

    _swiglustep_kernel[(num_vectorcore,)](
        x_2d,
        out_2d,
        M,
        TOTAL_COLS=total_cols,
        HALF_COLS=half_cols,
        LIMIT=limit,
        NUM_CORES=num_vectorcore,
        multibuffer=True,
    )

    return out_2d.view(*orig_shape[:-1], half_cols)
