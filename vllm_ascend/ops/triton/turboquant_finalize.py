# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correct the TurboQuant scale and write compact cache rows in one launch."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num
from vllm_ascend.quantization.methods.kv_cache.turboquant import HEAD_DIM, SLOT_BYTES

# TurboQuant uses one byte per pair of 4-bit codes and stores one FP16 scale
# after the code bytes. Reuse the cache layout constants for allocation and
# pass the derived sizes as compile-time kernel arguments.
CODE_BYTES = HEAD_DIM // 2


# Prefill produces arbitrary row counts. Keep them out of the compilation key
# so a new prompt length does not compile a kernel while decoding is active.
@triton.jit(do_not_specialize=["rows"])
def _finalize(packed, norm, norm_lut, output, rows, CODE_BYTES: tl.constexpr, ROW_BYTES: tl.constexpr):
    SCALE_LOW_OFFSET: tl.constexpr = CODE_BYTES
    SCALE_HIGH_OFFSET: tl.constexpr = SCALE_LOW_OFFSET + 1
    columns = tl.arange(0, CODE_BYTES)
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        codes = tl.load(packed + row * CODE_BYTES + columns)
        squared_norm = tl.load(norm_lut + codes.to(tl.int32))
        selected_norm = tl.sqrt(tl.sum(squared_norm, 0))
        original_norm = tl.load(norm + row).to(tl.float32)
        scale = (original_norm / selected_norm).to(tl.float16).to(tl.uint16, bitcast=True)
        tl.store(output + row * ROW_BYTES + columns, codes)
        tl.store(output + row * ROW_BYTES + SCALE_LOW_OFFSET, (scale & 255).to(tl.uint8))
        tl.store(output + row * ROW_BYTES + SCALE_HIGH_OFFSET, (scale >> 8).to(tl.uint8))


def turboquant_finalize(packed: torch.Tensor, norm: torch.Tensor, norm_lut: torch.Tensor) -> torch.Tensor:
    rows = packed.shape[0]
    output = torch.empty((rows, 1, SLOT_BYTES), dtype=torch.uint8, device=packed.device)
    if rows:
        _finalize[(min(rows, get_vectorcore_num()),)](packed, norm, norm_lut, output, rows, CODE_BYTES, SLOT_BYTES)
    return output
