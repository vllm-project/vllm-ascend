# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correct the TurboQuant scale and write compact cache rows in one launch."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


# Prefill produces arbitrary row counts. Keep them out of the compilation key
# so a new prompt length does not compile a kernel while decoding is active.
@triton.jit(do_not_specialize=["rows"])
def _finalize(packed, norm, norm_lut, output, rows):
    columns = tl.arange(0, 256)
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        codes = tl.load(packed + row * 256 + columns)
        squared_norm = tl.load(norm_lut + codes.to(tl.int32))
        selected_norm = tl.sqrt(tl.sum(squared_norm, 0))
        original_norm = tl.load(norm + row).to(tl.float32)
        scale = (original_norm / selected_norm).to(tl.float16).to(tl.uint16, bitcast=True)
        tl.store(output + row * 258 + columns, codes)
        tl.store(output + row * 258 + 256, (scale & 255).to(tl.uint8))
        tl.store(output + row * 258 + 257, (scale >> 8).to(tl.uint8))


def turboquant_finalize(packed: torch.Tensor, norm: torch.Tensor, norm_lut: torch.Tensor) -> torch.Tensor:
    rows = packed.shape[0]
    output = torch.empty((rows, 1, 258), dtype=torch.uint8, device=packed.device)
    if rows:
        _finalize[(min(rows, get_vectorcore_num()),)](packed, norm, norm_lut, output, rows)
    return output
