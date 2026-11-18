# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check chunk-relative Engram lookup with NPU-resident pointer tables."""

import pytest
import torch

from vllm_ascend.ops.triton.engram_lookup import _engram_host_uva_gather_dequant_kernel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.mark.parametrize("row_offset", [0, 3])
@pytest.mark.parametrize("quantized", [False, True])
def test_engram_chunk_offset(row_offset, quantized):
    # Four-row chunks make an unaligned slice cross several chunk boundaries.
    # Width and quantization group match DeepSeek V4.1 Engram.
    chunk, width, group, vocab_start, owned_rows = 4, 256, 32, 10, 7
    num_chunks = (row_offset + owned_rows + chunk - 1) // chunk
    dtype = torch.int8 if quantized else torch.bfloat16
    chunks = [torch.full((chunk, width), i + 1, dtype=dtype, device="npu") for i in range(num_chunks)]
    ptrs = torch.tensor([x.data_ptr() for x in chunks], dtype=torch.int64, device="npu")
    scales = [torch.ones((chunk, width // group), dtype=torch.float32, device="npu") for _ in chunks]
    scale_ptrs = torch.tensor([x.data_ptr() for x in scales], dtype=torch.int64, device="npu")
    ids = torch.tensor([[9, 10, 11, 13, 14, 16, 17]], dtype=torch.int64, device="npu")
    out = torch.empty((ids.numel(), width), dtype=torch.bfloat16, device="npu")
    init_device_properties_triton()
    _engram_host_uva_gather_dequant_kernel[(ids.numel(),)](
        ptrs,
        scale_ptrs if quantized else None,
        ids,
        out,
        ids.numel(),
        vocab_start,
        vocab_start + owned_rows,
        ids.stride(0),
        CHUNK=chunk,
        ROW_OFFSET=row_offset,
        WIDTH=width,
        GROUP=group,
        HEAD_START=0,
        LOCAL_HEADS=ids.numel(),
        PAD_HEADS=ids.numel(),
        QUANTIZED=quantized,
        MXFP8=False,
        num_warps=4,
    )
    expected = torch.tensor(
        [0, *[(index - vocab_start + row_offset) // chunk + 1 for index in (10, 11, 13, 14, 16)], 0],
        dtype=torch.bfloat16,
    )[:, None].expand(-1, width)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
