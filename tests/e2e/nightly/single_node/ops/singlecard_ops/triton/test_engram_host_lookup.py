# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check chunk-relative Engram lookup with NPU-resident pointer tables."""

import pytest
import torch

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.triton.engram_lookup import _engram_host_uva_gather_dequant_kernel
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

pytestmark = [
    pytest.mark.skipif(
        is_950(),
        reason="Engram shared-pool host lookup has not been validated on Ascend 950 (A5)",
    ),
]


@pytest.mark.parametrize("row_offset", [0, 3])
@pytest.mark.parametrize("quantized", [False, True])
def test_engram_chunk_offset(row_offset, quantized):
    # Four-row chunks make an unaligned slice cross several chunk boundaries.
    # Width and quantization group match DeepSeek V4.1 Engram.
    chunk, width, group, vocab_start, owned_rows = 4, 256, 32, 10, 7
    num_chunks = (row_offset + owned_rows + chunk - 1) // chunk
    dtype = torch.int8 if quantized else torch.bfloat16
    chunks = [
        (torch.arange(chunk * width, dtype=torch.int32).reshape(chunk, width) + i * chunk * width)
        .remainder(255)
        .sub(127)
        .to(dtype)
        .npu()
        for i in range(num_chunks)
    ]
    ptrs = torch.tensor([x.data_ptr() for x in chunks], dtype=torch.int64, device="npu")
    scales = [
        (
            torch.arange(chunk * (width // group), dtype=torch.float32).reshape(chunk, -1).remainder(7).add(1).mul(0.5)
            + i
        ).npu()
        for i in range(num_chunks)
    ]
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
    rows = [index - vocab_start + row_offset for index in (10, 11, 13, 14, 16)]
    expected = []
    for local in rows:
        chunk_index, row_index = divmod(local, chunk)
        value = chunks[chunk_index][row_index].float()
        if quantized:
            scale = scales[chunk_index][row_index]
            value = value.unflatten(-1, (width // group, group)).mul(scale[:, None]).flatten()
        expected.append(value.bfloat16().cpu())
    zero = torch.zeros(width, dtype=torch.bfloat16)
    expected = torch.stack([zero, *expected, zero])
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
