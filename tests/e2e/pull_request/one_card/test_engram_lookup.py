# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.models.deepseek_v41.engram import npu


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("offload", [False, True])
def test_engram_lookup_preserves_checkpoint_rows(monkeypatch, quantized, offload):
    # Exercise several pointer-table chunks without allocating a full table.
    monkeypatch.setattr(npu, "CHUNK_ROWS", 7)
    device = torch.device("npu:0")
    source = torch.linspace(-1.37, 2.41, 29 * 256).reshape(29, 256).bfloat16()
    weight, scales = npu.quantize_engram_rows(source) if quantized else (source, None)
    reference = npu.dequantize_engram_rows(weight, scales) if quantized else source
    ids = torch.tensor(
        [[0, 0, 1000, 1006, 1007], [0, 0, 1013, 1027, 1028], [0, 0, -1, 999, 1029]],
        dtype=torch.int64,
        device=device,
    )
    expected = reference[torch.tensor([[0, 6, 7], [13, 27, 28], [0, 0, 0]])].clone()
    expected[-1] = 0
    output = torch.full((3, 4, 256), 42, dtype=torch.bfloat16, device=device)
    kwargs = dict(
        head_start=2, local_heads=3, pad_heads=4, vocab_start=1000, vocab_end=1029, output=output.view(-1, 256)
    )
    buffers = []
    try:
        if offload:
            weight_buffer = npu.HostUvaBuffer(weight.shape, weight.dtype, device)
            buffers.append(weight_buffer)
            weight_buffer.tensor.copy_(weight)
            scale_buffer = None
            if scales is not None:
                scale_buffer = npu.HostUvaBuffer(scales.shape, scales.dtype, device)
                buffers.append(scale_buffer)
                scale_buffer.tensor.copy_(scales)
            npu.gather_dequantize_host_uva(weight_buffer, scale_buffer, ids, **kwargs)
        else:
            npu.gather_dequantize_engram_int8(
                weight.to(device), scales.to(device) if scales is not None else None, ids, 256, **kwargs
            )
        assert torch.equal(output[:, :3].cpu(), expected)
        assert torch.all(output[:, 3].cpu() == 42)
    finally:
        for buffer in buffers:
            buffer.close()
