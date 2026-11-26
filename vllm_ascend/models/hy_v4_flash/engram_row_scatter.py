# SPDX-License-Identifier: Apache-2.0
"""Graph-external CPU-offload row conversion, retaining the caller's H2D."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _scatter_engram_rows(
    source,
    output0,
    output1,
    output2,
    output3,
    strides: tl.constexpr,
    N: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    SOURCE_LAYER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    layer = tl.program_id(1)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    value = tl.load(source + layer * SOURCE_LAYER + i, valid, other=0).to(tl.float32)
    token = i // (HEADS * DIM)
    head = (i // DIM) % HEADS
    channel = i % DIM
    if layer == 0:
        offset = token * strides[0][0] + head * strides[0][1] + channel * strides[0][2]
        tl.store(output0 + offset, value, valid)
    elif layer == 1:
        offset = token * strides[1][0] + head * strides[1][1] + channel * strides[1][2]
        tl.store(output1 + offset, value, valid)
    elif layer == 2:
        offset = token * strides[2][0] + head * strides[2][1] + channel * strides[2][2]
        tl.store(output2 + offset, value, valid)
    else:
        offset = token * strides[3][0] + head * strides[3][1] + channel * strides[3][2]
        tl.store(output3 + offset, value, valid)


def scatter_engram_rows(source, outputs):
    """Convert stacked BF16 NPU rows into existing FP32 graph buffers once.

    Returns False for unsupported layouts so callers retain their old copies.
    Performs no allocation, address-table upload, or device-to-host conversion.
    """
    # Ascend launch ABI requires explicit output pointer arguments. The current
    # HY4 model has four Engram layers; other layouts retain the original copies.
    if len(outputs) != 4 or source.ndim != 4 or not source.is_contiguous():
        return False
    if source.dtype != torch.bfloat16 or source.device.type != "npu":
        return False
    if source.shape[0] != len(outputs):
        return False
    shape = tuple(source.shape[1:])
    if any(tuple(out.shape) != shape or out.dtype != torch.float32 or out.device != source.device for out in outputs):
        return False
    # Outputs are distinct, persistent layer buffers owned by model_state.
    size = source[0].numel()
    if size:
        _scatter_engram_rows[(triton.cdiv(size, 256), len(outputs))](
            source,
            outputs[0],
            outputs[1],
            outputs[2],
            outputs[3],
            tuple(tuple(out.stride()) for out in outputs),
            size,
            shape[1],
            shape[2],
            source.stride(0),
            256,
        )
    return True
