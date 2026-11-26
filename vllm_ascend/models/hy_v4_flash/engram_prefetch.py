# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton


class PrefetchedEngramRows:
    """Wait only for the table read by the consuming layer."""

    def __init__(self, outputs):
        self.outputs = outputs

    def __getitem__(self, index):
        rows, ready = self.outputs[index]
        stream = torch.npu.current_stream(device=rows.device)
        stream.wait_event(ready)
        rows.record_stream(stream)
        return rows


@triton.jit
def _flash_engram_mapped_rows_kernel(
    hashes, weight_ptrs, scale_ptrs, output, vocab_start, vocab_end,
    HASH_STRIDE: tl.constexpr, HEAD_START: tl.constexpr, LOCAL_HEADS: tl.constexpr,
    WIDTH: tl.constexpr, BLOCK: tl.constexpr, CHUNK: tl.constexpr, FP8: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // LOCAL_HEADS
    head = HEAD_START + row % LOCAL_HEADS
    index = tl.load(hashes + token * HASH_STRIDE + head).to(tl.int64)
    local_row = tl.minimum(tl.maximum(index - vocab_start, 0), vocab_end - vocab_start - 1)
    chunk = local_row // CHUNK
    offset = local_row % CHUNK
    cols = tl.arange(0, BLOCK)
    addr = tl.load(weight_ptrs + chunk)
    if FP8:
        weight = addr.to(tl.pointer_type(tl.uint8))
        bits = tl.load(weight + offset * WIDTH + cols, cols < WIDTH, other=0)
        value = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        value = tl.where((bits & 127) == 127, float("nan"), value)
        scales = tl.load(scale_ptrs + chunk).to(tl.pointer_type(tl.bfloat16))
        value = value * tl.load(scales + offset).to(tl.float32)
    else:
        weight = addr.to(tl.pointer_type(tl.bfloat16))
        value = tl.load(weight + offset * WIDTH + cols, cols < WIDTH, other=0).to(tl.float32)
    # A negative hash is a padded row. Write the mask with the gathered values.
    tl.store(output + row * WIDTH + cols, tl.where(index >= 0, value, 0.0), cols < WIDTH)


def gather_mapped_rows(hashes, weight, scale, head_start, head_end, vocab_start, vocab_end):
    from vllm_ascend.models.deepseek_v41.engram.npu import CHUNK_ROWS
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    tokens, local_heads = hashes.size(0), head_end - head_start
    width = weight.tensor.size(1)
    output = torch.empty((tokens, local_heads, width), dtype=torch.float32, device=hashes.device)
    if output.numel():
        init_device_properties_triton()
        _flash_engram_mapped_rows_kernel[(tokens * local_heads,)](
            hashes, weight.ptrs, weight.ptrs if scale is None else scale.ptrs, output,
            vocab_start, vocab_end, hashes.stride(0), head_start, local_heads,
            width, triton.next_power_of_2(width), CHUNK_ROWS, scale is not None,
        )
    return output
