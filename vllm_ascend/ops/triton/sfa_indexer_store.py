# SPDX-License-Identifier: Apache-2.0
"""Store native-quantized indexer keys and scales in one kernel.

Keep native quantization and its INT8 tie behavior. Only final scale conversion
and the two scatters are fused. Valid cache slots must be unique, as provided
by decode slot mapping; padded negative slots are skipped.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["rows", "capacity", "key_block_stride", "scale_block_stride"])
def _store_indexer_key_scale(
    quant_ptr,
    scales_ptr,
    slots_ptr,
    cache_ptr,
    scale_cache_ptr,
    rows,
    capacity,
    key_block_stride,
    scale_block_stride,
    KEY_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    column = tl.arange(0, KEY_WIDTH)
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        key = tl.load(quant_ptr + row * KEY_WIDTH + column)
        scale = tl.load(scales_ptr + row)
        slot = tl.load(slots_ptr + row).to(tl.int64)
        valid = (slot >= 0) & (slot < capacity)
        page = slot // BLOCK_SIZE
        offset = slot % BLOCK_SIZE
        tl.store(cache_ptr + page * key_block_stride + offset * KEY_WIDTH + column, key, valid)
        tl.store(scale_cache_ptr + page * scale_block_stride + offset, scale, valid)


def store_indexer_key_scale(
    quant: torch.Tensor,
    scales: torch.Tensor,
    slots: torch.Tensor,
    cache: torch.Tensor,
    scale_cache: torch.Tensor,
    *,
    block_size: int = 1,
    key_block_stride: int | None = None,
    scale_block_stride: int | None = None,
) -> None:
    """Publish at the existing cache-write point, after the connector wait."""
    rows = slots.numel()
    if rows == 0:
        return
    if cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        quant = quant.view(torch.uint8)
        cache = cache.view(torch.uint8)
    quant = quant.contiguous()
    scales = scales.reshape(-1).contiguous()
    slots = slots.reshape(-1).contiguous()
    key_width = quant.shape[-1]
    key_block_stride = block_size * key_width if key_block_stride is None else key_block_stride
    scale_block_stride = block_size if scale_block_stride is None else scale_block_stride
    num_programs = min(get_vectorcore_num(), rows) if cache.device.type == "npu" else rows
    _store_indexer_key_scale[(num_programs,)](
        quant,
        scales,
        slots,
        cache,
        scale_cache,
        rows,
        cache.numel() // key_width,
        key_block_stride,
        scale_block_stride,
        key_width,
        block_size,
    )
