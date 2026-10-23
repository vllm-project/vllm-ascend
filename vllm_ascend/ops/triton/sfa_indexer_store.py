# SPDX-License-Identifier: Apache-2.0
"""Store native-quantized indexer keys and FP16 scales in one kernel.

Keep native quantization and its INT8 tie behavior. Only final scale conversion
and the two scatters are fused. Valid cache slots must be unique, as provided
by decode slot mapping; padded negative slots are skipped.
"""

import torch
from vllm.triton_utils import tl, triton

_KEY_WIDTH = 128
_MAX_DECODE_ROWS = 12


@triton.jit(do_not_specialize=["capacity"])
def _store_indexer_key_scale(
    quant,
    scales,
    slots,
    cache,
    scale_cache,
    capacity,
):
    row = tl.program_id(0)
    column = tl.arange(0, 128)
    key = tl.load(quant + row * 128 + column)
    scale = tl.load(scales + row)
    slot = tl.load(slots + row).to(tl.int64)
    valid = (slot >= 0) & (slot < capacity)
    tl.store(cache + slot * 128 + column, key, valid)
    tl.store(scale_cache + slot, scale, valid)


def can_fuse_store(cache: torch.Tensor, scale_cache: torch.Tensor, slots: torch.Tensor, rows: int) -> bool:
    return (
        cache.dtype == torch.int8
        and scale_cache.dtype == torch.float16
        and slots.dtype in (torch.int32, torch.int64)
        and cache.ndim > 0
        and cache.shape[-1] == _KEY_WIDTH
        and cache.numel() // _KEY_WIDTH == scale_cache.numel()
        and cache.numel() > 0
        and cache.is_contiguous()
        and scale_cache.is_contiguous()
        and slots.is_contiguous()
        and slots.ndim == 1
        and slots.numel() == rows
        and 0 < rows <= _MAX_DECODE_ROWS
    )


def store_indexer_key_scale(
    quant: torch.Tensor,
    scales: torch.Tensor,
    slots: torch.Tensor,
    cache: torch.Tensor,
    scale_cache: torch.Tensor,
) -> None:
    """Publish at the existing cache-write point, after the connector wait."""
    rows = slots.numel()
    _store_indexer_key_scale[(rows,)](quant, scales, slots, cache, scale_cache, cache.numel() // _KEY_WIDTH)
