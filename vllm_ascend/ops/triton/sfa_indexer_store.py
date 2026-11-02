# SPDX-License-Identifier: Apache-2.0
"""Store native-quantized indexer keys and FP16 scales in one kernel.

Keep native quantization and its INT8 tie behavior. Only final scale conversion
and the two scatters are fused. Valid cache slots must be unique, as provided
by decode slot mapping; padded negative slots are skipped.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["capacity"])
def _store_indexer_key_scale(
    quant_ptr,
    scales_ptr,
    slots_ptr,
    cache_ptr,
    scale_cache_ptr,
    capacity,
    KEY_WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, KEY_WIDTH)
    key = tl.load(quant_ptr + row * KEY_WIDTH + column)
    scale = tl.load(scales_ptr + row)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    valid = (slot >= 0) & (slot < capacity)
    tl.store(cache_ptr + slot * KEY_WIDTH + column, key, valid)
    tl.store(scale_cache_ptr + slot, scale, valid)


def store_indexer_key_scale(
    quant: torch.Tensor,
    scales: torch.Tensor,
    slots: torch.Tensor,
    cache: torch.Tensor,
    scale_cache: torch.Tensor,
) -> None:
    """Publish at the existing cache-write point, after the connector wait."""
    rows = slots.numel()
    key_width = quant.shape[-1]
    # Rows only set the launch grid; 12 was a test setting, not a kernel limit.
    _store_indexer_key_scale[(rows,)](quant, scales, slots, cache, scale_cache, cache.numel() // key_width, key_width)
