# SPDX-License-Identifier: Apache-2.0
"""Fuse native-DCP decode remap arithmetic around native sort and gather.

The validated fast path is DCP8, interleave128, Top-K2048, contiguous int32
indices and at most12 decode rows. Other inputs retain the caller's upstream
path. Integer coordinates preserve this release's remapping semantics even
above 2**24; only the unique position keys used for sorting are FP32.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_DCP_SIZE = 8
_INTERLEAVE_SIZE = 128
_TOPK = 2048
_MAX_DECODE_ROWS = 12
_SINGLE_ROW_BLOCK = 128
_MULTI_ROW_BLOCK = 256


if triton is not None:

    @triton.jit
    def _prepare_remap(
        indices,
        keys,
        mapped,
        elements: tl.constexpr,
        count: tl.constexpr,
        ranks: tl.constexpr,
        rank: tl.constexpr,
        interleave: tl.constexpr,
        programs: tl.constexpr,
        block: tl.constexpr,
    ):
        for tile in tl.range(tl.program_id(0), tl.cdiv(elements, block), programs):
            offset = tile * block + tl.arange(0, block)
            value = tl.load(indices + offset, mask=offset < elements, other=-1)
            local_block = value // interleave
            owner = local_block - (local_block // ranks) * ranks
            owned = (value >= 0) & (owner == rank)
            if interleave == 1:
                local = value // ranks
            else:
                local_offsets = value - local_block * interleave
                local = (value // (ranks * interleave)) * interleave + local_offsets
            key = (offset % count).to(tl.float32) + tl.where(owned, 0.0, count)
            tl.store(keys + offset, key, mask=offset < elements)
            tl.store(mapped + offset, tl.where(owned, local, -1), mask=offset < elements)


def can_fuse_remap(indices: torch.Tensor) -> bool:
    return (
        triton is not None
        and indices.device.type == "npu"
        and indices.dtype == torch.int32
        and indices.ndim > 0
        and indices.shape[-1] == _TOPK
        and 0 < indices.numel() <= _MAX_DECODE_ROWS * _TOPK
        and indices.is_contiguous()
    )


def fused_remap(indices: torch.Tensor, rank: int) -> torch.Tensor:
    if not can_fuse_remap(indices) or not 0 <= rank < _DCP_SIZE:
        raise ValueError("Unsupported native-DCP remap input or rank")
    elements = indices.numel()
    keys = torch.empty_like(indices, dtype=torch.float32)
    mapped = torch.empty_like(indices)
    block = _SINGLE_ROW_BLOCK if elements == _TOPK else _MULTI_ROW_BLOCK
    cores = triton.runtime.driver.active.utils.get_device_properties(indices.device.index)["num_vectorcore"]
    programs = min(triton.cdiv(elements, block), cores)
    _prepare_remap[(programs,)](
        indices,
        keys,
        mapped,
        elements,
        _TOPK,
        _DCP_SIZE,
        rank,
        _INTERLEAVE_SIZE,
        programs,
        block,
    )
    # Native sort/gather outperform the tested scan/scatter and indirect-load
    # Triton variants. Unique position keys preserve the original Top-K order.
    _, order = torch.sort(keys, dim=-1)
    return torch.gather(mapped, dim=-1, index=order.to(torch.int32))
