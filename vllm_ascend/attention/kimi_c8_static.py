from __future__ import annotations

import torch

FP8_E4M3_MAX = 448.0
PA_NZ_TILE = 32
PA_NZ_BLOCK_SIZE = 128  # Prolog/V2 table granularity; FA uses cache.shape[3].
K3_MLA_CACHE_WIDTH = 576


def quantize_static_fp8(x: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    s = torch.as_tensor(scale, device=x.device, dtype=torch.float32)
    if s.numel() != 1 or not bool(torch.isfinite(s).all()) or bool((s <= 0).any()):
        raise ValueError("static scale must be one finite positive scalar")
    return (x.float() / s).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def dequantize_static_fp8(x: torch.Tensor, scale: torch.Tensor | float, dtype=torch.bfloat16) -> torch.Tensor:
    s = torch.as_tensor(scale, device=x.device, dtype=torch.float32)
    if s.numel() != 1 or not bool(torch.isfinite(s).all()) or bool((s <= 0).any()):
        raise ValueError("static scale must be one finite positive scalar")
    return x.view(torch.float8_e4m3fn).float().mul(s).to(dtype)


def pa_nz_indices(
    slots: torch.Tensor, block_size: int, width: int, *, page_stride_bytes: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if width <= 0 or width % PA_NZ_TILE:
        raise ValueError("width must be a positive multiple of 32")
    _kernel_blocks_per_manager(block_size)
    page_stride_bytes = block_size * width if page_stride_bytes is None else page_stride_bytes
    flat = slots.reshape(-1).to(torch.int64)
    valid = flat >= 0
    slot = flat[valid]
    tiles = width // PA_NZ_TILE
    indices = (
        (slot // block_size)[:, None] * (page_stride_bytes // PA_NZ_TILE)
        + torch.arange(tiles, device=slot.device)[None, :] * block_size
        + slot.remainder(block_size)[:, None]
    ).reshape(-1)
    return valid, indices


def write_static_fp8_cache(merged: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor, scale) -> None:
    if merged.ndim != 2 or cache.ndim != 5 or cache.dtype != torch.uint8:
        raise ValueError("merged/cache layout is invalid")
    block_size, tiles = cache.shape[3], cache.shape[2]
    if cache.shape[1] != 1 or cache.shape[4] != PA_NZ_TILE or merged.shape[1] != tiles * PA_NZ_TILE:
        raise ValueError("cache and merged widths do not match PA_NZ")
    if slots.numel() != merged.shape[0]:
        raise ValueError("C8 writer requires one slot per input token")
    if bool((slots >= cache.shape[0] * block_size).any()):
        raise ValueError("C8 writer slot exceeds physical cache capacity")
    valid, indices = pa_nz_indices(slots, block_size, merged.shape[1], page_stride_bytes=cache.stride(0))
    rows = quantize_static_fp8(merged[valid], scale).reshape(-1, PA_NZ_TILE)
    pa_nz_storage_view(cache).view(-1, PA_NZ_TILE).index_copy_(0, indices, rows)


def pa_nz_cache_view(
    raw: torch.Tensor,
    num_blocks: int,
    block_size: int,
    width: int = K3_MLA_CACHE_WIDTH,
    *,
    page_stride_bytes: int | None = None,
) -> torch.Tensor:
    """Expose one native PA_NZ block per manager page, including page stride."""
    if raw.dtype != torch.uint8 or raw.ndim != 1 or raw.stride(0) != 1:
        raise ValueError("raw PA_NZ backing must be a contiguous uint8 vector")
    if width <= 0 or width % PA_NZ_TILE or num_blocks <= 0:
        raise ValueError("invalid PA_NZ geometry")
    page_stride_bytes = block_size * width if page_stride_bytes is None else page_stride_bytes
    validate_pa_nz_page_stride(block_size, page_stride_bytes, width)
    required = num_blocks * page_stride_bytes
    if raw.numel() < required:
        raise ValueError("raw backing is smaller than PA_NZ payload")
    return torch.as_strided(
        raw[:required],
        size=(num_blocks, 1, width // PA_NZ_TILE, block_size, PA_NZ_TILE),
        stride=(page_stride_bytes, block_size * width, block_size * PA_NZ_TILE, PA_NZ_TILE, 1),
        storage_offset=raw.storage_offset(),
    )


def _kernel_blocks_per_manager(block_size: int) -> int:
    if block_size <= 0 or block_size % PA_NZ_BLOCK_SIZE:
        raise ValueError("C8 manager block size must be a positive multiple of 128")
    return block_size // PA_NZ_BLOCK_SIZE


def pa_nz_prolog_view(cache: torch.Tensor) -> torch.Tensor:
    """Expose the logical PA descriptor without changing physical NZ bytes."""
    if (
        cache.ndim != 5
        or cache.shape[1:3] != (1, K3_MLA_CACHE_WIDTH // PA_NZ_TILE)
        or cache.shape[4] != PA_NZ_TILE
        or cache.dtype not in (torch.uint8, torch.float8_e4m3fn)
        or not cache[0].is_contiguous()
    ):
        raise ValueError("C8 requires [blocks,1,18,block_tokens,32] PA_NZ cache")
    return cache.view(torch.float8_e4m3fn).view(cache.shape[0], cache.shape[3], 1, K3_MLA_CACHE_WIDTH)


def pa_nz_storage_view(cache: torch.Tensor) -> torch.Tensor:
    """Byte coverage of complete physical pages for zero/CoW, including padding."""
    return torch.as_strided(
        cache.view(torch.uint8),
        size=(cache.shape[0], cache.stride(0)),
        stride=(cache.stride(0), 1),
        storage_offset=cache.storage_offset(),
    )


def read_static_fp8_cache(
    cache: torch.Tensor,
    kernel_block_table: torch.Tensor,
    seq_lens: torch.Tensor | list[int],
    scale: torch.Tensor | float,
    *,
    starts: torch.Tensor | list[int] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Gather eager prefill history as request-ordered [sum(lengths),1,576].

    The table contains native physical PA_NZ block IDs. Each length is
    the requested chunk length; starts are absolute positions in each request.
    Only the requested rows are read, so unused negative/padded table entries
    are harmless. Physical NZ tiles are gathered before merging feature lanes.
    """
    pa_nz_prolog_view(cache)
    block_size = cache.shape[3]
    if kernel_block_table.ndim != 2 or kernel_block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("C8 history block table must be a rank-2 integer tensor")
    # Keep scheduler-owned host lengths/starts on CPU for validation and the
    # output size. Copying them to NPU before sum().item() stalls every layer.
    lengths = torch.as_tensor(seq_lens, dtype=torch.int64)
    offsets = torch.zeros_like(lengths) if starts is None else torch.as_tensor(starts, dtype=torch.int64)
    if lengths.ndim != 1 or offsets.shape != lengths.shape or lengths.numel() != kernel_block_table.shape[0]:
        raise ValueError("C8 history lengths/starts must match block table requests")
    if bool((lengths < 0).any()) or bool((offsets < 0).any()):
        raise ValueError("C8 history lengths/starts must be non-negative")
    host_bounds = lengths.device.type == offsets.device.type == "cpu"
    if host_bounds and bool(((lengths > 0) & (offsets + lengths > kernel_block_table.shape[1] * block_size)).any()):
        raise ValueError("C8 history block table does not cover requested positions")
    total = int(lengths.sum().item())
    lengths = lengths.to(device=cache.device)
    offsets = offsets.to(device=cache.device)
    if total == 0:
        return torch.empty((0, 1, K3_MLA_CACHE_WIDTH), dtype=dtype, device=cache.device)
    requests = torch.repeat_interleave(torch.arange(lengths.numel(), device=cache.device), lengths, output_size=total)
    positions = torch.arange(total, device=cache.device) - (lengths.cumsum(0) - lengths)[requests] + offsets[requests]
    columns = positions // block_size
    if not host_bounds and bool((columns >= kernel_block_table.shape[1]).any()):
        raise ValueError("C8 history block table does not cover requested positions")
    table = kernel_block_table.to(device=cache.device)
    block_ids = table[requests, columns].long()
    if bool(((block_ids < 0) | (block_ids >= cache.shape[0])).any()):
        raise ValueError("C8 history active block table entry is outside cache bounds")
    tiles = cache.view(torch.uint8)[block_ids, 0, :, positions % block_size, :]
    rows = tiles.reshape(total, K3_MLA_CACHE_WIDTH)
    return dequantize_static_fp8(rows, scale, dtype).unsqueeze(1)


def validate_pa_nz_page_stride(block_size: int, page_stride_bytes: int, width: int = K3_MLA_CACHE_WIDTH) -> None:
    """Allow only complete physical kernel pages between manager-page bases."""
    _kernel_blocks_per_manager(block_size)
    if (
        width <= 0
        or width % PA_NZ_TILE
        or page_stride_bytes < block_size * width
        or page_stride_bytes % (PA_NZ_BLOCK_SIZE * width)
    ):
        raise ValueError("C8 PA_NZ page padding must contain whole 128-token kernel pages")


def map_pa_nz_slots(slots: torch.Tensor, block_size: int, page_stride_bytes: int) -> torch.Tensor:
    """Native-page slots remain logical token indices; cache stride carries padding."""
    return slots


def map_pa_nz_kernel_block_table(kernel_ids: torch.Tensor, block_size: int, page_stride_bytes: int) -> torch.Tensor:
    """Coalesce V2's small table units into native FA page IDs."""
    ratio = _kernel_blocks_per_manager(block_size)
    return torch.div(kernel_ids[..., ::ratio], ratio, rounding_mode="floor").contiguous()
