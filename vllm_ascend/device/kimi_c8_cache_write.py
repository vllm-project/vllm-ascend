from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.attention.kimi_c8_static import FP8_E4M3_MAX


@triton.jit
def _scatter_pa_nz_bytes_kernel(
    raw,
    payload,
    slots,
    NUM_BYTES: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    SOURCE_BLOCK_SIZE: tl.constexpr,
):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    active = index < NUM_BYTES
    token, feature = index // WIDTH, index % WIDTH
    slot = tl.load(slots + token, mask=active, other=-1).to(tl.int64)
    safe_slot = tl.maximum(slot, 0)
    offset = (
        safe_slot // BLOCK_SIZE * PAGE_STRIDE
        + feature // 32 * (BLOCK_SIZE * 32)
        + safe_slot % BLOCK_SIZE * 32
        + feature % 32
    )
    source = index
    if SOURCE_BLOCK_SIZE:
        source = (
            token // SOURCE_BLOCK_SIZE * (SOURCE_BLOCK_SIZE * WIDTH)
            + feature // 32 * (SOURCE_BLOCK_SIZE * 32)
            + token % SOURCE_BLOCK_SIZE * 32
            + feature % 32
        )
    value = tl.load(payload + source, mask=active, other=0)
    tl.store(raw + offset, value, mask=active & (slot >= 0))


def _scatter_pa_nz_bytes(
    cache: torch.Tensor, payload: torch.Tensor, slots: torch.Tensor, *, source_block_size=0
) -> None:
    count, width = slots.numel(), 576
    _scatter_pa_nz_bytes_kernel[(triton.cdiv(count * width, 256),)](
        cache,
        payload,
        slots,
        NUM_BYTES=count * width,
        WIDTH=width,
        TILE=256,
        BLOCK_SIZE=cache.shape[3],
        PAGE_STRIDE=cache.stride(0),
        SOURCE_BLOCK_SIZE=source_block_size,
    )


def copy_prolog_kv_to_cache(source: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor) -> None:
    """Scatter only this step's Prolog-128 rows into the native FA cache."""
    _scatter_pa_nz_bytes(cache.view(torch.uint8), source.view(torch.uint8), slots, source_block_size=128)


def write_static_fp8_cache_graph(merged: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor, k_descale) -> None:
    """Fixed-shape quantization plus masked byte scatter for ACL Graph.

    k_descale is a loaded, validated positive scalar. Slot mapping has already
    converted logical manager pages into aligned physical pages. Every token
    retains its row, including padded negative slots; no dynamic nonzero/index
    tensor or host read is introduced during capture/replay.
    """
    if merged.ndim != 2 or merged.shape[1] != 576 or slots.numel() != merged.shape[0]:
        raise ValueError("C8 Graph writer requires [T,576] and one slot per token")
    if cache.ndim != 5 or cache.shape[1:3] != (1, 18) or cache.shape[4] != 32 or not cache[0].is_contiguous():
        raise ValueError("C8 Graph writer requires dense native PA_NZ block payloads")
    if cache.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError("C8 Graph writer requires E4M3 byte storage")
    scale = torch.as_tensor(k_descale, device=merged.device, dtype=torch.float32)
    if scale.numel() != 1:
        raise ValueError("C8 Graph writer requires one prevalidated static KV descale")
    payload = (merged.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
    if merged.shape[0]:
        _scatter_pa_nz_bytes(cache.view(torch.uint8), payload.contiguous(), slots.reshape(-1).to(torch.int64))
