"""PCP KV cache transfer helpers."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_ub_size_bytes, get_vectorcore_num, init_device_properties_triton


@triton.jit
def _copy_pcp_kv_cache_kernel(
    K,
    R,
    S,
    P,
    N,
    BS: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS3: tl.constexpr,
    RS0: tl.constexpr,
    RS1: tl.constexpr,
    RS3: tl.constexpr,
    KD: tl.constexpr,
    RD: tl.constexpr,
    B: tl.constexpr,
    ROWS: tl.constexpr,
):
    if ROWS == 1:
        # Keep scalar slot addressing for small batches: the Ascend compiler
        # cannot lower the modulo expression in a singleton 2D tile.
        d = tl.arange(0, B)
        for t in range(tl.program_id(0), N, tl.num_programs(0)):
            s = tl.load(S + t).to(tl.int64)
            valid = s >= 0
            ko = (s // BS) * KS0 + (s % BS) * KS1 + d * KS3
            ro = (s // BS) * RS0 + (s % BS) * RS1 + d * RS3
            k = tl.load(K + ko, mask=valid & (d < KD), other=0)
            r = tl.load(R + ro, mask=valid & (d < RD), other=0)
            tl.store(P + t * (KD + RD) + d, k, mask=d < KD)
            tl.store(P + t * (KD + RD) + KD + d, r, mask=d < RD)
    else:
        d = tl.arange(0, B)[None, :]
        for tile in range(tl.program_id(0), tl.cdiv(N, ROWS), tl.num_programs(0)):
            t = tile * ROWS + tl.arange(0, ROWS)
            s = tl.load(S + t, mask=t < N, other=-1).to(tl.int64)
            valid = (t < N) & (s >= 0)
            ko = (s[:, None] // BS) * KS0 + (s[:, None] % BS) * KS1 + d * KS3
            k = tl.load(K + ko, mask=valid[:, None] & (d < KD), other=0)
            tl.store(P + t[:, None] * (KD + RD) + d, k, mask=(t[:, None] < N) & (d < KD))
            if RD > 0:
                ro = (s[:, None] // BS) * RS0 + (s[:, None] % BS) * RS1 + d * RS3
                r = tl.load(R + ro, mask=valid[:, None] & (d < RD), other=0)
                tl.store(P + t[:, None] * (KD + RD) + KD + d, r, mask=(t[:, None] < N) & (d < RD))


def _get_pcp_kv_cache_rows(num_tokens: int, num_cores: int, block_cols: int, element_size: int, num_caches: int) -> int:
    """Bound row batching by core occupancy and a conservative UB estimate."""
    # Reserve half the UB for compiler temporaries and buffering.
    # Per cache, budget two INT64 address tiles plus four payload/scratch tiles.
    # INT8 loads may use FP16 temporaries, so budget at least two bytes per element.
    # This estimates compiler usage; it is not an exact peak-liveness calculation.
    bytes_per_row = block_cols * num_caches * (2 * 8 + 4 * max(element_size, 2))
    ub_rows = (get_ub_size_bytes() // 2) // bytes_per_row
    # Keep the validated maximum, and preserve the single-row path when even one
    # row exceeds the estimate. Wider layouts still need compilation validation.
    row_limit = max(1, min(8, num_tokens // num_cores, ub_rows))
    return 1 << (row_limit.bit_length() - 1)


def copy_pcp_kv_cache(cache, slots):
    """Read selected cache rows into a contiguous tensor, zero-filling -1 slots.

    A single tensor represents the complete C8 row, including RoPE and scales.
    Access that layout through an int8 view to preserve all bits, even when its
    storage dtype is FP8. Two tensors represent separate latent and RoPE caches.
    """
    assert len(cache) in (1, 2)
    k = cache[0]
    assert k.ndim == 4 and k.shape[2] == 1
    if len(cache) == 1:
        assert k.element_size() == 1
        k = k.view(torch.int8)
        r = k  # Unused pointer: RD=0 eliminates the second cache's accesses.
        rope_dim = 0
    else:
        r = cache[1]
        assert r.ndim == 4 and r.shape[2] == 1
        assert k.dtype == r.dtype and k.shape[:3] == r.shape[:3]
        rope_dim = r.shape[-1]
    slots = slots.contiguous()
    packed = torch.empty((slots.numel(), k.shape[-1] + rope_dim), dtype=k.dtype, device=k.device)
    if slots.numel():
        init_device_properties_triton()
        num_cores = get_vectorcore_num()
        block_cols = triton.next_power_of_2(max(k.shape[-1], rope_dim))
        rows = _get_pcp_kv_cache_rows(slots.numel(), num_cores, block_cols, k.element_size(), len(cache))
        _copy_pcp_kv_cache_kernel[(min(triton.cdiv(slots.numel(), rows), num_cores),)](
            k,
            r,
            slots,
            packed,
            slots.numel(),
            k.shape[1],
            k.stride(0),
            k.stride(1),
            k.stride(3),
            r.stride(0),
            r.stride(1),
            r.stride(3),
            k.shape[-1],
            rope_dim,
            block_cols,
            rows,
        )
    return packed
