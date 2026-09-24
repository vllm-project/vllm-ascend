"""PCP KV cache transfer helpers."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


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
    WRITE: tl.constexpr,
    B: tl.constexpr,
):
    d = tl.arange(0, B)
    for t in range(tl.program_id(0), N, tl.num_programs(0)):
        s = tl.load(S + t).to(tl.int64)
        valid = s >= 0
        ko = (s // BS) * KS0 + (s % BS) * KS1 + d * KS3
        ro = (s // BS) * RS0 + (s % BS) * RS1 + d * RS3
        if WRITE:
            k = tl.load(P + t * (KD + RD) + d, mask=d < KD, other=0)
            r = tl.load(P + t * (KD + RD) + KD + d, mask=d < RD, other=0)
            tl.store(K + ko, k, mask=valid & (d < KD))
            tl.store(R + ro, r, mask=valid & (d < RD))
        else:
            k = tl.load(K + ko, mask=valid & (d < KD), other=0)
            r = tl.load(R + ro, mask=valid & (d < RD), other=0)
            tl.store(P + t * (KD + RD) + d, k, mask=d < KD)
            tl.store(P + t * (KD + RD) + KD + d, r, mask=d < RD)


def copy_pcp_kv_cache(cache, slots, packed=None):
    """Pack selected cache rows, or scatter packed rows back, skipping -1 slots.

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
    write = packed is not None
    if packed is None:
        packed = torch.empty((slots.numel(), k.shape[-1] + rope_dim), dtype=k.dtype, device=k.device)
    assert packed.dtype == k.dtype and packed.device == k.device
    assert packed.is_contiguous() and packed.shape == (slots.numel(), k.shape[-1] + rope_dim)
    if slots.numel():
        init_device_properties_triton()
        _copy_pcp_kv_cache_kernel[(min(slots.numel(), get_vectorcore_num()),)](
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
            write,
            triton.next_power_of_2(max(k.shape[-1], rope_dim)),
        )
    return packed
