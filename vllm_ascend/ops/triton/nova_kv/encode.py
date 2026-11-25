# SPDX-License-Identifier: Apache-2.0
"""Encode NOVA-KV keys and INT2 values directly into paged decode storage."""

from vllm.triton_utils import tl, triton

from vllm_ascend.quantization.methods.kv_cache.nova_kv.codec import head_transform


def prepare_encoder(codec):
    if not hasattr(codec, "_encoder_books"):
        codec._encoder_books = codec._books_float.transpose(-1, -2).contiguous()


@triton.jit
def _encode_write(
    K,
    V,
    BOOK,
    NORM,
    TOKEN_SLOTS,
    REQUESTS,
    KC,
    VC,
    SC,
    KEEP,
    H: tl.constexpr,
    PAGE: tl.constexpr,
    KS: tl.constexpr,
    SS: tl.constexpr,
    N: tl.constexpr,
    Q: tl.constexpr,
    MASK_REQUESTS: tl.constexpr,
):
    rh, group_tile = tl.program_id(0), tl.program_id(1)
    token, head = rh // H, rh % H
    slot = tl.load(TOKEN_SLOTS + token).to(tl.int64)
    valid = slot >= 0
    if MASK_REQUESTS:
        request = tl.load(REQUESTS + token)
        valid &= (request >= 0) & (request < KEEP)
    if valid:
        dim = tl.arange(0, 256)
        groups = group_tile * Q + tl.arange(0, Q)
        c = tl.arange(0, 256)
        comp = tl.arange(0, 4)
        k = tl.load(K + rh * 256 + dim)
        rho = tl.maximum(tl.sqrt(tl.sum(k * k, 0) * (1 / 256.0)), 1e-8)
        row = tl.load(K + rh * 256 + groups[:, None] * 4 + comp[None, :]) / rho
        lookup = tl.load(
            BOOK + ((head * 64 + groups[:, None, None]) * 4 + comp[None, :, None]) * 256 + c[None, None, :]
        )
        dot = tl.sum(lookup * row[:, :, None], 1)
        norm = tl.load(NORM + (head * 64 + groups[:, None]) * 256 + c[None, :])
        distance = tl.sum(row * row, 1)[:, None] + norm - 2.0 * dot
        code = tl.argmin(distance, 1, tie_break_left=True)
        v = tl.load(V + rh * 256 + dim)
        low = tl.min(v, 0)
        step = tl.maximum((tl.max(v, 0) - low) / 3.0, 1e-8)
        values = tl.load(V + rh * 256 + groups[:, None] * 4 + comp[None, :])
        y = (values - low) / step
        lower = tl.floor(y)
        frac = y - lower
        rounded = lower + ((frac > 0.5) | ((frac == 0.5) & ((lower.to(tl.int32) & 1) != 0))).to(tl.float32)
        symbols = tl.minimum(tl.maximum(rounded, 0.0), 3.0).to(tl.int32)
        packed = tl.sum(symbols << (2 * comp[None, :]), 1)
        p, offset = slot // PAGE, slot % PAGE
        dst = p * KS + (head * 64 + groups) * PAGE + offset
        tl.store(KC + dst, code)
        tl.store(VC + dst, packed)
        if group_tile == 0:
            scales = SC + p * SS + head * 3 * PAGE + offset
            tl.store(scales, rho)
            tl.store(scales + PAGE, step)
            tl.store(scales + 2 * PAGE, low)


def encode_decode_cache(key, value, codec, views, slots, requests=None, limit=0, groups=4):
    rows, heads, dim = key.shape
    if dim != 256 or rows > 4 or value.shape != key.shape or groups not in (1, 2, 4, 8):
        raise ValueError("NOVA-KV decode cache encoding expects up to four tokens and D256")
    if not hasattr(codec, "_encoder_books"):
        raise RuntimeError("prepare encoder before launching")
    k = head_transform(key.float() - codec.key_mean, codec.key_transform).contiguous()
    v = head_transform(value, codec.value_transform).contiguous()
    kc, vc, sc = views
    _encode_write[(rows * heads, 64 // groups)](
        k,
        v,
        codec._encoder_books,
        codec._book_norms,
        slots,
        slots if requests is None else requests,
        kc,
        vc,
        sc,
        limit,
        heads,
        kc.shape[-1],
        kc.stride(0),
        sc.stride(0),
        rows,
        groups,
        requests is not None,
        compile_mode="simd",
        num_stages=1,
        enable_fp_fusion=False,
        enable_auto_bind_sub_block=False,
    )
