# SPDX-License-Identifier: Apache-2.0
"""Paged NOVA-KV attention with tile-local reconstruction and online softmax."""

import math

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout

_QUERY_TILE = 16


@tilelang.jit(target="ascend")
def _build_prepare(batch_heads, groups):
    block_q = _QUERY_TILE

    @T.prim_func
    def main(
        Q: T.Buffer((batch_heads, groups, 256), "float32"),
        MEAN: T.Buffer((batch_heads, groups), "float32"),
        PADDED: T.Buffer((batch_heads, block_q, 256), "float16"),
        PADDED_MEAN: T.Buffer((batch_heads, block_q), "float32"),
    ):
        with T.Kernel(batch_heads) as bh, T.SimtVF(threads=128):
            for g, d in T.Parallel(block_q, 256):
                PADDED[bh, g, d] = T.if_then_else(g < groups, T.Cast("float16", Q[bh, g, d]), 0.0)
            for g in T.Parallel(block_q):
                PADDED_MEAN[bh, g] = T.if_then_else(g < groups, MEAN[bh, g], 0.0)

    return main


@tilelang.jit(target="ascend")
def _build_merge(batch_heads, groups, splits):
    block_q = _QUERY_TILE

    @T.prim_func
    def main(
        PARTIAL: T.Buffer((batch_heads, splits, block_q, 256), "float32"),
        LSE: T.Buffer((batch_heads, splits, block_q), "float32"),
        OUT: T.Buffer((batch_heads, groups, 256), "float32"),
        OUT_LSE: T.Buffer((batch_heads, groups), "float32"),
    ):
        with T.Kernel(batch_heads * groups) as bx:
            bh = bx // groups
            group = bx % groups
            with T.SimtVF(threads=128):
                for d in T.Parallel(256):
                    peak = T.alloc_var("float32", init=-1.0e30)
                    for s in T.serial(splits):
                        peak = T.max(peak, LSE[bh, s, group])
                    denominator = T.alloc_var("float32", init=0.0)
                    numerator = T.alloc_var("float32", init=0.0)
                    for s in T.serial(splits):
                        weight = T.exp(LSE[bh, s, group] - peak)
                        denominator += weight
                        numerator += weight * PARTIAL[bh, s, group, d]
                    OUT[bh, group, d] = numerator / T.max(denominator, 1.0e-30)
                    if d == 0:
                        OUT_LSE[bh, group] = T.if_then_else(
                            denominator > 0, peak + T.log(denominator), -T.infinity("float32")
                        )

    return main


@tilelang.jit(
    target="ascend",
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
    },
)
def _build_nova_fia(
    batch,
    heads,
    groups,
    pages,
    page_tokens,
    page_stride,
    scale_stride,
    table_width,
    splits,
    split_tokens,
    softmax_scale,
    stages=1,
    num_blocks=32,
):
    """Build a persistent mixed kernel with UB-resident books and streaming NZ tiles."""
    dim = 256
    block_q = _QUERY_TILE
    block_kv = 128
    rows = block_q // 2
    kv_columns = dim // 2
    tasks = batch * heads * splits
    tiles = split_tokens // block_kv
    dtype = "float16"

    @T.prim_func
    def main(
        Q: T.Buffer((batch * heads, block_q, dim), dtype),
        MEAN: T.Buffer((batch * heads, block_q), "float32"),
        KC: T.Buffer((pages, heads, 64, page_tokens), "uint8", strides=(page_stride, 64 * page_tokens, page_tokens, 1)),
        VC: T.Buffer((pages, heads, 64, page_tokens), "uint8", strides=(page_stride, 64 * page_tokens, page_tokens, 1)),
        SC: T.Buffer(
            (pages, heads, 3, page_tokens), "bfloat16", strides=(scale_stride, 3 * page_tokens, page_tokens, 1)
        ),
        BOOK: T.Buffer((heads * 64 * 256 * 4,), dtype),
        TABLE: T.Buffer((batch, table_width), "int32"),
        STARTS: T.Buffer((batch,), "int32"),
        ENDS: T.Buffer((batch,), "int32"),
        OUT: T.Buffer((batch * heads, splits, block_q, dim), "float32"),
        LSE: T.Buffer((batch * heads * splits * block_q,), "float32"),
    ):
        with T.MixedKernel(num_blocks) as (bx, sid):
            # D128 subproducts share L0B; each AIV owns half the codebook channels.
            q_l1 = T.alloc_l1((block_q, dim), dtype)
            k_l1 = T.alloc_l1((dim, block_kv), dtype)
            v_l1 = T.alloc_l1((dim, block_kv), dtype)
            p_l1 = T.alloc_l1((block_q, block_kv), dtype)
            q_l0 = T.alloc_l0a((block_q, kv_columns), dtype)
            k_l0 = T.alloc_l0b((block_kv, kv_columns), dtype)
            p_l0 = T.alloc_l0a((block_q, block_kv), dtype)
            qk_l0 = T.alloc_l0c((block_q, block_kv), "float32")
            pv_l0 = T.alloc_l0c((block_q, kv_columns), "float32")
            book_ub = T.alloc_shared((32 * 256 * 4,), dtype)
            codes_ub = T.alloc_shared((33, block_kv), "uint8")
            values_ub = T.alloc_shared((33, block_kv), "uint8")
            scales_ub = T.alloc_shared((3, block_kv), "bfloat16")
            bounds_ub = T.alloc_shared((16,), "int32")
            k_ub = T.alloc_shared((kv_columns + 1, block_kv), dtype)
            v_ub = T.alloc_shared((kv_columns + 1, block_kv), dtype)
            T.annotate_layout({k_ub: make_ascend_compact_nz_layout(k_ub), v_ub: make_ascend_compact_nz_layout(v_ub)})
            score = T.alloc_shared((rows, block_kv), "float32")
            prob_half = T.alloc_shared((rows + 1, block_kv), dtype)
            T.annotate_layout({prob_half: make_ascend_compact_nz_layout(prob_half)})
            output = T.alloc_shared((rows, dim), "float32")
            subtotal = T.alloc_shared((rows, dim), "float32")
            maximum = T.alloc_shared((rows,), "float32")
            denominator = T.alloc_shared((rows,), "float32")
            alpha = T.alloc_shared((rows,), "float32")
            final_lse = T.alloc_shared((rows,), "float32")

            for task in T.Persistent([tasks], num_blocks, bx):
                bh = task // splits
                split = task % splits
                request = bh // heads
                head = bh % heads
                T.copy(Q[bh, 0:block_q, 0:dim], q_l1)
                T.copy(BOOK[head * 65536 + sid * 32768 : head * 65536 + (sid + 1) * 32768], book_ub)
                T.copy(STARTS[request : request + 1], bounds_ub[0:1])
                T.copy(ENDS[request : request + 1], bounds_ub[8:9])
                with T.SimtVF(threads=128):
                    for r, d in T.Parallel(rows, dim):
                        output[r, d] = 0.0
                    for r in T.Parallel(rows):
                        maximum[r] = -1.0e30
                        denominator[r] = 0.0

                for tile in T.Pipelined(tiles, num_stages=stages):
                    tile_start = split * split_tokens + tile * block_kv
                    page = TABLE[request, T.min(tile_start // page_tokens, table_width - 1)]
                    offset = tile_start % page_tokens
                    T.copy(
                        KC[page, head, sid * 32 : (sid + 1) * 32, offset : offset + block_kv],
                        codes_ub[0:32, 0:block_kv],
                    )
                    T.copy(
                        VC[page, head, sid * 32 : (sid + 1) * 32, offset : offset + block_kv],
                        values_ub[0:32, 0:block_kv],
                    )
                    T.copy(SC[page, head, 0:3, offset : offset + block_kv], scales_ub)
                    with T.SimdVF():
                        # Reconstruct directly in NZ order to avoid an ND transpose.
                        decode_full = S.pset(32, "PAT_ALL")
                        decode_half = S.pset(16, "PAT_ALL")
                        token_positions = S.vci(T.int16(0), "int16")
                        relative_start = T.max(0, T.min(block_kv, bounds_ub[0] - tile_start))
                        relative_end = T.max(0, T.min(block_kv, bounds_ub[8] - tile_start))
                        valid_tokens = S.pand(
                            S.vcmps(token_positions, T.Cast("int16", relative_start), decode_half, op="ge"),
                            S.vcmps(token_positions, T.Cast("int16", relative_end), decode_half, op="lt"),
                            decode_half,
                        )
                        key_nz_ptr = S.make_ubuf_ptr(T.access_ptr(k_ub[0, 0], "w", kv_columns + 1, block_kv), dtype)
                        value_nz_ptr = S.make_ubuf_ptr(T.access_ptr(v_ub[0, 0], "w", kv_columns + 1, block_kv), dtype)
                        rho_values = S.vld(scales_ub[0, 0], "NORM")
                        step_values = S.vld(scales_ub[1, 0], "NORM")
                        low_values = S.vld(scales_ub[2, 0], "NORM")
                        rho_even = S.vcvt(rho_values, "float32", decode_half, part=0)
                        rho_odd = S.vcvt(rho_values, "float32", decode_half, part=1)
                        step_even = S.vcvt(step_values, "float32", decode_half, part=0)
                        step_odd = S.vcvt(step_values, "float32", decode_half, part=1)
                        low_even = S.vcvt(low_values, "float32", decode_half, part=0)
                        low_odd = S.vcvt(low_values, "float32", decode_half, part=1)
                        for component_group in range(32):
                            code_vector = S.vunpack(S.vld(codes_ub[component_group, 0], "NORM"), part=0)
                            value_vector = S.vunpack(S.vld(values_ub[component_group, 0], "NORM"), part=0)
                            for component in range(4):
                                indices = S.vadds(
                                    S.vshl(code_vector, S.vdup(T.int16(2), "int16"), decode_half),
                                    T.uint16(component_group * 1024 + component),
                                    decode_half,
                                )
                                lookup = S.vgather2(book_ub[0], indices, decode_half)
                                symbols = S.vand(
                                    S.vshr(value_vector, S.vdup(T.int16(component * 2), "int16"), decode_half),
                                    S.vdup(T.uint16(3), "uint16"),
                                    decode_half,
                                )
                                key_even = S.vmul(S.vcvt(lookup, "float32", decode_half, part=0), rho_even, decode_full)
                                key_odd = S.vmul(S.vcvt(lookup, "float32", decode_half, part=1), rho_odd, decode_full)
                                value_even = S.vadd(
                                    S.vmul(
                                        S.vcvt(T.reinterpret(symbols, "int16x128"), "float32", decode_half, part=0),
                                        step_even,
                                        decode_full,
                                    ),
                                    low_even,
                                    decode_full,
                                )
                                value_odd = S.vadd(
                                    S.vmul(
                                        S.vcvt(T.reinterpret(symbols, "int16x128"), "float32", decode_half, part=1),
                                        step_odd,
                                        decode_full,
                                    ),
                                    low_odd,
                                    decode_full,
                                )
                                packed_key = S.vor(  # codespell:ignore vor
                                    T.reinterpret(
                                        S.vcvt(key_even, dtype, decode_full, sat=False, part=0), "uint16x128"
                                    ),
                                    T.reinterpret(S.vcvt(key_odd, dtype, decode_full, sat=False, part=1), "uint16x128"),
                                    decode_half,
                                )
                                packed_value = S.vor(  # codespell:ignore vor
                                    T.reinterpret(
                                        S.vcvt(value_even, dtype, decode_full, sat=False, part=0), "uint16x128"
                                    ),
                                    T.reinterpret(
                                        S.vcvt(value_odd, dtype, decode_full, sat=False, part=1), "uint16x128"
                                    ),
                                    decode_half,
                                )
                                key_nz_ptr = S.vsstb(
                                    S.vsel(
                                        T.reinterpret(packed_key, "float16x128"),
                                        S.vdup(T.float16(0), dtype),
                                        valid_tokens,
                                    ),
                                    key_nz_ptr,
                                    T.int32(((kv_columns + 1) << 16) | 1),
                                    decode_half,
                                    update=True,
                                )
                                value_nz_ptr = S.vsstb(
                                    S.vsel(
                                        T.reinterpret(packed_value, "float16x128"),
                                        S.vdup(T.float16(0), dtype),
                                        valid_tokens,
                                    ),
                                    value_nz_ptr,
                                    T.int32(((kv_columns + 1) << 16) | 1),
                                    decode_half,
                                    update=True,
                                )
                    T.dual_copy(k_ub[0:kv_columns, 0:block_kv], k_l1)
                    T.dual_copy(v_ub[0:kv_columns, 0:block_kv], v_l1)
                    for key_channel in T.serial(2):
                        T.copy(q_l1[0:block_q, key_channel * kv_columns : (key_channel + 1) * kv_columns], q_l0)
                        T.copy(
                            k_l1[key_channel * kv_columns : (key_channel + 1) * kv_columns, 0:block_kv],
                            k_l0,
                            transpose=True,
                        )
                        T.gemm(q_l0, k_l0, qk_l0, transpose_B=True, clear_accum=key_channel == 0)
                    T.dual_copy(qk_l0, score)
                    with T.SimtVF(threads=128):
                        for r, n in T.Parallel(rows, block_kv):
                            pos = split * split_tokens + tile * block_kv + n
                            valid = (pos >= STARTS[request]) & (pos < ENDS[request])
                            score[r, n] = T.if_then_else(
                                valid, (score[r, n] + MEAN[bh, sid * rows + r]) * softmax_scale, -1.0e30
                            )
                    with T.SimdVF():
                        softmax_full = S.pset(32, "PAT_ALL")
                        softmax_one = S.pset(32, "PAT_VL1")
                        softmax_half = S.pset(16, "PAT_ALL")
                        probability_ptr = S.make_ubuf_ptr(T.access_ptr(prob_half[0, 0], "w", rows + 1, block_kv), dtype)
                        for r in range(rows):
                            x, x1 = S.vld2(score[r, 0], "DINTLV_B32")
                            old_max = S.vld(maximum[r], "BRC_B32")
                            new_max = S.vmax(
                                S.vdupv(S.vcmax(S.vmax(x, x1, softmax_full), softmax_full)), old_max, softmax_full
                            )
                            rescale = S.vexpdif(old_max, new_max, softmax_full)
                            exponential = S.vexpdif(x, new_max, softmax_full)
                            exponential = S.vsel(
                                exponential,
                                S.vdup(T.float32(0), "float32"),
                                S.vcmps(x, T.float32(-1.0e29), softmax_full, op="gt"),
                            )
                            exponential1 = S.vexpdif(x1, new_max, softmax_full)
                            exponential1 = S.vsel(
                                exponential1,
                                S.vdup(T.float32(0), "float32"),
                                S.vcmps(x1, T.float32(-1.0e29), softmax_full, op="gt"),
                            )
                            total = S.vadd(
                                S.vmul(S.vld(denominator[r], "BRC_B32"), rescale, softmax_full),
                                S.vdupv(S.vcadd(S.vadd(exponential, exponential1, softmax_full), softmax_full)),
                                softmax_full,
                            )
                            S.vsts(maximum[r], new_max, softmax_one, "ONEPT_B32")
                            S.vsts(alpha[r], rescale, softmax_one, "ONEPT_B32")
                            S.vsts(denominator[r], total, softmax_one, "ONEPT_B32")
                            half_probability = S.vor(  # codespell:ignore vor
                                T.reinterpret(
                                    S.vcvt(exponential, dtype, softmax_full, sat=False, part=0), "uint16x128"
                                ),
                                T.reinterpret(
                                    S.vcvt(exponential1, dtype, softmax_full, sat=False, part=1), "uint16x128"
                                ),
                                softmax_half,
                            )
                            probability_ptr = S.vsstb(
                                T.reinterpret(half_probability, "float16x128"),
                                probability_ptr,
                                T.int32(((rows + 1) << 16) | 1),
                                softmax_half,
                                update=True,
                            )
                    T.dual_copy(prob_half[0:rows, 0:block_kv], p_l1)
                    T.copy(p_l1, p_l0)
                    for value_channel in T.serial(2):
                        T.copy(v_l1[value_channel * kv_columns : (value_channel + 1) * kv_columns, 0:block_kv], k_l0)
                        T.gemm(p_l0, k_l0, pv_l0, transpose_B=True, clear_accum=True)
                        T.dual_copy(
                            pv_l0, subtotal[0:rows, value_channel * kv_columns : (value_channel + 1) * kv_columns]
                        )
                    with T.SimdVF():
                        accumulate_full = S.pset(32, "PAT_ALL")
                        for r in range(rows):
                            factor = S.vld(alpha[r], "BRC_B32")
                            for d in range(0, dim, 64):
                                update = S.vadd(
                                    S.vmul(S.vld(output[r, d], "NORM"), factor, accumulate_full),
                                    S.vld(subtotal[r, d], "NORM"),
                                    accumulate_full,
                                )
                                S.vsts(output[r, d], update, accumulate_full, "NORM_B32")
                with T.SimtVF(threads=128):
                    for r, d in T.Parallel(rows, dim):
                        output[r, d] = output[r, d] / T.max(denominator[r], 1.0e-30)
                    for r in T.Parallel(rows):
                        final_lse[r] = T.if_then_else(
                            denominator[r] > 0, maximum[r] + T.log(denominator[r]), -T.infinity("float32")
                        )
                T.dual_copy(output, OUT[bh, split, 0:block_q, 0:dim])
                T.dual_copy(final_lse, LSE[(bh * splits + split) * block_q : (bh * splits + split + 1) * block_q])

    return main


class NovaDecodeKernel:
    """Own fixed-shape scratch and a compiled compressed-attention kernel."""

    def __init__(
        self,
        max_tokens,
        heads,
        groups,
        views,
        table,
        device,
        *,
        batch=1,
        split_tokens=1024,
        stages=1,
        num_blocks=32,
        scale=1 / 16,
    ):
        if heads < 1 or not 1 <= groups <= 16 or not 1 <= batch <= 4:
            raise ValueError("NOVA-KV decode requires 1..4 requests and at most 16 query groups")
        if split_tokens % 128 or split_tokens < 128:
            raise ValueError("split_tokens must be a positive multiple of 128")
        if max_tokens < 1 or stages != 1 or not 1 <= num_blocks <= 32:
            raise ValueError("NOVA-KV decode requires a positive context, one pipeline stage and 1..32 blocks")
        key, value, scales = views
        if key.shape != value.shape or key.shape[1:3] != (heads, 64):
            raise ValueError("NOVA-KV decode requires planar D256 KV")
        pages, _, _, page_tokens = key.shape
        if page_tokens % 128 or scales.shape != (pages, heads, 3, page_tokens):
            raise ValueError("NOVA-KV decode requires 128-aligned pages and three scale planes")
        if key.dtype != torch.uint8 or value.dtype != torch.uint8 or scales.dtype != torch.bfloat16:
            raise ValueError("NOVA-KV decode requires uint8 payloads and BF16 scales")
        if (
            key.stride()[1:] != (64 * page_tokens, page_tokens, 1)
            or value.stride() != key.stride()
            or scales.stride()[1:] != (3 * page_tokens, page_tokens, 1)
        ):
            raise ValueError("NOVA-KV decode requires contiguous planes within each padded page")
        if table.ndim != 2 or table.shape[0] != batch or table.shape[1] * page_tokens < max_tokens:
            raise ValueError("NOVA-KV decode block table does not cover the configured context")
        self.batch, self.heads, self.groups = batch, heads, groups
        self.split_tokens = split_tokens
        self.splits = math.ceil(max_tokens / split_tokens)
        self.query = torch.empty((batch * heads, _QUERY_TILE, 256), dtype=torch.float16, device=device)
        self.mean = torch.empty((batch * heads, _QUERY_TILE), device=device)
        self.partial = torch.empty((batch * heads, self.splits, _QUERY_TILE, 256), device=device)
        self.partial_lse = torch.empty((batch * heads, self.splits, _QUERY_TILE), device=device)
        self.output = torch.empty((batch, heads, groups, 256), device=device)
        self.lse = torch.empty((batch, heads, groups), device=device)
        self.kernel = _build_nova_fia(
            batch,
            heads,
            groups,
            key.shape[0],
            key.shape[-1],
            key.stride(0),
            scales.stride(0),
            table.shape[1],
            self.splits,
            split_tokens,
            scale,
            stages,
            num_blocks,
        )
        self.prepare = _build_prepare(batch * heads, groups)
        self.merge = _build_merge(batch * heads, groups, self.splits)

    def __call__(self, query, mean, views, books, table, starts, ends):
        if books.dtype != torch.float16 or not books.is_contiguous():
            raise ValueError("Prepare contiguous FP16 NOVA-KV codebooks before graph capture")
        self.prepare(query.reshape(-1, self.groups, 256), mean.reshape(-1, self.groups), self.query, self.mean)
        key, value, scales = views
        self.kernel(
            self.query,
            self.mean,
            key,
            value,
            scales,
            books.view(-1),
            table,
            starts,
            ends,
            self.partial,
            self.partial_lse.view(-1),
        )
        self.merge(
            self.partial, self.partial_lse, self.output.view(-1, self.groups, 256), self.lse.view(-1, self.groups)
        )
        return self.output, self.lse
