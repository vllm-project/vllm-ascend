# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from ..shared.sparse_attention import DIM, PACKED_WORDS, SCALE_GROUP, dequantize_group
from . import DENSE_PREFIX_TOKENS, PAGE_SIZE


def prepare_dense_cache(requests, pages, columns, tokens, score_rows_per_query):
    half_page = PAGE_SIZE // 2
    half_pages = DENSE_PREFIX_TOKENS // half_page
    stride_words = (PACKED_WORDS + 7) // 8 * 8

    @T.prim_func
    def main(
        Cache: T.Tensor((pages, PAGE_SIZE, PACKED_WORDS), T.int32),
        Table: T.Tensor((requests, columns), T.int32),
        Length: T.Tensor((requests,), T.int32),
        Ends: T.Tensor((requests,), T.int64),
        PreparedKV: T.Tensor(
            (tokens * score_rows_per_query, DIM),
            T.bfloat16,
        ),
    ):
        with T.Kernel(requests * half_pages) as bx:
            owner = bx // half_pages
            half_index = bx % half_pages
            first_token = half_index * half_page
            packed = T.alloc_shared((half_page, stride_words), T.int32)
            dequantized = T.alloc_shared((half_page, DIM), T.bfloat16)
            fp8 = T.view(packed, (half_page, stride_words * 4), dtype=T.float8_e4m3fn)
            scales = T.view(packed, (half_page, stride_words), dtype=T.float32)
            start_query = T.if_then_else(owner == 0, 0, T.cast(Ends[T.max(owner - 1, 0)], T.int32))
            fits = (
                Length[owner] < DENSE_PREFIX_TOKENS
                and T.ceildiv(Length[owner], PAGE_SIZE) * PAGE_SIZE
                <= (T.cast(Ends[owner], T.int32) - start_query) * score_rows_per_query
            )
            if fits and (half_index // 2) * PAGE_SIZE < Length[owner]:
                page = T.cast(T.max(Table[owner, half_index // 2], 0), T.int64)
                start = (half_index % 2) * half_page
                valid_rows = T.min(T.max(Length[owner] - first_token, 0), half_page)
                if valid_rows > 0:
                    T.copy(Cache[page, start : start + valid_rows, :], packed[:valid_rows, :PACKED_WORDS])
                if valid_rows < half_page:
                    with T.SimtVF(threads=256):
                        for i, j in T.Parallel(half_page, PACKED_WORDS):
                            if i >= valid_rows:
                                packed[i, j] = 0
                with T.SimdVF():
                    float_mask = S.pset(32, "PAT_ALL")
                    bf16_mask = S.pset(16, "PAT_ALL")
                    byte_mask = S.pset(8, "PAT_ALL")
                    for channel in range(DIM // SCALE_GROUP):
                        offset = channel * SCALE_GROUP
                        for row in range(half_page):
                            value = dequantize_group(scales, fp8, row, channel, float_mask, byte_mask, bf16_mask)
                            S.vsts(dequantized[row, offset], value, bf16_mask, "NORM_B16")
                row = start_query * score_rows_per_query + first_token
                T.copy(dequantized, PreparedKV[row : row + half_page, :])

    return main
