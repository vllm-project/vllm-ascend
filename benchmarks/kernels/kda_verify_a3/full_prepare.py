# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Vector packing for the short A3 KDA verification prefix."""

from functools import lru_cache

from full_kernel import HEAD_DIM, TOKEN_ALIGNMENT


@lru_cache(maxsize=16)
def _make_prepare_kernel(tokens: int, dim: int):
    import tilelang
    import tilelang.language as T

    padded = (tokens + TOKEN_ALIGNMENT - 1) // TOKEN_ALIGNMENT * TOKEN_ALIGNMENT
    row_count = 2 * padded
    jobs = T.symbolic("jobs")
    subblocks_per_block = 2

    @tilelang.jit(out_idx=[3, 4, 5, 6], target="ascendc", platform="A3")
    def build():
        @T.prim_func
        def main(
            decay: T.Tensor((jobs, tokens, dim), "float32"),
            key: T.Tensor((jobs, tokens, dim), "float32"),
            query: T.Tensor((jobs, tokens, dim), "float32"),
            rows_hi: T.Tensor((jobs, row_count, dim), "bfloat16"),
            rows_lo: T.Tensor((jobs, row_count, dim), "bfloat16"),
            rows: T.Tensor((jobs, row_count, dim), "float32"),
            inverse_rows: T.Tensor((jobs, padded, dim), "float32"),
        ):
            with T.Kernel(T.ceildiv(jobs, subblocks_per_block), is_npu=True) as (block, subblock):
                prefix_ub = T.alloc_ub((tokens, dim), "float32")
                key_ub = T.alloc_ub((tokens, dim), "float32")
                query_ub = T.alloc_ub((tokens, dim), "float32")
                rows_ub = T.alloc_ub((row_count, dim), "float32")
                inverse_ub = T.alloc_ub((padded, dim), "float32")
                hi_ub = T.alloc_ub((row_count, dim), "bfloat16")
                lo_ub = T.alloc_ub((row_count, dim), "bfloat16")
                rounded_ub = T.alloc_ub((row_count, dim), "float32")

                with T.Scope("V"):
                    job = block * subblocks_per_block + subblock
                    if job < jobs:
                        T.copy(decay[job, :, :], prefix_ub)
                        T.copy(key[job, :, :], key_ub)
                        T.copy(query[job, :, :], query_ub)
                        T.tile.fill(rows_ub, 0.0)
                        T.tile.fill(inverse_ub, 0.0)
                        T.barrier_all()

                        # Advance over tokens in order in FP32. A Vector op
                        # updates all independent key channels together.
                        for t in T.serial(1, tokens):
                            T.tile.mul(prefix_ub[t, :], prefix_ub[t - 1, :], prefix_ub[t, :])
                            T.pipe_barrier("V")

                        T.tile.mul(rows_ub[0:tokens, :], prefix_ub, key_ub)
                        T.tile.mul(rows_ub[padded : padded + tokens, :], prefix_ub, query_ub)
                        T.tile.div(inverse_ub[0:tokens, :], key_ub, prefix_ub)
                        T.pipe_barrier("V")

                        T.tile.cast(hi_ub, rows_ub, "CAST_RINT", row_count * dim)
                        T.pipe_barrier("V")
                        T.tile.cast(rounded_ub, hi_ub, "CAST_NONE", row_count * dim)
                        T.pipe_barrier("V")
                        T.tile.sub(rounded_ub, rows_ub, rounded_ub)
                        T.pipe_barrier("V")
                        T.tile.cast(lo_ub, rounded_ub, "CAST_RINT", row_count * dim)
                        T.barrier_all()

                        T.copy(hi_ub, rows_hi[job, :, :])
                        T.copy(lo_ub, rows_lo[job, :, :])
                        T.copy(rows_ub, rows[job, :, :])
                        T.copy(inverse_ub, inverse_rows[job, :, :])

        return main

    return build()


def compile_full_prepare(jobs: int, tokens: int, dim: int):
    """Compile a callable taking contiguous FP32 ``decay``, ``k`` and ``q``.

    Inputs are ``[jobs, tokens, dim]``. The callable returns BF16 high rows,
    BF16 residual rows, FP32 rows and FP32 inverse-term rows, in that order.
    Key and query rows occupy separate zero-padded panels, each aligned to
    ``TOKEN_ALIGNMENT`` rows. Inverse-term rows occupy one such panel.
    The compiled program shares its symbolic job dimension across batch sizes.
    """
    if jobs < 1:
        raise ValueError("the A3 prefix pack requires at least one job")
    if dim != HEAD_DIM or not 1 <= tokens <= 16:
        raise ValueError("the A3 prefix pack supports D=128 and 1–16 candidates")
    return _make_prepare_kernel(tokens, dim)
