# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small KDA verification kernel for the TileLang Ascend A3 backend.

This benchmark kernel consumes prepared decay-adjusted K/Q rows and recurrence
coefficients. It does not update the checkpoint or perform model projections.
The checkpoint and packed rows are explicitly FP16; accumulation and the
recurrence are FP32. Each of the two Vector subblocks owns 64 value channels.

The explicit C/V scopes and GM handoff follow tile-ai/tilelang-ascend's
``examples/simple_fusion/matmul_add.py``. Scalar ``T.tile.axpy`` and row slices
follow ``examples/mhc_post/example_mhc_post.py``. APIs were checked at commit
``9b5ac505c801ce038e31cc8ab6f1f83e14df1497`` (the ``ascendc_pto`` branch).
"""

from functools import lru_cache

HEAD_DIM = 128
MAX_TOKENS = 8
PACKED_ROWS = 2 * MAX_TOKENS
VECTOR_SUBBLOCKS = 2
VALUE_CHANNELS_PER_SUBBLOCK = HEAD_DIM // VECTOR_SUBBLOCKS


@lru_cache(maxsize=64, typed=True)
def make_kernel(jobs: int, tokens: int, prefetch: bool = True):
    """Compile an A3 kernel specialized for ``jobs`` and 1--8 tokens.

    The returned callable takes ``(packed, checkpoint, coeff, beta, values)``:

    * packed: FP16 [jobs, 16, 128], K in rows 0:tokens, Q in 8:8+tokens.
    * checkpoint: read-only FP16 [jobs, 128, 128], key/value axis order.
    * coeff: FP32 [jobs, 16, 16], historical correction coefficients.
    * beta: FP32 [jobs, 16], of which only the first tokens entries are used.
    * values: FP32 [jobs, 8, 128].

    Returns correction and output, both FP32 [jobs, 8, 128]. Only rows
    0:tokens are defined. Inputs must be contiguous NPU tensors. TileLang
    allocates the outputs and the private [jobs, 16, 128] FP32 workspace.
    Lazy imports keep CPU-only reference/fixture tools independent of CANN.

    ``prefetch`` is a compile-time scheduling choice. True lets Vector load
    coeff/beta/values while Cube produces base; False waits for base before
    those same loads. Both variants perform the same arithmetic and transfers.
    Overlap and any latency benefit require verification on an A3 device.
    """
    if not isinstance(jobs, int) or isinstance(jobs, bool) or jobs < 1:
        raise ValueError(f"jobs must be a positive integer, got {jobs!r}")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or not 1 <= tokens <= MAX_TOKENS:
        raise ValueError(f"tokens must be an integer in [1, {MAX_TOKENS}], got {tokens!r}")
    if not isinstance(prefetch, bool):
        raise ValueError(f"prefetch must be a bool, got {prefetch!r}")

    import tilelang
    import tilelang.language as T

    @tilelang.jit(
        out_idx=[5, 6],
        workspace_idx=[7],
        target="ascendc",
        platform="A3",
    )
    def build():
        @T.prim_func
        def main(
            packed: T.Tensor((jobs, PACKED_ROWS, HEAD_DIM), "float16"),
            checkpoint: T.Tensor((jobs, HEAD_DIM, HEAD_DIM), "float16"),
            coeff: T.Tensor((jobs, PACKED_ROWS, PACKED_ROWS), "float"),
            beta: T.Tensor((jobs, PACKED_ROWS), "float"),
            values: T.Tensor((jobs, MAX_TOKENS, HEAD_DIM), "float"),
            correction: T.Tensor((jobs, MAX_TOKENS, HEAD_DIM), "float"),
            output: T.Tensor((jobs, MAX_TOKENS, HEAD_DIM), "float"),
            base_workspace: T.Tensor((jobs, PACKED_ROWS, HEAD_DIM), "float"),
        ):
            with T.Kernel(jobs, is_npu=True) as (job, subblock):
                packed_l1 = T.alloc_L1((PACKED_ROWS, HEAD_DIM), "float16")
                checkpoint_l1 = T.alloc_L1((HEAD_DIM, HEAD_DIM), "float16")
                base_l0c = T.alloc_L0C((PACKED_ROWS, HEAD_DIM), "float")

                base_ub = T.alloc_ub((PACKED_ROWS, VALUE_CHANNELS_PER_SUBBLOCK), "float")
                coeff_ub = T.alloc_ub((PACKED_ROWS, PACKED_ROWS), "float")
                beta_ub = T.alloc_ub((PACKED_ROWS,), "float")
                values_ub = T.alloc_ub((MAX_TOKENS, VALUE_CHANNELS_PER_SUBBLOCK), "float")
                correction_ub = T.alloc_ub((MAX_TOKENS, VALUE_CHANNELS_PER_SUBBLOCK), "float")
                prediction_ub = T.alloc_ub((VALUE_CHANNELS_PER_SUBBLOCK,), "float")
                output_ub = T.alloc_ub((VALUE_CHANNELS_PER_SUBBLOCK,), "float")

                with T.Scope("C"):
                    T.copy(packed[job, 0:PACKED_ROWS, 0:HEAD_DIM], packed_l1)
                    T.copy(checkpoint[job, 0:HEAD_DIM, 0:HEAD_DIM], checkpoint_l1)
                    T.barrier_all()
                    T.gemm_v0(packed_l1, checkpoint_l1, base_l0c, init=True)
                    T.barrier_all()
                    T.copy(base_l0c, base_workspace[job, 0:PACKED_ROWS, 0:HEAD_DIM])
                    # FIX publishes the completed L0C -> GM transfer to both
                    # associated Vector subblocks. The workspace is never reused
                    # inside this launch, so no reverse handshake is necessary.
                    T.set_cross_flag("FIX", 0)

                with T.Scope("V"):
                    value_start = subblock * VALUE_CHANNELS_PER_SUBBLOCK
                    value_end = value_start + VALUE_CHANNELS_PER_SUBBLOCK
                    # The TileLang parser evaluates Python-bool conditions at
                    # construction time: each variant contains exactly one wait.
                    if not prefetch:
                        T.wait_cross_flag(0)

                    # These inputs are independent of the Cube result. In the
                    # prefetch variant their MTE2 transfers can run concurrently
                    # with Cube work. This local barrier does not wait for Cube.
                    T.copy(beta[job, 0:PACKED_ROWS], beta_ub)
                    for row in T.unroll(PACKED_ROWS):
                        T.copy(coeff[job, row, 0:PACKED_ROWS], coeff_ub[row, :])
                    for row in T.unroll(tokens):
                        T.copy(values[job, row, value_start:value_end], values_ub[row, :])
                    T.barrier_all()

                    if prefetch:
                        T.wait_cross_flag(0)
                    for row in T.unroll(PACKED_ROWS):
                        # Explicit regions preserve GM row strides when each
                        # subblock loads half of a 128-channel row. These loads
                        # must follow the Cube's FIX publication in both variants.
                        T.copy(base_workspace[job, row, value_start:value_end], base_ub[row, :])
                    T.barrier_all()

                    # tokens <= 8: specialize all loop bounds while preserving
                    # the sequential dependence through correction_ub.
                    for t in T.unroll(tokens):
                        T.tile.mul(prediction_ub, base_ub[t, :], 1.0)
                        T.barrier_all()
                        for u in T.unroll(tokens):
                            if u < t:
                                T.tile.axpy(prediction_ub, correction_ub[u, :], coeff_ub[t, u])
                                T.barrier_all()
                        T.tile.sub(correction_ub[t, :], values_ub[t, :], prediction_ub)
                        T.barrier_all()
                        T.tile.mul(correction_ub[t, :], correction_ub[t, :], beta_ub[t])
                        T.barrier_all()

                        T.tile.mul(output_ub, base_ub[MAX_TOKENS + t, :], 1.0)
                        T.barrier_all()
                        for u in T.unroll(tokens):
                            if u <= t:
                                T.tile.axpy(output_ub, correction_ub[u, :], coeff_ub[MAX_TOKENS + t, u])
                                T.barrier_all()
                        T.copy(correction_ub[t, :], correction[job, t, value_start:value_end])
                        T.copy(output_ub, output[job, t, value_start:value_end])
                        # Complete stores before recycling output_ub on the next
                        # token. The two Vector subblocks write disjoint columns.
                        T.barrier_all()

        return main

    return build()
