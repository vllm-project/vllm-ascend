# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 Vector fusion for the FP32 KDA recurrence and gated BF16 RMSNorm."""

from functools import lru_cache


def _repeat_mul(
    destination, left, right, offsets, strides, rows=128, *, operation="Mul"
):
    """Two FP32 repeat instructions cover a 128-column UB matrix."""
    # The pinned AscendC codegen emits these UB variable names verbatim.
    # T._src_code access pointers keep their lifetimes visible to TileLang.
    statements = ["AscendC::SetMaskNorm();"]
    for shift in (0, 64):
        operands = []
        for tensor, start, block_stride in zip(
            (destination, left, right), offsets, strides[:3], strict=True
        ):
            offset = start + (shift if block_stride else 0)
            operands.append(f"{tensor}[{offset}]" if offset else tensor)
        statements.append(
            f"AscendC::{operation}({', '.join(operands)}, static_cast<uint64_t>(64), "
            f"static_cast<uint8_t>({rows}), AscendC::BinaryRepeatParams{strides});"
        )
    return "\n".join(statements)


def _project(vector, offset):
    return _repeat_mul(
        "matrix_work", "current", vector, (0, 0, offset), (1, 1, 1, 16, 16, 0)
    )


# Strides use AscendC's 32-byte hardware blocks. FP32 mask 64 covers
# one half of the 128-key row. A vector repeat stride of zero reuses it.
_DECAY_STATE = _repeat_mul(
    "current", "decay", "current", (0, 0, 0), (1, 1, 1, 16, 0, 16)
)
_PROJECT_KEY = _project("qk", 128)
_PROJECT_QUERY = _project("qk", 0)
_PROJECT_WEIGHTED_KEY = _project("weighted", 128)
_PROJECT_WEIGHTED_QUERY = _project("weighted", 0)
# A source block stride of zero reuses the eight BRCB lanes within a row.
# The source repeat stride of one advances to the next packed correction.
_STATE_UPDATE = _repeat_mul(
    "current",
    "qk",
    "packed_correction",
    (0, 128, 0),
    (1, 1, 0, 16, 0, 1),
    operation="MulAddDst",
)


@lru_cache(maxsize=32)
def make_a5_middle_kernel(tokens: int, eps: float, lower_bound: float):
    """Compile the H=64, D=128 verification path with a symbolic batch size."""
    import tilelang
    import tilelang.language as T

    if not 1 <= tokens <= 16 or lower_bound is None:
        raise ValueError(
            "the fused A3 middle requires 1–16 candidates and a bounded gate"
        )

    heads = 64
    dim = 128
    width = heads * dim
    projected_width = 3 * width + heads + 2 * dim
    batch = T.symbolic("batch")
    norm_scale_blocks = (tokens + 7) // 8
    norm_by_rms = _repeat_mul(
        "current",
        "current",
        "packed_correction",
        (0, 0, 0),
        (1, 1, 0, 16, 16, 1),
        rows=tokens,
    )
    norm_by_weight = _repeat_mul(
        "current", "current", "weight", (0, 0, 0), (1, 1, 1, 16, 16, 0), rows=tokens
    )

    @tilelang.jit(out_idx=[8, 9, 10], target="ascendc", platform="A3")
    def build():
        @T.prim_func
        def main(
            qkv: T.Tensor((batch * tokens, 3 * width), "bfloat16"),
            projected: T.Tensor((batch * tokens, projected_width), "bfloat16"),
            gate_f: T.Tensor((batch * tokens, width), "bfloat16"),
            gate_g: T.Tensor((batch * tokens, width), "bfloat16"),
            state: T.Tensor((batch, heads, dim, dim), "float32"),
            a_log: T.Tensor((heads,), "float32"),
            dt_bias: T.Tensor((heads, dim), "float32"),
            norm_w: T.Tensor((dim,), "bfloat16"),
            normalized: T.Tensor((batch * tokens, width), "bfloat16"),
            correction: T.Tensor((batch, heads, tokens, dim), "float32"),
            kg: T.Tensor((batch, heads, tokens, 2 * dim), "bfloat16"),
        ):
            with T.Kernel(T.ceildiv(batch * heads, 2), is_npu=True) as (
                block,
                subblock,
            ):
                current = T.alloc_ub((dim, dim), "float32")
                matrix_work = T.alloc_ub((dim, dim), "float32")
                packed_correction = T.alloc_ub((dim, 8), "float32")

                q_by_token = T.alloc_ub((tokens, dim), "bfloat16")
                k_by_token = T.alloc_ub((tokens, dim), "bfloat16")
                v_by_token = T.alloc_ub((tokens, dim), "bfloat16")
                f_by_token = T.alloc_ub((tokens, dim), "bfloat16")
                g_by_token = T.alloc_ub((tokens, dim), "bfloat16")
                beta_by_token = T.alloc_ub((tokens, heads), "bfloat16")
                weight_bf16 = T.alloc_ub((dim,), "bfloat16")
                result_bf16 = T.alloc_ub((dim,), "bfloat16")
                normalized_by_token = T.alloc_ub((tokens, dim), "bfloat16")

                qk = T.alloc_ub((2, dim), "float32")
                v = T.alloc_ub((dim,), "float32")
                decay = T.alloc_ub((dim,), "float32")
                output_gate = T.alloc_ub((dim,), "float32")
                beta = T.alloc_ub((heads,), "float32")
                log_a = T.alloc_ub((heads,), "float32")
                bias = T.alloc_ub((dim,), "float32")
                weight = T.alloc_ub((dim,), "float32")
                corr_by_token = T.alloc_ub((tokens, dim), "float32")
                out = T.alloc_ub((dim,), "float32")
                ones = T.alloc_ub((dim,), "float32")
                vector_square = T.alloc_ub((2, dim), "float32")
                qk_norm = T.alloc_ub((2,), "float32")
                rms = T.alloc_ub((1,), "float32")
                rms_by_token = T.alloc_ub((norm_scale_blocks * 8,), "float32")
                key_query = T.alloc_ub((1,), "float32")
                weighted = T.alloc_ub((2, dim), "float32")

                with T.Scope("V"):
                    # Independent heads occupy the two AIV subblocks of each block.
                    job = 2 * block + subblock
                    request = job // heads
                    head = job % heads
                    start = head * dim
                    T.copy(state[request, head, :, :], current)
                    T.copy(a_log, log_a)
                    T.copy(dt_bias[head, :], bias)
                    T.copy(norm_w, weight_bf16)
                    row_start = request * tokens
                    row_end = row_start + tokens
                    # The six copies each issue one multi-row MTE2 transfer.
                    # For T=16 the BF16 staging buffers occupy 22 KiB.
                    T.copy(qkv[row_start:row_end, start : start + dim], q_by_token)
                    T.copy(
                        qkv[row_start:row_end, width + start : width + start + dim],
                        k_by_token,
                    )
                    T.copy(
                        qkv[
                            row_start:row_end,
                            2 * width + start : 2 * width + start + dim,
                        ],
                        v_by_token,
                    )
                    T.copy(gate_f[row_start:row_end, start : start + dim], f_by_token)
                    T.copy(gate_g[row_start:row_end, start : start + dim], g_by_token)
                    T.copy(
                        projected[row_start:row_end, 3 * width : 3 * width + heads],
                        beta_by_token,
                    )
                    T.tile.fill(ones, 1.0)
                    T.barrier_all()
                    T.tile.exp(log_a, log_a)
                    T.tile.cast(weight, weight_bf16, "CAST_NONE", dim)

                    for token in T.serial(tokens):
                        T.tile.cast(qk[0, :], q_by_token[token, :], "CAST_NONE", dim)
                        T.tile.cast(qk[1, :], k_by_token[token, :], "CAST_NONE", dim)
                        T.tile.cast(v, v_by_token[token, :], "CAST_NONE", dim)
                        T.tile.cast(decay, f_by_token[token, :], "CAST_NONE", dim)
                        if tokens == 1:
                            T.tile.cast(
                                output_gate, g_by_token[token, :], "CAST_NONE", dim
                            )
                        T.tile.cast(beta, beta_by_token[token, :], "CAST_NONE", heads)
                        T.pipe_barrier("V")

                        T.tile.mul(vector_square, qk, qk)
                        T.tile.add(decay, decay, bias)
                        T.tile.mul(beta, beta, -1.0)
                        if tokens == 1:
                            T.tile.mul(output_gate, output_gate, -1.0)
                        T.pipe_barrier("V")
                        T.reduce_sum(vector_square, qk_norm, dim=-1)
                        T.tile.mul(decay, decay, -log_a[head])
                        T.tile.exp(beta, beta)
                        if tokens == 1:
                            T.tile.exp(output_gate, output_gate)
                        T.pipe_barrier("V")
                        T.tile.exp(decay, decay)
                        T.tile.sqrt(qk_norm, qk_norm)
                        T.tile.add(beta, beta, 1.0)
                        if tokens == 1:
                            T.tile.add(output_gate, output_gate, 1.0)
                        T.pipe_barrier("V")
                        T.tile.add(decay, decay, 1.0)
                        # A3's Reciprocal and Rsqrt instructions are approximations.
                        T.tile.div(beta, ones[:heads], beta)
                        if tokens == 1:
                            T.tile.div(output_gate, ones, output_gate)
                        T.tile.div(qk_norm, ones[:2], qk_norm)
                        T.barrier_all()
                        T.tile.div(decay, ones, decay)
                        # Avoid 0 * inf for an all-zero convolved query or key.
                        T.tile.mul(
                            qk[0, :],
                            qk[0, :],
                            T.if_then_else(
                                qk_norm[0] < float("inf"), qk_norm[0] * dim**-0.5, 1.0
                            ),
                        )
                        T.tile.mul(
                            qk[1, :],
                            qk[1, :],
                            T.if_then_else(qk_norm[1] < float("inf"), qk_norm[1], 1.0),
                        )
                        T.pipe_barrier("V")
                        T.tile.mul(decay, decay, lower_bound)
                        T.pipe_barrier("V")
                        T.tile.exp(decay, decay)
                        T.pipe_barrier("V")

                        if token + 1 < tokens:
                            T._src_code(
                                _DECAY_STATE,
                                current.access_ptr("rw"),
                                decay.access_ptr("r"),
                            )
                            T.pipe_barrier("V")
                            T._src_code(
                                _PROJECT_KEY,
                                matrix_work.access_ptr("w"),
                                current.access_ptr("r"),
                                qk.access_ptr("r"),
                            )
                        else:
                            # The terminal token projects the incoming state
                            # with two decayed vectors and a scalar K·Q.
                            T.tile.mul(vector_square[0, :], qk[0, :], qk[1, :])
                            T.tile.mul(weighted[0, :], qk[0, :], decay)
                            T.tile.mul(weighted[1, :], qk[1, :], decay)
                            T.pipe_barrier("V")
                            T.reduce_sum(vector_square[0, :], key_query)
                            T._src_code(
                                _PROJECT_WEIGHTED_KEY,
                                matrix_work.access_ptr("w"),
                                current.access_ptr("r"),
                                weighted.access_ptr("r"),
                            )
                        T.pipe_barrier("V")
                        T.reduce_sum(matrix_work, corr_by_token[token, :], dim=-1)
                        T.pipe_barrier("V")
                        T.tile.sub(corr_by_token[token, :], v, corr_by_token[token, :])
                        T.pipe_barrier("V")
                        T.tile.mul(
                            corr_by_token[token, :], corr_by_token[token, :], beta[head]
                        )
                        T.pipe_barrier("V")

                        if token + 1 < tokens:
                            T.tile.brcb_experiment(
                                packed_correction,
                                corr_by_token[token, :],
                                dim // 8,
                                1,
                                8,
                            )
                            T.pipe_barrier("V")
                            T._src_code(
                                _STATE_UPDATE,
                                current.access_ptr("rw"),
                                qk.access_ptr("r"),
                                packed_correction.access_ptr("r"),
                            )
                            T.pipe_barrier("V")
                            T._src_code(
                                _PROJECT_QUERY,
                                matrix_work.access_ptr("w"),
                                current.access_ptr("r"),
                                qk.access_ptr("r"),
                            )
                        else:
                            T._src_code(
                                _PROJECT_WEIGHTED_QUERY,
                                matrix_work.access_ptr("w"),
                                current.access_ptr("r"),
                                weighted.access_ptr("r"),
                            )
                        T.pipe_barrier("V")
                        T.reduce_sum(matrix_work, out, dim=-1)
                        T.pipe_barrier("V")
                        if token + 1 == tokens:
                            T.tile.axpy(out, corr_by_token[token, :], key_query[0])
                            T.pipe_barrier("V")

                        # Match the native core's BF16 boundary before RMSNorm.
                        if tokens == 1:
                            T.tile.cast(result_bf16, out, "CAST_RINT", dim)
                            T.pipe_barrier("V")
                            T.tile.cast(out, result_bf16, "CAST_NONE", dim)
                            T.pipe_barrier("V")
                            T.tile.mul(vector_square[0, :], out, out)
                            T.pipe_barrier("V")
                            T.reduce_sum(vector_square[0, :], rms)
                            T.pipe_barrier("V")
                            T.tile.mul(rms, rms, 1.0 / dim)
                            T.pipe_barrier("V")
                            T.tile.add(rms, rms, eps)
                            T.pipe_barrier("V")
                            T.tile.sqrt(rms, rms)
                            T.pipe_barrier("V")
                            T.tile.div(rms, ones[:1], rms)
                            T.barrier_all()
                            T.tile.mul(out, out, rms[0])
                            T.pipe_barrier("V")
                            T.tile.mul(out, out, weight)
                            T.pipe_barrier("V")
                            T.tile.mul(out, out, output_gate)
                            T.pipe_barrier("V")
                            T.tile.cast(
                                normalized_by_token[token, :], out, "CAST_RINT", dim
                            )
                        else:
                            T.tile.cast(
                                normalized_by_token[token, :], out, "CAST_RINT", dim
                            )

                    if tokens > 1:
                        # Neither recurrence matrix is read after the last token.
                        # Reuse their rows for the rounded outputs, squares,
                        # sigmoid gates and the gate division's numerator.
                        T.pipe_barrier("V")
                        T.tile.cast(
                            current[:tokens, :],
                            normalized_by_token,
                            "CAST_NONE",
                            tokens * dim,
                        )
                        T.tile.cast(
                            matrix_work[tokens : 2 * tokens, :],
                            g_by_token,
                            "CAST_NONE",
                            tokens * dim,
                        )
                        T.tile.fill(matrix_work[2 * tokens : 3 * tokens, :], 1.0)
                        T.tile.fill(rms_by_token, 1.0)
                        T.pipe_barrier("V")
                        T.tile.mul(
                            matrix_work[:tokens, :],
                            current[:tokens, :],
                            current[:tokens, :],
                        )
                        T.tile.mul(
                            matrix_work[tokens : 2 * tokens, :],
                            matrix_work[tokens : 2 * tokens, :],
                            -1.0,
                        )
                        T.pipe_barrier("V")
                        T.reduce_sum(
                            matrix_work[:tokens, :], rms_by_token[:tokens], dim=-1
                        )
                        T.tile.exp(
                            matrix_work[tokens : 2 * tokens, :],
                            matrix_work[tokens : 2 * tokens, :],
                        )
                        T.pipe_barrier("V")
                        T.tile.mul(
                            rms_by_token[:tokens], rms_by_token[:tokens], 1.0 / dim
                        )
                        T.tile.add(
                            matrix_work[tokens : 2 * tokens, :],
                            matrix_work[tokens : 2 * tokens, :],
                            1.0,
                        )
                        T.pipe_barrier("V")
                        T.tile.add(rms_by_token[:tokens], rms_by_token[:tokens], eps)
                        T.tile.div(
                            matrix_work[tokens : 2 * tokens, :],
                            matrix_work[2 * tokens : 3 * tokens, :],
                            matrix_work[tokens : 2 * tokens, :],
                        )
                        T.pipe_barrier("V")
                        T.tile.sqrt(rms_by_token[:tokens], rms_by_token[:tokens])
                        T.pipe_barrier("V")
                        T.tile.div(
                            rms_by_token[:tokens], ones[:tokens], rms_by_token[:tokens]
                        )
                        T.pipe_barrier("V")
                        T.tile.brcb_experiment(
                            packed_correction, rms_by_token, norm_scale_blocks, 1, 8
                        )
                        T.pipe_barrier("V")
                        T._src_code(
                            norm_by_rms,
                            current.access_ptr("rw"),
                            packed_correction.access_ptr("r"),
                        )
                        T.pipe_barrier("V")
                        T._src_code(
                            norm_by_weight,
                            current.access_ptr("rw"),
                            weight.access_ptr("r"),
                        )
                        T.pipe_barrier("V")
                        T.tile.mul(
                            current[:tokens, :],
                            current[:tokens, :],
                            matrix_work[tokens : 2 * tokens, :],
                        )
                        T.pipe_barrier("V")
                        T.tile.cast(
                            normalized_by_token,
                            current[:tokens, :],
                            "CAST_RINT",
                            tokens * dim,
                        )

                    # Batch the correction, key/gate records, and normalized
                    # output stores after the ordered recurrence.
                    T.barrier_all()
                    T.copy(k_by_token, kg[request, head, :, :dim])
                    T.copy(f_by_token, kg[request, head, :, dim:])
                    T.copy(corr_by_token, correction[request, head, :, :])
                    T.copy(
                        normalized_by_token,
                        normalized[row_start:row_end, start : start + dim],
                    )
                    T.barrier_all()

        return main

    return build()
