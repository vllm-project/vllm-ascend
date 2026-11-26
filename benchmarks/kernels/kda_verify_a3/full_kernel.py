# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cube state projection, Vector recurrence and gated RMSNorm on Ascend A3."""

from functools import lru_cache

HEAD_DIM = 128
TOKEN_ALIGNMENT = 8


@lru_cache(maxsize=16)
def make_middle_kernel(tokens: int, eps: float, *, cv_slots: int = 0, core_count: int = 24):
    """Use one job per block or schedule jobs on a fixed set of C/V groups.

    A job is one request and head. The optional ring holds one or two
    checkpoint projection tiles per group.
    """
    if cv_slots not in (0, 1, 2) or core_count < 1:
        raise ValueError("Expected zero, one or two C/V slots and a positive core count")

    import tilelang
    import tilelang.language as T

    dim = HEAD_DIM
    padded = (tokens + TOKEN_ALIGNMENT - 1) // TOKEN_ALIGNMENT * TOKEN_ALIGNMENT
    rows = 2 * padded
    jobs = T.symbolic("jobs")
    persistent = cv_slots > 0
    blocks = core_count if persistent else jobs
    workspace_tiles = core_count * cv_slots if persistent else jobs
    slot_count = max(cv_slots, 1)
    first_ready_flag, second_ready_flag = 0, 1
    first_free_flag, second_free_flag = 2, 3

    @tilelang.jit(out_idx=[9, 10], workspace_idx=[11], target="ascendc", platform="A3")
    def build():
        @T.prim_func
        def main(
            kq: T.Tensor((jobs, rows, dim), "bfloat16"),
            kq_residual: T.Tensor((jobs, rows, dim), "bfloat16"),
            state: T.Tensor((jobs, dim, dim), "bfloat16"),
            state_residual: T.Tensor((jobs, dim, dim), "bfloat16"),
            coeff: T.Tensor((jobs, rows, padded), "float32"),
            beta: T.Tensor((jobs, padded), "float32"),
            values: T.Tensor((jobs, tokens, dim), "float32"),
            gate: T.Tensor((jobs, tokens, dim), "bfloat16"),
            norm_weight: T.Tensor((dim,), "bfloat16"),
            correction: T.Tensor((jobs, tokens, dim), "float32"),
            normalized: T.Tensor((jobs, tokens, dim), "bfloat16"),
            projection: T.Tensor((workspace_tiles, rows, dim), "float32"),
        ):
            with T.Kernel(blocks, is_npu=True) as (block, subblock):
                kq_l1 = T.alloc_L1((rows, dim), "bfloat16")
                state_l1 = T.alloc_L1((dim, dim), "bfloat16")
                base_l0c = T.alloc_L0C((rows, dim), "float32")

                coeff_ub = T.alloc_ub((rows, padded), "float32")
                beta_ub = T.alloc_ub((padded,), "float32")
                value_ub = T.alloc_ub((tokens, dim), "float32")
                gate_bf16 = T.alloc_ub((tokens, dim), "bfloat16")
                gate_ub = T.alloc_ub((tokens, dim), "float32")
                norm_bf16 = T.alloc_ub((dim,), "bfloat16")
                norm_ub = T.alloc_ub((dim,), "float32")
                base_ub = T.alloc_ub((rows, dim), "float32")
                corr_ub = T.alloc_ub((tokens, dim), "float32")
                out_ub = T.alloc_ub((dim,), "float32")
                out_bf16 = T.alloc_ub((dim,), "bfloat16")
                square_ub = T.alloc_ub((1, dim), "float32")
                inverse_rms = T.alloc_ub((1, 1), "float32")

                iterations = T.ceildiv(jobs - block, core_count) if persistent else 1

                with T.Scope("C"):
                    for step in T.serial(iterations):
                        job = block + step * core_count if persistent else block
                        slot = step % slot_count
                        workspace_tile = block * slot_count + slot if persistent else block
                        if persistent and step >= slot_count:
                            if cv_slots == 2 and slot == 1:
                                T.wait_cross_flag(second_free_flag)
                            else:
                                T.wait_cross_flag(first_free_flag)
                        T.copy(kq[job, :, :], kq_l1)
                        T.copy(state[job, :, :], state_l1)
                        T.barrier_all()
                        # The persistent checkpoint has [value, key] axis order.
                        T.gemm_v0(kq_l1, state_l1, base_l0c, transpose_B=True, init=True)
                        T.barrier_all()
                        # Keep the FP32 checkpoint contract on BF16 Cube using
                        # the leading product and its two first-order corrections.
                        T.copy(kq_residual[job, :, :], kq_l1)
                        T.barrier_all()
                        T.gemm_v0(kq_l1, state_l1, base_l0c, transpose_B=True)
                        T.barrier_all()
                        T.copy(kq[job, :, :], kq_l1)
                        T.copy(state_residual[job, :, :], state_l1)
                        T.barrier_all()
                        T.gemm_v0(kq_l1, state_l1, base_l0c, transpose_B=True)
                        T.barrier_all()
                        T.copy(base_l0c, projection[workspace_tile, :, :])
                        if cv_slots == 2 and slot == 1:
                            T.set_cross_flag("FIX", second_ready_flag)
                        else:
                            T.set_cross_flag("FIX", first_ready_flag)
                        if persistent:
                            # Protect the L0C source until the FIX store completes.
                            T.barrier_all()

                with T.Scope("V"):
                    for step in T.serial(iterations):
                        job = block + step * core_count if persistent else block
                        slot = step % slot_count
                        workspace_tile = block * slot_count + slot if persistent else block
                        # A whole head fits into one UB. Give it to one subblock so
                        # RMSNorm needs no reduction between the two subblocks.
                        if subblock == 0:
                            T.copy(coeff[job, :, :], coeff_ub)
                            T.copy(beta[job, :], beta_ub)
                            T.copy(values[job, :, :], value_ub)
                            T.copy(gate[job, :, :], gate_bf16)
                            T.copy(norm_weight, norm_bf16)
                            T.barrier_all()
                            T.tile.cast(norm_ub, norm_bf16, "CAST_NONE", dim)
                            T.tile.cast(gate_ub, gate_bf16, "CAST_NONE", tokens * dim)
                            T.barrier_all()
                            T.tile.mul(gate_ub, gate_ub, -1.0)
                            T.barrier_all()
                            T.tile.exp(gate_ub, gate_ub)
                            T.barrier_all()
                            T.tile.add(gate_ub, gate_ub, 1.0)
                            T.barrier_all()

                        if cv_slots == 2 and slot == 1:
                            T.wait_cross_flag(second_ready_flag)
                        else:
                            T.wait_cross_flag(first_ready_flag)
                        if subblock == 0:
                            T.copy(projection[workspace_tile, :, :], base_ub)
                            T.barrier_all()

                            for t in T.serial(tokens):
                                T.tile.mul(out_ub, base_ub[t, :], 1.0)
                                T.barrier_all()
                                for u in T.serial(t):
                                    T.tile.axpy(out_ub, corr_ub[u, :], coeff_ub[t, u])
                                    T.barrier_all()
                                T.tile.sub(corr_ub[t, :], value_ub[t, :], out_ub)
                                T.barrier_all()
                                T.tile.mul(corr_ub[t, :], corr_ub[t, :], beta_ub[t])
                                T.barrier_all()

                                T.tile.mul(out_ub, base_ub[padded + t, :], 1.0)
                                T.barrier_all()
                                for u in T.serial(t + 1):
                                    T.tile.axpy(out_ub, corr_ub[u, :], coeff_ub[padded + t, u])
                                    T.barrier_all()
                                T.copy(corr_ub[t, :], correction[job, t, :])
                                # Preserve the native core's BF16 boundary before RMSNorm.
                                T.tile.cast(out_bf16, out_ub, "CAST_RINT", dim)
                                T.barrier_all()
                                T.tile.cast(out_ub, out_bf16, "CAST_NONE", dim)
                                T.barrier_all()
                                T.tile.mul(square_ub[0, :], out_ub, out_ub)
                                T.barrier_all()
                                T.reduce_sum(square_ub, inverse_rms, dim=-1)
                                T.barrier_all()
                                T.tile.mul(inverse_rms, inverse_rms, 1.0 / dim)
                                T.barrier_all()
                                T.tile.add(inverse_rms, inverse_rms, eps)
                                T.barrier_all()
                                T.tile.rsqrt(inverse_rms, inverse_rms)
                                T.barrier_all()
                                T.tile.mul(out_ub, out_ub, inverse_rms[0, 0])
                                T.barrier_all()
                                T.tile.mul(out_ub, out_ub, norm_ub)
                                T.barrier_all()
                                T.tile.div(out_ub, out_ub, gate_ub[t, :])
                                T.barrier_all()
                                T.tile.cast(out_bf16, out_ub, "CAST_RINT", dim)
                                T.barrier_all()
                                T.copy(out_bf16, normalized[job, t, :])
                                T.barrier_all()

                        # Mode 2 requires both AIVs to signal completion, even
                        # though only subblock 0 consumes the projection data.
                        # Omit the return when Cube will not reuse this slot.
                        if persistent and step + slot_count < iterations:
                            if cv_slots == 2 and slot == 1:
                                T.set_cross_flag("V", second_free_flag)
                            else:
                                T.set_cross_flag("V", first_free_flag)

        return main

    return build()
