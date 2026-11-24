# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Operator-level A/B: stock (flush/commit + verify) vs fused (in-kernel replay verify).

Sweeps power-of-two batches 1..256, 5 timed steps each (averaged), 12 KDA
layers, GLM-5.3-Flash shapes (heads=12, dim=128, spec_q=4, blocks=492). Prints
one line per batch with the per-batch average times:

    batch  stock(flush+verify)  fused(verify)  saving%

A positive ``saving%`` means the fused path wins.

    python tests/ut/ops/bench_kda_verify_fused.py
"""

from __future__ import annotations

import math
import time

import torch

from test_kda_deferred_recoverssm import DEVICE, DTYPE, _make_context
from vllm_ascend.ops.triton.kda.recoverssm import kda_recoverssm_verify

LAYERS = 12
HEADS = 12
KEY_DIM = 128
VALUE_DIM = 128
SPEC_Q = 4  # num_speculative_tokens + 1
BLOCKS = 492
STEPS = 5
WARMUP = 5
ACCEPT_RATE = 0.8
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def _accepted_lengths(batch: int) -> torch.Tensor:
    """Truncated-geometric accepted count per request (capped at spec_q - 1)."""
    max_len = SPEC_Q - 1
    rate = min(max(ACCEPT_RATE, 1e-6), 1 - 1e-6)
    u = torch.rand(
        batch, generator=torch.Generator(device=DEVICE).manual_seed(17)
    ).clamp_min(1e-7)
    n = torch.floor(torch.log(u) / math.log(rate)).to(torch.int64).clamp(0, max_len)
    return n.to(torch.int32).to(DEVICE)


def _time(fn, steps: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1e3 / steps


def main() -> None:
    print(f"{'batch':>5} {'stock(flush+verify)':>21} {'fused(verify)':>15} {'saving%':>8}")
    for batch in BATCHES:
        total = batch * SPEC_Q
        g = torch.Generator(device=DEVICE).manual_seed(11)

        def randn(*shape, dtype=DTYPE):
            return torch.randn(*shape, device=DEVICE, generator=g, dtype=dtype)

        q = randn(1, total, HEADS, KEY_DIM)
        k = randn(1, total, HEADS, KEY_DIM)
        v = randn(1, total, HEADS, VALUE_DIM)
        raw_g = randn(1, total, HEADS, KEY_DIM)
        raw_beta = randn(1, total, HEADS)
        out = torch.empty_like(v)

        state_indices = torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE)
        record_rows = torch.arange(0, batch, dtype=torch.int32, device=DEVICE)
        pending = _accepted_lengths(batch)
        query_start_loc = torch.arange(
            0, (batch + 1) * SPEC_Q, SPEC_Q, dtype=torch.int32, device=DEVICE
        )

        ctx_stock, layers_stock = _make_context(
            LAYERS, HEADS, KEY_DIM, VALUE_DIM, SPEC_Q, BLOCKS,
            batch, records=batch, seed=0,
        )
        ctx_fused, layers_fused = _make_context(
            LAYERS, HEADS, KEY_DIM, VALUE_DIM, SPEC_Q, BLOCKS,
            batch, records=batch, seed=100,
        )

        def run_stock() -> None:
            ctx_stock.flush(state_indices, record_rows, pending)
            for layer in layers_stock:
                kda_recoverssm_verify(
                    q, k, v, raw_g, raw_beta,
                    layer.A_log.reshape(-1),
                    layer.dt_bias,
                    layer.gate_lower_bound,
                    layer.kv_cache[1],
                    layer.kv_cache[2],
                    layer.kv_cache[3],
                    query_start_loc,
                    state_indices,
                    SPEC_Q,
                    out=out,
                    record_indices=record_rows,
                    records_pool_size=batch,
                )

        def run_fused() -> None:
            ctx_fused.compact_conv(state_indices, record_rows, pending)
            for layer in layers_fused:
                kda_recoverssm_verify(
                    q, k, v, raw_g, raw_beta,
                    layer.A_log.reshape(-1),
                    layer.dt_bias,
                    layer.gate_lower_bound,
                    layer.kv_cache[1],
                    layer.kv_cache[2],
                    layer.kv_cache[3],
                    query_start_loc,
                    state_indices,
                    SPEC_Q,
                    out=out,
                    record_indices=record_rows,
                    replay_commit_lens=ctx_fused.commit_lens,
                    replay_final_state_indices=ctx_fused.final_state_indices,
                    replay_boundary_state_indices=ctx_fused.boundary_state_indices,
                    replay_boundary_recovery_lens=ctx_fused.boundary_recovery_lens,
                    records_pool_size=batch,
                )

        t_stock = _time(run_stock, STEPS, WARMUP)
        t_fused = _time(run_fused, STEPS, WARMUP)
        saving = (t_stock - t_fused) / t_stock * 100
        print(f"{batch:5d} {t_stock:21.3f} {t_fused:15.3f} {saving:7.1f}%")

        del ctx_stock, ctx_fused, layers_stock, layers_fused
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()