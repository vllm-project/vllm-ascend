# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Operator-level A/B: stock (flush/commit + verify) vs fused (in-kernel replay verify).

Runs a single batch (5 timed steps, averaged), GLM-5.3-Flash KDA shapes.
Prints one line with the per-step average times:

    batch  stock(flush+verify)  fused(verify)  saving%

A positive ``saving%`` means the fused path wins.

    python tests/ut/ops/bench_kda_verify_fused.py --batch 23 --layers 12 --heads 12 --spec-q 4 --blocks 492
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from test_kda_deferred_recoverssm import DEVICE, DTYPE, _make_context
from vllm_ascend.ops.triton.kda.recoverssm import kda_recoverssm_verify

KEY_DIM = 128
VALUE_DIM = 128
STEPS = 5
WARMUP = 5
ACCEPT_RATE = 0.8


def _accepted_lengths(batch: int, spec_q: int) -> torch.Tensor:
    """Truncated-geometric accepted count per request (capped at spec_q - 1)."""
    max_len = spec_q - 1
    rate = min(max(ACCEPT_RATE, 1e-6), 1 - 1e-6)
    u = torch.rand(batch, generator=torch.Generator().manual_seed(17)).clamp_min(1e-7)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=23,
                        help="number of requests in flight (default 23)")
    parser.add_argument("--layers", type=int, default=12,
                        help="number of KDA layers (default 12)")
    parser.add_argument("--heads", type=int, default=12,
                        help="local heads per layer (default 12)")
    parser.add_argument("--spec-q", type=int, default=4,
                        help="spec query len = num_speculative_tokens + 1 (default 4)")
    parser.add_argument("--blocks", type=int, default=492,
                        help="checkpoint block count (default 492, GLM-5.3-Flash profile)")
    args = parser.parse_args()
    batch = args.batch
    layers = args.layers
    heads = args.heads
    spec_q = args.spec_q
    blocks = args.blocks
    if blocks <= batch:
        parser.error(f"--blocks ({blocks}) must exceed the batch ({batch})")

    total = batch * spec_q
    g = torch.Generator(device=DEVICE).manual_seed(11)

    def randn(*shape, dtype=DTYPE):
        return torch.randn(*shape, device=DEVICE, generator=g, dtype=dtype)

    q = randn(1, total, heads, KEY_DIM)
    k = randn(1, total, heads, KEY_DIM)
    v = randn(1, total, heads, VALUE_DIM)
    raw_g = randn(1, total, heads, KEY_DIM)
    raw_beta = randn(1, total, heads)
    out = torch.empty_like(v)

    state_indices = torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE)
    record_rows = torch.arange(0, batch, dtype=torch.int32, device=DEVICE)
    pending = _accepted_lengths(batch, spec_q)
    query_start_loc = torch.arange(
        0, (batch + 1) * spec_q, spec_q, dtype=torch.int32, device=DEVICE
    )

    ctx_stock, layers_stock = _make_context(
        layers, heads, KEY_DIM, VALUE_DIM, spec_q, blocks,
        batch, records=batch, seed=0,
    )
    ctx_fused, layers_fused = _make_context(
        layers, heads, KEY_DIM, VALUE_DIM, spec_q, blocks,
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
                spec_q,
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
                spec_q,
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
    print(f"{'batch':>5} {'stock(flush+verify)':>21} {'fused(verify)':>15} {'saving%':>8}")
    print(f"{batch:5d} {t_stock:21.3f} {t_fused:15.3f} {saving:7.1f}%")


if __name__ == "__main__":
    main()
