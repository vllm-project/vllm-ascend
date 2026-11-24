# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Micro-benchmark: immediate commit vs deferred (defer + flush) RecoverSSM.

    python tests/ut/ops/bench_kda_deferred_recoverssm.py --preset small
    python tests/ut/ops/bench_kda_deferred_recoverssm.py --preset realistic
    python tests/ut/ops/bench_kda_deferred_recoverssm.py --preset realistic --verify

Caveat: this times the kernels in isolation.  The deferred path does *more*
total kernel work per step (an extra defer kernel and a flush), so it can look
slower here.  Its real benefit is moving the fold off the post-sampling
critical path onto the next verify.  The useful numbers are:

  * ``commit`` (immediate)  = critical path per step today
  * ``defer``  (deferred)   = critical path per step with deferral
  * ``flush``  (deferred)   = work moved to the next verify

``--verify`` additionally times the stock deferred path (per group one
``flush``, then one verify kernel per layer) against the fused deferred path
(``compact_conv`` + verify with the commit folded into the kernel as an
in-kernel replay): the saving is fused vs stock, both against the same
deferred commit.

``--sweep`` runs that comparison over a range of batch sizes and reports
where the fused path wins — with ``--heads`` set to the model's *local* heads
(``num_heads / tp``), the ``heads*batch`` crossover is exactly the
``VLLM_ASCEND_RECOVERSSM_FUSED_ITEMS`` threshold the deferred orchestration
uses.

``--blocks`` sizes the paged checkpoint pool, and ``--records`` gives the
correction/key-gate records their own pool instead of packing them into the
state pool. Set ``--blocks 492 --records 64`` to reproduce an upstream vLLM
profile whose checkpoint is ``[492,H,128,128]`` and records are
``[64,H,4,128]`` (records indexed by request slot, not state block).

``--layer-groups`` splits the layers into per-commit-context groups matching
the scheduler's mamba cache groups: each group flushes/commits once per step,
so serving pays ``len(groups)`` plan/compact/commit kernel launches instead of
one. GLM-5.3-Flash's 34 KDA layers group as ``12,11,11``; the default keeps a
single context over all layers.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from test_kda_deferred_recoverssm import DEVICE, DTYPE, _make_context
from vllm_ascend.ops.triton.kda.recoverssm import kda_recoverssm_verify, recoverssm_replays

PRESETS = {
    # Tiny: launch-overhead dominated, useful as a sanity check.
    "small": dict(layers=2, heads=2, key_dim=8, value_dim=8, spec_q=3, batch=2, steps=100),
    # GLM5-Next KDA shape: head_dim=128, 32 heads, 16 KDA layers, batch 32.
    # Keep `blocks` ~= batch: the kernels touch one state block per request, so
    # the checkpoint allocation stays small while the per-step work is realistic.
    "realistic": dict(layers=16, heads=32, key_dim=128, value_dim=128, spec_q=4, batch=32, steps=50),
}


def _sync() -> None:
    torch.npu.synchronize()


def _time(fn, steps: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    _sync()
    return (time.perf_counter() - start) * 1e3 / steps  # ms/step


def _accepted_lengths(
    batch: int,
    spec_q: int,
    accept_rate: float | None,
    accept_len: int | None,
) -> torch.Tensor:
    """Replay length per request: fixed, geometric(accept_rate), or all accepted.

    A truncated geometric with per-token rate ``r`` models speculative decoding:
    P(n) = (1 - r) * r**n for n < spec_q - 1, capped at spec_q - 1.
    """
    max_len = spec_q - 1
    if accept_len is not None:
        return torch.full(
            (batch,), min(max(accept_len, 0), max_len),
            dtype=torch.int32, device=DEVICE,
        )
    if accept_rate is not None:
        rate = min(max(accept_rate, 1e-6), 1 - 1e-6)
        u = torch.rand(batch, generator=torch.Generator().manual_seed(17)).clamp_min(1e-7)
        n = torch.floor(torch.log(u) / math.log(rate)).to(torch.int64).clamp(0, max_len)
        return n.to(torch.int32).to(DEVICE)
    return torch.full((batch,), max_len, dtype=torch.int32, device=DEVICE)


def _verify_section(
    layers_count: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    spec_q: int,
    batch: int,
    steps: int,
    warmup: int,
    accept_rate: float | None = None,
    accept_len: int | None = None,
    blocks_override: int | None = None,
    records_override: int | None = None,
    pipeline_stages: int = 2,
    layer_groups: tuple[int, ...] | None = None,
) -> tuple[float, float]:
    """Time the stock deferred path (``flush + verify``) against the fused
    deferred path (``compact_conv`` + verify with in-kernel replay).

    ``layer_groups`` models the scheduler's mamba cache groups: serving splits
    GLM-5.3-Flash's 34 KDA layers into 12/11/11 groups, each with its own
    commit context that flushes/compacts once per step (``commit_step`` loops
    over the group metadata). None keeps one context over all layers.
    """
    group_sizes = layer_groups or (layers_count,)
    blocks = batch + 1 if blocks_override is None else blocks_override
    total = batch * spec_q
    g = torch.Generator(device=DEVICE).manual_seed(11)

    def randn(*shape, dtype=DTYPE):
        return torch.randn(*shape, device=DEVICE, generator=g, dtype=dtype)

    q = randn(1, total, heads, key_dim)
    k = randn(1, total, heads, key_dim)
    v = randn(1, total, heads, value_dim)
    raw_g = randn(1, total, heads, key_dim)
    raw_beta = randn(1, total, heads)
    state_indices = torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE)
    record_rows = state_indices.clone()
    pending = _accepted_lengths(batch, spec_q, accept_rate, accept_len)
    query_start_loc = torch.arange(
        0, (batch + 1) * spec_q, spec_q, dtype=torch.int32, device=DEVICE
    )

    def make_ctxs():
        contexts, all_layers = [], []
        for gi, size in enumerate(group_sizes):
            ctx, group_layers = _make_context(
                num_layers=size, heads=heads, key_dim=key_dim,
                value_dim=value_dim, spec_q=spec_q, blocks=blocks,
                max_reqs=batch, records=records_override, seed=300 + gi,
            )
            contexts.append(ctx)
            all_layers.extend(group_layers)
        return contexts, all_layers

    ctxs_stock, layers_stock = make_ctxs()
    ctxs_fused, layers_fused = make_ctxs()
    layer_group_index = [gi for gi, size in enumerate(group_sizes) for _ in range(size)]

    def run(contexts, layers, fused: bool) -> None:
        plans = []
        for ctx in contexts:
            if fused:
                ctx.compact_conv(state_indices, record_rows, pending)
                plans.append(
                    (
                        ctx.commit_lens,
                        ctx.final_state_indices,
                        ctx.boundary_state_indices,
                        ctx.boundary_recovery_lens,
                    )
                )
            else:
                ctx.flush(state_indices, record_rows, pending)
                plans.append(None)
        for gi, layer in zip(layer_group_index, layers):
            replay = plans[gi] or (None, None, None, None)
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
                record_indices=record_rows,
                replay_commit_lens=replay[0],
                replay_final_state_indices=replay[1],
                replay_boundary_state_indices=replay[2],
                replay_boundary_recovery_lens=replay[3],
                records_pool_size=records_override,
                pipeline_stages=pipeline_stages,
            )

    t_stock = _time(lambda: run(ctxs_stock, layers_stock, False), steps, warmup)
    t_fused = _time(lambda: run(ctxs_fused, layers_fused, True), steps, warmup)
    if accept_len is not None:
        accept = f"accept_len={accept_len}"
    elif accept_rate is not None:
        accept = f"accept_rate={accept_rate}"
    else:
        accept = "all accepted"
    groups_str = ""
    if len(group_sizes) > 1:
        groups_str = f", groups={'+'.join(str(s) for s in group_sizes)}"
    print(f"verify section ({layers_count} layers{groups_str}, "
          f"batch={batch}, spec_q={spec_q}, {accept}):")
    print(f"  flush + verify        : {t_stock:8.3f} ms/step")
    print(f"  compact + fused verify: {t_fused:8.3f} ms/step")
    print(f"  saving                : {t_stock - t_fused:8.3f} ms/step "
          f"({(1 - t_fused / t_stock) * 100:5.1f}%)")
    print()
    # Free both contexts before the next sweep point; large batches allocate
    # two full checkpoint sets per call.
    del ctxs_stock, ctxs_fused, layers_stock, layers_fused
    torch.npu.empty_cache()
    return t_stock, t_fused


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="small")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--key-dim", type=int)
    parser.add_argument("--value-dim", type=int)
    parser.add_argument("--spec-q", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--blocks", type=int, default=None,
                        help="state-pool block count (default batch+1); set to "
                             "reproduce a profiled checkpoint shape like 492 blocks")
    parser.add_argument("--records", type=int, default=None,
                        help="independent record-pool size (default: share the "
                             "state pool); set to 64 to reproduce upstream "
                             "vLLM's max_num_reqs-sized records like [64,H,4,128]")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--verify", action="store_true",
                        help="also time stock flush+verify vs fused verify "
                             "(deferred commit folded in-kernel)")
    parser.add_argument("--sweep", type=str, default=None,
                        help="comma-separated batch sizes: time stock vs fused for each "
                             "and report where fused starts to win (the critical point; "
                             "set --heads to the model's local heads = num_heads / tp)")
    parser.add_argument("--accept-rate", type=float, default=None,
                        help="per-token acceptance rate: sample each request's replay "
                             "length from a truncated geometric (default: all accepted)")
    parser.add_argument("--accept-len", type=int, default=None,
                        help="fixed replay length for every request (overrides --accept-rate)")
    parser.add_argument("--pipeline-stages", type=int, default=2,
                        help="software-pipeline depth for the verify kernel's "
                             "per-token loop")
    parser.add_argument("--layer-groups", type=str, default=None,
                        help="comma-separated commit-context sizes splitting the layers, "
                             "matching the scheduler's mamba cache groups (GLM-5.3-Flash "
                             "serving: '12,11,11' for its 34 KDA layers); default: one "
                             "context over all layers")
    args = parser.parse_args()

    cfg = dict(PRESETS[args.preset])
    for name in ("layers", "heads", "key_dim", "value_dim", "spec_q", "batch", "steps"):
        value = getattr(args, name)
        if value is not None:
            cfg[name] = value

    layers, heads = cfg["layers"], cfg["heads"]
    key_dim, value_dim = cfg["key_dim"], cfg["value_dim"]
    spec_q, batch, steps = cfg["spec_q"], cfg["batch"], cfg["steps"]
    if args.layer_groups is not None:
        group_sizes = tuple(int(x) for x in args.layer_groups.split(","))
        if sum(group_sizes) != layers or any(n <= 0 for n in group_sizes):
            raise SystemExit(
                f"--layer-groups ({args.layer_groups}) must be positive sizes "
                f"summing to --layers ({layers})"
            )
    else:
        group_sizes = None
    blocks = batch + 1 if args.blocks is None else args.blocks
    if blocks <= batch:
        raise SystemExit(
            f"--blocks ({blocks}) must exceed the batch ({batch}): every request "
            "needs its own state block in the pool"
        )
    records = args.records
    if records is not None and records < batch:
        raise SystemExit(
            f"--records ({records}) must be at least the batch ({batch}): "
            "each request needs its own record slot"
        )
    max_reqs = batch

    state_indices = torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE)
    record_rows = state_indices.clone()
    query_start_loc = torch.arange(
        0, (batch + 1) * spec_q, spec_q, dtype=torch.int32, device=DEVICE
    )
    num_accepted = _accepted_lengths(batch, spec_q, args.accept_rate, args.accept_len)

    ckpt_mb = blocks * heads * value_dim * key_dim * 4 / 2**20
    pool_str = f"blocks={blocks}"
    if records is not None:
        pool_str += f" records={records}"
    if group_sizes is not None:
        pool_str += f" layer_groups={'+'.join(str(n) for n in group_sizes)}"
    print(f"config: layers={layers} batch={batch} heads={heads} key_dim={key_dim} "
          f"value_dim={value_dim} spec_q={spec_q} {pool_str}")
    if args.accept_len is not None:
        accept = f"accept_len={args.accept_len}"
    elif args.accept_rate is not None:
        accept = f"accept_rate={args.accept_rate} (truncated geometric)"
    else:
        accept = "all accepted (worst case)"
    print(f"steps={steps} warmup={args.warmup} acceptance={accept}  "
          f"checkpoint~{ckpt_mb:.0f} MB/layer (~{ckpt_mb * layers / 1024:.2f} GB total)\n")

    if args.sweep:
        batches = [int(x) for x in args.sweep.split(",")]
        rows = []
        for b in batches:
            t_stock, t_fused = _verify_section(
                layers, heads, key_dim, value_dim, spec_q, b, steps, args.warmup,
                args.accept_rate, args.accept_len, args.blocks, records,
                args.pipeline_stages, layer_groups=group_sizes,
            )
            rows.append((b, heads * b, t_stock, t_fused))
        print("sweep summary (fused wins when saving > 0)")
        print(f"{'batch':>6} {'heads*batch':>12} {'flush+verify':>14} "
              f"{'fused':>10} {'saving':>10} {'saving%':>8} {'code picks':>12}")
        for b, hb, t_stock, t_fused in rows:
            choice = "fused" if recoverssm_replays(heads, b) else "flush"
            print(f"{b:6d} {hb:12d} {t_stock:14.3f} {t_fused:10.3f} "
                  f"{t_stock - t_fused:10.3f} {(1 - t_fused / t_stock) * 100:7.1f}% "
                  f"{choice:>12}")
        wins = [b for b, _, t_stock, t_fused in rows if t_fused < t_stock]
        if wins:
            print(f"\nfused wins from batch={min(wins)} "
                  f"(heads*batch={heads * min(wins)})")
            print(f"set VLLM_ASCEND_RECOVERSSM_FUSED_ITEMS={heads * min(wins)} "
                  "to fuse at and above this point")
        else:
            print("\nfused never wins in this sweep: raise batch or lower layers")
        return

    def make_ctxs(seed: int):
        return [
            _make_context(
                num_layers=n, heads=heads, key_dim=key_dim, value_dim=value_dim,
                spec_q=spec_q, blocks=blocks, max_reqs=max_reqs,
                records=records, seed=seed + gi,
            )[0]
            for gi, n in enumerate(group_sizes or (layers,))
        ]

    ctxs_imm = make_ctxs(100)

    def immediate_step() -> None:
        for ctx in ctxs_imm:
            ctx.commit(
                num_accepted_tokens=num_accepted,
                state_indices=state_indices,
                query_start_loc=query_start_loc,
            )

    imm_ms = _time(immediate_step, steps, args.warmup)

    ctxs_def = make_ctxs(100)
    pending = torch.zeros(max_reqs, dtype=torch.int32, device=DEVICE)

    def deferred_defer_step() -> None:
        for ctx in ctxs_def:
            ctx.commit(
                num_accepted_tokens=num_accepted,
                state_indices=state_indices,
                query_start_loc=query_start_loc,
                record_rows=record_rows,
                pending=pending,
                eager=None,
            )

    def deferred_flush_step() -> None:
        for ctx in ctxs_def:
            ctx.flush(state_indices, record_rows, pending)

    defer_ms = _time(deferred_defer_step, steps, args.warmup)
    flush_ms = _time(deferred_flush_step, steps, args.warmup)

    ctxs_def2 = make_ctxs(100)
    pending2 = torch.zeros(max_reqs, dtype=torch.int32, device=DEVICE)

    def deferred_full_step() -> None:
        for ctx in ctxs_def2:
            ctx.flush(state_indices, record_rows, pending2)
            ctx.commit(
                num_accepted_tokens=num_accepted,
                state_indices=state_indices,
                query_start_loc=query_start_loc,
                record_rows=record_rows,
                pending=pending2,
                eager=None,
            )

    def_ms = _time(deferred_full_step, steps, args.warmup)

    print(f"immediate commit        : {imm_ms:8.3f} ms/step")
    print(f"deferred defer only     : {defer_ms:8.3f} ms/step  (critical path)")
    print(f"deferred flush only     : {flush_ms:8.3f} ms/step  (moved to next verify)")
    print(f"deferred defer + flush  : {def_ms:8.3f} ms/step  (total kernel work)")
    print()
    print(f"critical-path saving     : {imm_ms - defer_ms:8.3f} ms/step "
          f"({(1 - defer_ms / imm_ms) * 100:5.1f}%)")
    print(f"total-work overhead      : {def_ms - imm_ms:8.3f} ms/step "
          f"({(def_ms / imm_ms - 1) * 100:5.1f}%)")
    print()

    # The immediate/deferred section above held three full checkpoint sets
    # (blocks x layers, e.g. ~12 GB each at 492 blocks x 34 layers). Free them
    # before the verify section allocates its two contexts (stock + fused), or
    # a large --blocks run over-commits the device.
    del ctxs_imm, ctxs_def, ctxs_def2
    torch.npu.empty_cache()

    if args.verify:
        _verify_section(
            layers, heads, key_dim, value_dim, spec_q, batch, steps, args.warmup,
            args.accept_rate, args.accept_len, args.blocks, records,
            args.pipeline_stages, layer_groups=group_sizes,
        )


if __name__ == "__main__":
    main()
