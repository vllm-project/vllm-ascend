# SPDX-License-Identifier: Apache-2.0
"""Standalone performance benchmarks for the TurboQuant custom ops.

Mirrors the data generation of test_turboquant_custom_ops.py (the precision
tests) but measures device time instead of correctness:

  compress : torch.ops._C_ascend.turbo_quant_compress_latent over a numTokens sweep
  store    : the production store path (tq_latent_store.compress_kernel, i.e. the
             Hadamard matmul + the fused op) over the same sweep -- the gap
             between "store" and "compress" is exactly the Hadamard/launch cost
             that in-kernel WHT (see turboquant/turbo_quant_compress_latent.md,
             OPT-1) would remove
  sfa      : torch.ops._C_ascend.turboquant_sparse_flash_attention over realistic
             paged-cache cases (BATCH:Q_TOKENS:CTX:TOPK): a full physical block
             pool with randomized per-request block tables, and per-query topk
             selection from the WHOLE context -- mirroring the on-board 128k
             capture scheme in turboquant/tq4-onboard-msprof/. Every case is
             first validated against the numpy reference (ported from
             test_turboquant_custom_ops.py, same tolerance) before it is timed;
             a failing case aborts the benchmark.

Usage:
  python turboquant_custom_ops_perf.py --ops all
  python turboquant_custom_ops_perf.py --ops compress --tokens 1 1024 65536 --json /tmp/tq.json
  python turboquant_custom_ops_perf.py --ops sfa --sfa-cases 1:4096:130176:2048

Timing methodology: warmup calls, then N timed reps, each rep timing a loop of
`--inner` back-to-back op calls with one torch.npu.Event pair (amortizes event
overhead for the ~5us kernels). Reports min/median per call. min matches the
"minimum kernel Duration(us)" convention used in the op READMEs.
"""

import argparse
import gc
import json
import math
import os
import statistics
import sys
from collections.abc import Callable

# This directory is a package containing a "triton" subpackage, which would
# shadow the real triton dependency during torch._dynamo import when the script
# is run directly (sys.path[0] = script dir). Nothing is imported from the
# script directory itself, so drop it before importing torch.
_here = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _here:
    sys.path.pop(0)

import numpy as np  # noqa: E402
import torch  # noqa: E402

try:
    # Same import order as the precision tests: torch first, then torch_npu.
    import torch_npu
except Exception as err:  # noqa: BLE001 - environment guard (broken triton etc.)
    raise SystemExit(
        f"torch_npu failed to import ({err!r}). This benchmark needs a working "
        "Ascend CANN + torch_npu environment (the nightly CI image has one)."
    ) from err

from vllm_ascend.utils import enable_custom_op  # noqa: E402

enable_custom_op()
torch_npu.npu.config.allow_internal_format = True

TQ_HEAD_DIM = 512
TQ_ROPE_HEAD_DIM = 64
TQ_COMBINE_DIM = TQ_HEAD_DIM + TQ_ROPE_HEAD_DIM  # TND query last dim
TQ_BLOCK_SIZE = 128
TQ_PACKED_BYTES = TQ_HEAD_DIM // 2
TQ_COMPRESS_SLOT_BYTES = 320  # alignUp(256 + 2, 64), the compress op contract
TQ_FUSED_SLOT_BYTES = TQ_PACKED_BYTES + TQ_ROPE_HEAD_DIM * 2 + 2  # 386
TQ_SLOT_ROW_BYTES = 416  # 386 rounded up to the 32B data-block pitch used by SFA
# Per token: fp32 latent in + uint8 slot out (README counts 2048 + 320).
TQ_COMPRESS_BYTES_PER_TOKEN = TQ_HEAD_DIM * 4 + TQ_COMPRESS_SLOT_BYTES

TQ_CENTROIDS = np.array(
    [
        -0.12091285,
        -0.09111122,
        -0.07112455,
        -0.05513602,
        -0.04132067,
        -0.02874970,
        -0.01700489,
        -0.00568677,
        0.00547294,
        0.01680406,
        0.02857605,
        0.04108622,
        0.05492980,
        0.07101817,
        0.09115373,
        0.12037795,
    ],
    dtype=np.float32,
)


def bench(fn: Callable[[], object], warmup: int, active: int, inner: int) -> dict[str, float]:
    """Time `fn` on the NPU. Returns per-call microseconds (min/median/mean)."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    per_call_us: list[float] = []
    for _ in range(active):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        torch.npu.synchronize()
        per_call_us.append(start.elapsed_time(end) * 1000.0 / inner)

    return {
        "min_us": min(per_call_us),
        "median_us": statistics.median(per_call_us),
        "mean_us": statistics.fmean(per_call_us),
    }


def cleanup() -> None:
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def bench_compress(args: argparse.Namespace) -> list[dict[str, object]]:
    """Sweep numTokens against the fused compress op alone (fp32 latent in)."""
    rng = np.random.default_rng(2026)  # same seed as the precision test
    centroids = torch.from_numpy(TQ_CENTROIDS).npu()
    rows: list[dict[str, object]] = []

    for num_tokens in args.tokens:
        # RMSNorm'd-scale latent, as in the precision test.
        latent_np = (rng.standard_normal((num_tokens, TQ_HEAD_DIM)) / math.sqrt(TQ_HEAD_DIM)).astype(np.float32)
        latent = torch.from_numpy(latent_np).npu()

        stats = bench(
            lambda latent=latent: torch.ops._C_ascend.turbo_quant_compress_latent(latent, centroids),
            args.warmup,
            args.active,
            args.inner,
        )
        rows.append(
            {
                "op": "compress",
                "shape": f"N={num_tokens}",
                **stats,
                "Mtok/s": num_tokens / stats["min_us"],
                "GB/s": num_tokens * TQ_COMPRESS_BYTES_PER_TOKEN / stats["min_us"] / 1e3,
            }
        )
        del latent
        cleanup()
    return rows


def bench_store(args: argparse.Namespace) -> list[dict[str, object]]:
    """Sweep the production store path: Hadamard matmul + fused op (bf16 in)."""
    from vllm_ascend.ops import tq_latent_store

    rng = np.random.default_rng(2026)
    rows: list[dict[str, object]] = []

    for num_tokens in args.tokens:
        latent = (
            torch.from_numpy(
                (rng.standard_normal((num_tokens, TQ_HEAD_DIM)) / math.sqrt(TQ_HEAD_DIM)).astype(np.float32)
            )
            .to(torch.bfloat16)
            .npu()
        )

        stats = bench(
            lambda latent=latent: tq_latent_store.compress_kernel(latent, head_dim=TQ_HEAD_DIM),
            args.warmup,
            args.active,
            args.inner,
        )
        rows.append(
            {
                "op": "store(Hadamard+compress)",
                "shape": f"N={num_tokens}",
                **stats,
                "Mtok/s": num_tokens / stats["min_us"],
                # fp32 z round-trip adds 2 x 512 x 4B of GM traffic on top of the op.
                "GB/s": num_tokens * (2 * TQ_HEAD_DIM + TQ_COMPRESS_BYTES_PER_TOKEN) / stats["min_us"] / 1e3,
            }
        )
        del latent
        cleanup()
    return rows


def make_sfa_inputs(
    batch: int,
    q_tokens: int,
    ctx_tokens: int,
    topk: int,
    q_heads: int,
    pool_blocks: int,
) -> dict[str, object]:
    """Realistic sparse-attention case over a paged TurboQuant KV pool.

    Mirrors the on-board 128k capture scheme (turboquant/tq4-onboard-msprof/
    tq4_single_op_msprof_128k_20260826_113546/driver.py):

      - the cache is a full physical block pool (pool_blocks x 128 x 1 x 386 int8,
        tens-to-hundreds of MB), not an array sized to just topk
      - each batch row gets its own randomized logical->physical block_table, so
        selected slots scatter across the pool like real paging fragmentation
      - every query selects its topk tokens from the WHOLE context (ctx_tokens,
        e.g. 130176 ~= 128k), not from [0, topk)
      - packed nibbles are random bytes (random 0-15 codebook indices; zero packing
        would let the dequant gather broadcast and understate cost ~4.6% on prefill),
        rope stays zero (valid bf16 0.0) with a trailing fp16 scale of 1.0
        (kv[..., -1] = 0x3c): keeps softmax finite
    """
    logical_blocks = (ctx_tokens + TQ_BLOCK_SIZE - 1) // TQ_BLOCK_SIZE
    g = torch.Generator().manual_seed(14715)

    kv = torch.zeros((pool_blocks, TQ_BLOCK_SIZE, 1, TQ_FUSED_SLOT_BYTES), dtype=torch.int8)
    # Random nibble content (uniform bytes = independent 0-15 codebook indices per
    # nibble): measured 2026-08-31, zero packing understates the byte-LUT dequant
    # cost by ~4.6% on prefill (uniform gather addresses let the 1KB LUT read
    # broadcast; scattered indices take bank conflicts) while the pre-LUT kernel
    # is content-invariant (<0.1%). Rope stays zero (valid bf16 0.0).
    kv[..., :TQ_PACKED_BYTES] = torch.randint(
        0, 256, (pool_blocks, TQ_BLOCK_SIZE, 1, TQ_PACKED_BYTES), generator=g, dtype=torch.uint8
    ).view(torch.int8)
    kv[..., -1] = 0x3C  # little-endian fp16 1.0 scale
    query = (torch.randn(batch * q_tokens, q_heads, TQ_COMBINE_DIM, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    # Per-batch randomized paging: logical block i of request b lives in a random
    # physical slot (no sharing between requests, as under real occupancy).
    block_table = torch.stack([torch.randperm(pool_blocks, generator=g)[:logical_blocks] for _ in range(batch)]).to(
        torch.int32
    )
    # Per-query scattered selection over the whole context, ascending like the
    # topk output. randint may collide on a position; duplicates match the real
    # addressing pattern (gather) and do not change kernel cost.
    indices = torch.sort(
        torch.randint(0, ctx_tokens, (batch * q_tokens, 1, topk), generator=g, dtype=torch.int32), dim=-1
    ).values

    kv_npu = kv.npu()
    query_npu = query.npu()
    indices_npu = indices.npu()
    block_table_npu = block_table.npu()
    # TND actual_seq_lengths_query is cumulative over requests.
    seq_q = torch.arange(q_tokens, batch * q_tokens + 1, q_tokens, dtype=torch.int32).npu()
    seq_kv = torch.full((batch,), ctx_tokens, dtype=torch.int32).npu()
    torch.npu.synchronize()

    def call() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops._C_ascend.turboquant_sparse_flash_attention(
            query_npu,
            kv_npu,
            kv_npu,
            indices_npu,
            key_dequant_scale=None,
            value_dequant_scale=None,
            block_table=block_table_npu,
            actual_seq_lengths_query=seq_q,
            actual_seq_lengths_kv=seq_kv,
            scale_value=1.0 / math.sqrt(TQ_HEAD_DIM),
            key_quant_mode=3,
            value_quant_mode=3,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=TQ_BLOCK_SIZE,
            rope_head_dim=TQ_ROPE_HEAD_DIM,
            return_softmax_lse=False,
        )

    return {
        "call": call,
        "pool_blocks": pool_blocks,
        "pool_mb": pool_blocks * TQ_BLOCK_SIZE * TQ_FUSED_SLOT_BYTES / 1e6,
        # exposed for sfa_correctness (checked against the same tensors that get timed)
        "kv": kv_npu,
        "query": query_npu,
        "indices": indices_npu,
        "block_table": block_table_npu,
        "batch": batch,
        "q_tokens": q_tokens,
    }


def _sample_rows(batch: int, q_tokens: int, cap: int = 256) -> list[int]:
    """Evenly spaced query rows covering every request (correctness subset)."""
    total = batch * q_tokens
    if total <= cap:
        return list(range(total))
    rows: list[int] = []
    per = max(1, cap // batch)
    for b in range(batch):
        pos = np.unique(np.linspace(0, q_tokens - 1, per).round().astype(int))
        rows.extend(b * q_tokens + int(p) for p in pos)
    return rows


def sfa_correctness(inputs: dict, max_rows: int = 256) -> dict[str, float]:
    """Validate one op call against the numpy reference before timing it.

    Ports test_turboquant_custom_ops.py::_reference_sfa onto the harness content
    (random nibbles, zero rope, fp16 1.0 scale): dequant goes through
    bf16-rounded TQ_CENTROIDS -- exactly what the kernel byte-LUT emits -- the
    rope term drops out (stored rope is zero) and V = unit * scale_j == unit.
    Same tolerance as the e2e test: element error <= 2**-9 * (1 + |expected|),
    >= 99% elements within it, max abs error <= 0.1. When B*Q exceeds
    `max_rows`, rows are subsampled evenly per request (attention rows are
    independent; the subset still covers every request's block table).
    """
    rows = _sample_rows(inputs["batch"], inputs["q_tokens"], max_rows)
    rows_t = torch.tensor(rows, dtype=torch.long)
    q_rows = inputs["query"][rows_t].float().cpu().numpy()  # [R, heads, 576]
    actual, _, _ = inputs["call"]()
    out_rows = actual[rows_t].float().cpu().numpy()  # [R, heads, 512]
    indices_np = inputs["indices"].cpu().numpy()
    bt_np = inputs["block_table"].cpu().numpy()
    # the kernel LUT emits bf16(centroid); mirror that bit-exactly
    cent = torch.from_numpy(TQ_CENTROIDS).to(torch.bfloat16).float().numpy()
    kv_flat = inputs["kv"].view(-1, TQ_FUSED_SLOT_BYTES)
    q_tokens = inputs["q_tokens"]
    scale_value = 1.0 / math.sqrt(TQ_HEAD_DIM)

    matched = 0
    total = 0
    max_err = 0.0
    for r, t in enumerate(rows):
        b = t // q_tokens
        idx = indices_np[t, 0].astype(np.int64)
        phys = bt_np[b, idx // TQ_BLOCK_SIZE].astype(np.int64) * TQ_BLOCK_SIZE + idx % TQ_BLOCK_SIZE
        sel = kv_flat[torch.from_numpy(phys).npu(), :TQ_PACKED_BYTES].cpu().numpy().view(np.uint8)
        lo = (sel & 0x0F).astype(np.int64)
        hi = (sel >> 4).astype(np.int64)
        unit = np.empty((sel.shape[0], TQ_HEAD_DIM), dtype=np.float32)
        unit[:, 0::2] = cent[lo]  # low-nibble-first packing: even dims from the low nibble
        unit[:, 1::2] = cent[hi]
        scores = q_rows[r][:, :TQ_HEAD_DIM].astype(np.float64) @ unit.astype(np.float64).T
        scores *= scale_value  # rope term is zero; scale_j == 1.0 on this content
        scores -= scores.max(axis=1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=1, keepdims=True)
        expected = p @ unit.astype(np.float64)
        error = np.abs(out_rows[r].astype(np.float64) - expected)
        tol = 2.0**-9 * (1.0 + np.abs(expected))
        matched += int((error <= tol).sum())
        total += error.size
        max_err = max(max_err, float(error.max()))
    return {"match_ratio": matched / total, "max_abs_error": max_err, "rows": len(rows)}


def bench_sfa(args: argparse.Namespace) -> list[dict[str, object]]:
    q_heads = args.sfa_heads
    rows: list[dict[str, object]] = []

    for case in args.sfa_cases:
        batch, q_tokens, ctx_tokens, topk = case
        logical_blocks = (ctx_tokens + TQ_BLOCK_SIZE - 1) // TQ_BLOCK_SIZE
        # Real single-request pool from the 128k capture; auto-grow so B requests
        # with disjoint physical blocks still fit.
        pool_blocks = args.sfa_pool_blocks or max(1595, (batch * logical_blocks * 11 + 9) // 10)
        case_inputs = make_sfa_inputs(batch, q_tokens, ctx_tokens, topk, q_heads, pool_blocks)
        call = case_inputs["call"]

        # Correctness gate before timing: each case must match the numpy
        # reference (ported from test_turboquant_custom_ops.py) first.
        check = sfa_correctness(case_inputs, max_rows=args.sfa_check_rows)
        ok = check["match_ratio"] >= 0.99 and check["max_abs_error"] <= 0.1
        print(
            f"[check] B={batch} Q={q_tokens} CTX={ctx_tokens} K={topk}: "
            f"match={check['match_ratio']:.6f} max_abs={check['max_abs_error']:.8f} "
            f"rows={check['rows']} {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            raise SystemExit(
                f"sfa correctness gate failed for B={batch} Q={q_tokens} CTX={ctx_tokens} K={topk}; "
                "not timing an incorrect kernel"
            )

        stats = bench(call, args.warmup, args.active, args.inner)
        # MM1 (T1 x N1 x K x 576) + MM2 (T1 x N1 x K x 512), 2 flops per MAC.
        flops = (
            2 * batch * q_tokens * q_heads * topk * TQ_COMBINE_DIM + 2 * batch * q_tokens * q_heads * topk * TQ_HEAD_DIM
        )
        kv_bytes = batch * q_tokens * topk * TQ_SLOT_ROW_BYTES  # kvHeadNum=1: slots shared across heads
        rows.append(
            {
                "op": "sfa",
                "shape": (
                    f"B={batch} Q={q_tokens} CTX={ctx_tokens} N1={q_heads} K={topk} pool={case_inputs['pool_mb']:.0f}MB"
                ),
                **stats,
                "TFLOP/s": flops / stats["min_us"] / 1e6,
                "KV GB/s": kv_bytes / stats["min_us"] / 1e3,
            }
        )
        del call, case_inputs
        cleanup()
    return rows


def print_rows(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = ["op", "shape", "min_us", "median_us", "mean_us"] + [
        k for k in rows[0] if k not in ("op", "shape", "min_us", "median_us", "mean_us")
    ]
    widths = {k: max(len(k), *(len(f"{r[k]:.3f}" if isinstance(r[k], float) else r[k]) for r in rows)) for k in keys}
    header = "  ".join(k.ljust(widths[k]) for k in keys)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = (
            f"{v:.3f}".ljust(widths[k]) if isinstance(v, float) else str(v).ljust(widths[k])
            for k, v in ((k, row[k]) for k in keys)
        )
        print("  ".join(cells))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--ops",
        nargs="+",
        choices=["compress", "store", "sfa", "all"],
        default=["all"],
        help="which benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=[1, 32, 128, 512, 1024, 2048, 4096, 8192, 16384, 65536],
        help="numTokens sweep for compress/store (default: the op README table)",
    )
    parser.add_argument(
        "--sfa-cases",
        nargs="+",
        metavar="BATCH:Q_TOKENS:CTX:TOPK",
        default=[
            "1:4096:130176:2048",  # the on-board 128k prefill capture (driver.py)
            "1:8192:130176:2048",
            "2:4096:130176:2048",  # multi-request prefill
            "2:8192:130176:2048",
            "4:4096:130176:2048",
            "4:8192:130176:2048",
            "1:1:130176:2048",  # decode @128k ctx
            "4:1:130176:2048",
            "16:1:130176:2048",
            "64:1:130176:2048",
            "1:1:65536:2048",  # decode @128k ctx
            "4:1:65536:2048",
            "16:1:65536:2048",  # 64k ctx decode
            "64:1:65536:2048",
        ],
        help="SFA cases: requests : query tokens per request : context tokens : selected KV per query "
        "(CTX=130176 is the ~128k single-request capture; CTX>=topk required)",
    )
    parser.add_argument(
        "--sfa-pool-blocks",
        type=int,
        default=0,
        help="physical KV pool size in 128-token blocks; 0 = auto (max of the 128k "
        "capture pool 1595 and 1.1x the per-case disjoint paging need, default: 0)",
    )
    parser.add_argument("--sfa-heads", type=int, default=16, help="query head count for SFA (default: 16)")
    parser.add_argument(
        "--sfa-check-rows",
        type=int,
        default=256,
        help="max query rows validated against the reference before timing each sfa case "
        "(0 = full check; default: 256, sampled evenly across requests)",
    )
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations (default: 5, as in the READMEs)")
    parser.add_argument("--active", type=int, default=15, help="timed reps (default: 15)")
    parser.add_argument("--inner", type=int, default=10, help="op calls per timed rep (default: 10)")
    parser.add_argument("--json", type=str, default=None, help="also write results as JSON to this path")
    args = parser.parse_args()

    if "all" in args.ops:
        args.ops = ["compress", "store", "sfa"]
    cases = []
    for c in args.sfa_cases:
        fields = [int(x) for x in c.split(":")]
        if len(fields) != 4:
            raise SystemExit(f"bad --sfa-cases entry {c!r}: expected BATCH:Q_TOKENS:CTX:TOPK")
        batch, q_tokens, ctx_tokens, topk = fields
        if batch < 1 or q_tokens < 1 or topk < 1 or ctx_tokens < topk:
            raise SystemExit(f"bad --sfa-cases entry {c!r}: need B>=1, Q>=1, CTX>=TOPK>=1")
        cases.append((batch, q_tokens, ctx_tokens, topk))
    args.sfa_cases = cases
    return args


def main() -> None:
    args = parse_args()

    if not torch.npu.is_available():
        raise SystemExit("NPU is required: no available device found")

    try:
        device_name = torch.npu.get_device_name(0)
    except Exception:  # noqa: BLE001 - best-effort label only
        device_name = "unknown"
    print(f"# device: {device_name} | warmup={args.warmup} active={args.active} inner={args.inner}")

    rows: list[dict[str, object]] = []
    runners = {
        "compress": bench_compress,
        "store": bench_store,
        "sfa": bench_sfa,
    }
    for op in args.ops:
        print(f"\n== {op} ==")
        op_rows = runners[op](args)
        print_rows(op_rows)
        rows.extend(op_rows)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": device_name, "rows": rows}, f, indent=2)
        print(f"\nresults written to {args.json}")


if __name__ == "__main__":
    main()
