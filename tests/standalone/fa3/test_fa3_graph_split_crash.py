# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C9: reproduce the production MTE crash — split-K kernel + LONG replay.
#
# The production crash log names the faulting kernel:
#   SplitFuse::FAInfer<bf16, bf16, float, MaskType=1, inputLayout=1, LseMode=1>
#   -> "MTE accesses an invalid GM address" (SMMU page fault).
# LseMode=1 is the SPLIT-LSE path: the KV dimension is split into kvSegUpper
# segments and partial LSE/output land in the split workspace.  C6 ran this same
# split kernel at batch=256/maxk=32768 WITHOUT crashing because its replay used
# cache_seqlens=128 (1 segment).  Production replays with REAL request lengths up
# to max_model_len (up to 256 segments), which is the combination C6/C7 never hit.
#
# This experiment fixes maxk (baked split count) and sweeps the REPLAY length so
# the split kernel actually walks up to `ceil(replay_len/128)` segments, exactly
# like production.  Blocks are shared (num_blocks = ceil(maxk/128)) so K/V stays
# small (~67 MB) and the block table is always valid — a crash here is therefore
# in the split workspace / tiling, not the block table.
#
# Read:
#   crash appears as replay_len grows  -> split-kernel workspace/tiling is the
#       root cause; we get the max segments FA3 decode can survive (informs the
#       max_seqlen_k bound for the "bounded over-cover" fix).
#   no crash even at replay_len=maxk   -> not reproducible standalone; the fault
#       is production-pipeline-specific (graph capture / pointer staleness).
#
# Usage:
#   python test_fa3_graph_split_crash.py                # sweep replay_len @ batch=256/maxk=32768
#   python test_fa3_graph_split_crash.py --batch 256 --maxk 32768 --replay 4096  # single cell

import argparse
import subprocess
import sys
from importlib import util as importlib_util

import torch

HEAD_SIZE = 128
NUM_HEADS = 32
NUM_KV_HEADS = 8
BLOCK_SIZE = 128
DTYPE = torch.bfloat16
SCALE = 1.0 / (HEAD_SIZE ** 0.5)

REPLAY_LENS = [128, 1024, 4096, 16384, 32768]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def load_fa3():
    for mod_name in ("flash_attn_npu_3", "flash_attn_npu_3"):
        if importlib_util.find_spec(mod_name) is not None:
            mod = __import__(
                mod_name,
                fromlist=["flash_attn_with_kvcache", "get_scheduler_metadata"],
            )
            return mod.flash_attn_with_kvcache, mod.get_scheduler_metadata, mod_name
    return None, None, None


def run_cell(batch: int, maxk: int, replay_len: int) -> int:
    fa3_kvcache, get_scheduler_metadata, mod_name = load_fa3()
    if fa3_kvcache is None:
        print(f"[b{batch} k{maxk} r{replay_len}] SKIP (FA3 not installed)", flush=True)
        return 3

    blocks_per_seq = _ceil_div(maxk, BLOCK_SIZE)
    segs = _ceil_div(replay_len, BLOCK_SIZE)
    print(
        f"[b{batch} k{maxk} r{replay_len}] blocks/seq={blocks_per_seq} "
        f"replay_segments={segs}", flush=True,
    )

    torch.npu.empty_cache()

    q = torch.randn(batch, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    # shared K/V: only blocks_per_seq blocks; every seq points at the same blocks
    k = torch.randn(
        blocks_per_seq, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE
    ).npu()
    v = torch.randn_like(k)

    cu_q = torch.arange(batch + 1, dtype=torch.int32).npu()
    shared_row = list(range(blocks_per_seq))
    page_table = torch.tensor([shared_row] * batch, dtype=torch.int32).npu()
    plan_buf = torch.full((batch,), maxk, dtype=torch.int32).npu()
    cache_seqlens_buf = torch.full((batch,), maxk, dtype=torch.int32).npu()

    meta = get_scheduler_metadata(
        batch_size=batch,
        max_seqlen_q=1,
        max_seqlen_k=maxk,
        num_heads_q=NUM_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        headdim=HEAD_SIZE,
        cache_seqlens=plan_buf,
        qkv_dtype=DTYPE,
        cu_seqlens_q=cu_q,
        page_size=BLOCK_SIZE,
        causal=True,
    )

    torch.npu.synchronize()
    base = torch.npu.memory_allocated()

    def run():
        return fa3_kvcache(
            q, k, v,
            cache_seqlens=cache_seqlens_buf,
            page_table=page_table,
            cu_seqlens_q=cu_q,
            max_seqlen_q=1,
            softmax_scale=SCALE,
            causal=True,
            window_size=(-1, -1),
            scheduler_metadata=meta,
        )

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    torch.npu.synchronize()

    after = torch.npu.memory_allocated()
    delta = after - base

    # LONG replay: force the split kernel to walk up to `segs` segments.
    cache_seqlens_buf.copy_(
        torch.full((batch,), replay_len, dtype=torch.int32).npu()
    )
    graph.replay()
    torch.npu.synchronize()
    _ = captured.clone()

    print(
        f"[b{batch} k{maxk} r{replay_len}] OK  ws_meas={delta / 1e6:.1f} MB",
        flush=True,
    )
    return 0


def sweep(batch: int, maxk: int):
    print("=" * 72)
    print(f"C9 split-kernel crash sweep  batch={batch}  maxk={maxk}")
    print("=" * 72)
    for replay_len in REPLAY_LENS:
        cmd = [
            sys.executable, __file__,
            "--batch", str(batch), "--maxk", str(maxk), "--replay", str(replay_len),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            print(f"  replay={replay_len:>6} -> TIMEOUT")
            continue
        result = "OK" if r.returncode == 0 else f"CRASH({r.returncode})"
        print(f"  replay={replay_len:>6} ({_ceil_div(replay_len, BLOCK_SIZE):>3} seg) -> {result}")
        if r.returncode != 0:
            tail = "\n".join(r.stderr.splitlines()[-8:])
            if tail.strip():
                print(f"         stderr tail: {tail}")
    print("=" * 72)
    print("If a CRASH leaves the device in a bad state, reset it before re-running.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--maxk", type=int, default=32768)
    ap.add_argument("--replay", type=int, default=None)
    args = ap.parse_args()

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")
    if args.replay is not None:
        sys.exit(run_cell(args.batch, args.maxk, args.replay))
    sweep(args.batch, args.maxk)


if __name__ == "__main__":
    main()
