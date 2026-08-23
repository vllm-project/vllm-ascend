# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C12: reproduce the production MTE crash — block-table STRIDE
# mismatch combined with vllm's -1 sentinel for unallocated block slots.
#
# Why C11 came back ALL OK:
#   C11 used an IDENTITY block table (every slot a valid block id).  With an
#   identity table, a wrong row stride only makes the kernel read the *wrong
#   but valid* block id, so the output is wrong (WRONG) — it never crashes.
#   The production crash needs a -1 sentinel: vllm block tables hold `-1` in
#   every slot beyond a request's current KV length.  When the graph-baked
#   stride (maxNumBlocksPerBatch = ceil(max_seqlen_k / page_size)) is smaller
#   than the block-table width (page_table.shape[1]), the kernel reads across
#   the row boundary into a *previous* request's unallocated (-1) slots:
#
#       blockTableId = gBlockTable.GetValue(BIdx*stride + nowNIdx) == -1
#       kOffset = 0xFFFFFFFF * blockSize * strideKV   -> invalid GM address
#       -> MTE "invalid GM address" fault (err 507011, SMMU).
#
#   The EAGER path is correct: flash_api.cpp sets maxNumBlocksPerBatch =
#   block_table.size(1), and ascend950's tiling_from_tensors.hpp does the same
#   (ctx.maxNumBlocksPerBatch = max_num_blocks_per_seq).  Only the ascend910 v3
#   GRAPH path (get_scheduler_metadata -> flash_api.cpp:621) bakes the stride
#   from max_seqlen_k, which vllm-ascend passes as max(seq_lens_list) = the
#   warmup batch's short max KV length.
#
# This script:
#   (1) prints the imported module path and the underlying .so mtime, to
#       confirm WHICH build is actually running (C11's ALL-OK is consistent
#       with an installed package that does NOT bake the stride from
#       max_seqlen_k — i.e. not this source tree);
#   (2) prints maxNumBlocksPerBatch read directly out of the scheduler_metadata
#       tiling (FAInferTilingData byte offset 36);
#   (3) reproduces the crash: a block table with -1 sentinels, graph captured
#       with a SHORT max_seqlen_k, replayed at the SAME (still short) KV length.
#       Because stride < table width, batch row 1 reads row 0's -1 slots.
#
# Read:
#   GRAPH cell -> CRASH (MTE invalid GM address) + EAGER cell -> OK
#       => stride mismatch is the root cause, confirmed.
#   BOTH OK => the running package does NOT bake stride from max_seqlen_k
#       (stride == table width), so the bug is only in this source tree.
#
# Usage:
#   python test_fa3_graph_stride_sentinel.py            # print version, run both
#   python test_fa3_graph_stride_sentinel.py --graph    # graph cell only
#   python test_fa3_graph_stride_sentinel.py --eager    # eager cell only
#
# WARNING: a CRASH leaves the NPU in a bad state; reset the device before
# re-running anything.

import argparse
import os
import subprocess
import sys
import time
from importlib import util as importlib_util

import torch

NUM_TOKENS = 128
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_SIZE = 128
BLOCK_SIZE = 128
DTYPE = torch.bfloat16
SCALE = 1.0 / (HEAD_SIZE ** 0.5)

# fa_metadata::MASK_BYTES (2048x2048) = TilingOffset for the causal path.
MASK_BYTES = 2048 * 2048
# FAInferTilingData.maxNumBlocksPerBatch sits after 9 uint32 fields.
MAX_NUM_BLOCKS_OFFSET = 9 * 4

BT_WIDTH = 256          # block-table width == page_table.shape[1]
KV_LEN = 1024           # each request uses 8 blocks; rest of the row is -1
NUM_BLOCKS = 256        # KV cache holds 256 blocks (valid ids 0..255)
SENTINEL = -1           # vllm's "unallocated" marker


def load_fa3():
    for mod_name in ("flash_attn_npu_3", "flash_attn_npu_3"):
        if importlib_util.find_spec(mod_name) is not None:
            mod = __import__(
                mod_name,
                fromlist=["flash_attn_with_kvcache", "get_scheduler_metadata"],
            )
            return mod.flash_attn_with_kvcache, mod.get_scheduler_metadata, mod_name
    return None, None, None


def print_version():
    for mod_name in ("flash_attn_npu_3", "flash_attn_npu_3"):
        spec = importlib_util.find_spec(mod_name)
        if spec is None:
            continue
        print(f"[C12] module={mod_name} origin={spec.origin}")
        try:
            m = __import__(mod_name)
            so = getattr(m, "flash_attn_npu_3", None)
            p = getattr(so, "__file__", None)
            if p:
                mt = time.ctime(os.path.getmtime(p))
                print(f"[C12] .so={p} mtime={mt}")
            else:
                print("[C12] .so path not found")
        except Exception as e:  # noqa: BLE001
            print(f"[C12] version probe failed: {e}")


def build_inputs(kv_len):
    torch.manual_seed(0)
    q = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    cu_q = torch.arange(NUM_TOKENS + 1, dtype=torch.int32).npu()
    n_valid = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    row = [i % NUM_BLOCKS for i in range(n_valid)] + [SENTINEL] * (BT_WIDTH - n_valid)
    page_table = torch.tensor([row] * NUM_TOKENS, dtype=torch.int32).npu()
    cache_seqlens_buf = torch.full((NUM_TOKENS,), kv_len, dtype=torch.int32).npu()
    plan_buf = torch.full((NUM_TOKENS,), kv_len, dtype=torch.int32).npu()
    return q, k, v, cu_q, page_table, plan_buf, cache_seqlens_buf


def dump_stride(get_scheduler_metadata, max_seqlen_k):
    """Print maxNumBlocksPerBatch baked into the scheduler_metadata tiling."""
    q, k, v, cu_q, page_table, plan_buf, _ = build_inputs(KV_LEN)
    meta = get_scheduler_metadata(
        batch_size=NUM_TOKENS,
        max_seqlen_q=1,
        max_seqlen_k=max_seqlen_k,
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
    cpu = meta.cpu()
    val = int(cpu[MASK_BYTES + MAX_NUM_BLOCKS_OFFSET:
                 MASK_BYTES + MAX_NUM_BLOCKS_OFFSET + 4].view(torch.int32)[0].item())
    print(f"[C12] max_seqlen_k={max_seqlen_k} -> maxNumBlocksPerBatch={val} "
          f"(block-table width={BT_WIDTH})")
    return meta, val


def run_eager():
    fa3_kvcache, _, _ = load_fa3()
    if fa3_kvcache is None:
        print("[C12 eager] SKIP (FA3 not installed)")
        return 3
    q, k, v, cu_q, page_table, _, cache_seqlens_buf = build_inputs(KV_LEN)
    out = fa3_kvcache(
        q, k, v,
        cache_seqlens=cache_seqlens_buf,
        page_table=page_table,
        cu_seqlens_q=cu_q,
        max_seqlen_q=1,
        softmax_scale=SCALE,
        causal=True,
        window_size=(-1, -1),
    )
    torch.npu.synchronize()
    _ = out.clone()
    print("[C12 eager] OK (no fault)")
    return 0


def run_graph(max_seqlen_k):
    fa3_kvcache, get_scheduler_metadata, _ = load_fa3()
    if fa3_kvcache is None:
        print("[C12 graph] SKIP (FA3 not installed)")
        return 3
    q, k, v, cu_q, page_table, _, cache_seqlens_buf = build_inputs(KV_LEN)
    meta, stride = dump_stride(get_scheduler_metadata, max_seqlen_k)

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
    graph.replay()
    torch.npu.synchronize()
    _ = captured.clone()
    print(f"[C12 graph] OK (stride={stride}, no fault)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--msk", type=int, default=1024)
    args = ap.parse_args()

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    if not args.graph and not args.eager:
        print("=" * 72)
        print("C12 block-table stride + -1 sentinel repro")
        print("=" * 72)
        print_version()
        r = subprocess.run(
            [sys.executable, __file__, "--eager"],
            capture_output=True, text=True, timeout=600,
        )
        print(f"  eager  -> {'OK' if r.returncode == 0 else f'CRASH({r.returncode})'}")
        if r.stdout.strip():
            print("  " + r.stdout.strip().replace("\n", "\n  "))
        r = subprocess.run(
            [sys.executable, __file__, "--graph"],
            capture_output=True, text=True, timeout=600,
        )
        tag = "OK" if r.returncode == 0 else f"CRASH({r.returncode})"
        print(f"  graph  -> {tag}")
        if r.stdout.strip():
            print("  " + r.stdout.strip().replace("\n", "\n  "))
        if r.returncode not in (0, 3) and r.stderr.strip():
            print("  stderr tail: " + "\n  ".join(r.stderr.splitlines()[-6:]))
        print("=" * 72)
        print("Expect: eager OK, graph CRASH (MTE invalid GM address)")
        print("If a CRASH leaves the device bad, reset it before re-running.")
        return

    if args.eager:
        sys.exit(run_eager())
    else:
        sys.exit(run_graph(args.msk))


if __name__ == "__main__":
    main()
