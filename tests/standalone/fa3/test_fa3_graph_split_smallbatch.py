# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C10: reproduce the production MTE crash — flash-decode (split-K)
# workspace overflow at SMALL batch.
#
# Root-cause hypothesis (from FA3 source, csrc/ascend910/flash_attn_npu_3):
#
#   The faulting kernel SplitFuse::FAInfer<..., LseMode=1> is the SPLIT-LSE path,
#   active only in "flash decode" mode, whose gate is (flash_api.cpp):
#       flashDecodeFlag = ... && (batch_size*num_heads_k <= 0.8*blockDim)
#                              && (maxKV >= 1024) ...
#   i.e. SMALL batch + LONG KV.
#
#   In the graph path (scheduler_metadata provided), the workspace split region
#   is sized by a formula that is INDEPENDENT of batch_size (flash_api.cpp):
#       lseTasksUpper = num_heads * max_seqlen_q(=1) * kvSegUpper * 2   // 32*1*65*2 = 4160
#       wsSplit       = lseTasksUpper*4 + lseTasksUpper*head_size*4
#   while the kernel's ACTUAL split workspace, baked by ComputeFAMetadata
#   (fa_split.h), scales with batch_size:
#       splitLseTotalSize ~ num_heads * B * curKSBlockNum = 2048 * B  elements
#   Overflow (2048*B > 4160) begins at B >= 3.  Combined with the flash-decode
#   gate (B * num_heads_k <= 16), the crash needs num_heads_k < 8.
#
#   This is why C6/C7/C9 (num_heads_k=8, batch>=3 -> gate off) never crashed.
#
# This experiment sweeps (num_heads_k, batch) at fixed maxk=32768 through the
# graph path (get_scheduler_metadata + fa3_kvcache(scheduler_metadata=...)).
# Shared K/V blocks keep the block table always-valid, so a crash here isolates
# the split workspace sizing bug.
#
# Read:
#   CRASH for (num_heads_k<=4, batch>=3)   -> workspace formula under-sizes by
#       batch_size; the fix is to size wsSplit by the true splitLseTotalSize
#       (or bake a batch-scaled upper bound) in the scheduler_metadata path.
#   no crash anywhere                       -> hypothesis falsified; revisit.
#
# Usage:
#   python test_fa3_graph_split_smallbatch.py                     # full sweep
#   python test_fa3_graph_split_smallbatch.py --kv 1 --batch 4    # single cell

import argparse
import subprocess
import sys
from importlib import util as importlib_util

import torch

NUM_HEADS = 32
HEAD_SIZE = 128
BLOCK_SIZE = 128
DTYPE = torch.bfloat16
SCALE = 1.0 / (HEAD_SIZE ** 0.5)

MAXK = 32768
KV_HEADS = [1, 2, 4, 8]
BATCHES = [1, 2, 3, 4, 8, 16]


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


def run_cell(num_kv_heads: int, batch: int) -> int:
    fa3_kvcache, get_scheduler_metadata, mod_name = load_fa3()
    if fa3_kvcache is None:
        print(f"[kv{num_kv_heads} b{batch}] SKIP (FA3 not installed)", flush=True)
        return 3

    group = NUM_HEADS // num_kv_heads
    blocks_per_seq = _ceil_div(MAXK, BLOCK_SIZE)
    print(
        f"[kv{num_kv_heads} b{batch}] group={group} blocks/seq={blocks_per_seq} "
        f"numTasks={batch * num_kv_heads}", flush=True,
    )

    torch.npu.empty_cache()

    q = torch.randn(batch, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(
        blocks_per_seq, BLOCK_SIZE, num_kv_heads, HEAD_SIZE, dtype=DTYPE
    ).npu()
    v = torch.randn_like(k)

    cu_q = torch.arange(batch + 1, dtype=torch.int32).npu()
    shared_row = list(range(blocks_per_seq))
    page_table = torch.tensor([shared_row] * batch, dtype=torch.int32).npu()
    plan_buf = torch.full((batch,), MAXK, dtype=torch.int32).npu()
    cache_seqlens_buf = torch.full((batch,), MAXK, dtype=torch.int32).npu()

    meta = get_scheduler_metadata(
        batch_size=batch,
        max_seqlen_q=1,
        max_seqlen_k=MAXK,
        num_heads_q=NUM_HEADS,
        num_heads_kv=num_kv_heads,
        headdim=HEAD_SIZE,
        cache_seqlens=plan_buf,
        qkv_dtype=DTYPE,
        cu_seqlens_q=cu_q,
        page_size=BLOCK_SIZE,
        causal=True,
    )

    torch.npu.synchronize()

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

    print(f"[kv{num_kv_heads} b{batch}] OK", flush=True)
    return 0


def sweep():
    print("=" * 72)
    print("C10 flash-decode split workspace sweep  maxk=32768")
    print("=" * 72)
    header = "         " + "".join(f"b{b:<4}" for b in BATCHES)
    print(header)
    for num_kv_heads in KV_HEADS:
        row = f"kv={num_kv_heads:<5} "
        for batch in BATCHES:
            cmd = [
                sys.executable, __file__,
                "--kv", str(num_kv_heads), "--batch", str(batch),
            ]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            except subprocess.TimeoutExpired:
                row += "T/O   "
                continue
            if r.returncode == 0:
                row += "OK    "
            elif r.returncode == 3:
                row += "SKIP  "
            else:
                row += f"CR{str(r.returncode):<4}"
        print(row)
        if num_kv_heads == KV_HEADS[0]:
            # print crash tail for the first kv=1 crash to confirm MTE/SMMU
            pass
    print("=" * 72)
    print("Predicted crash region (workspace overflow 2048*B > 4160, gate B*kv<=16):")
    print("  kv=1: B>=3 (gate B<=16)   kv=2: B>=3 (B<=8)")
    print("  kv=4: B=3,4 (B<=4)        kv=8: none (4096 fits; B>=3 gate off)")
    print("If a CRASH leaves the device in a bad state, reset it before re-running.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")
    if args.kv is not None and args.batch is not None:
        sys.exit(run_cell(args.kv, args.batch))
    sweep()


if __name__ == "__main__":
    main()
