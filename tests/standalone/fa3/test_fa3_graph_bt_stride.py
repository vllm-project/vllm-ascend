# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C11: reproduce the production MTE crash — block-table STRIDE
# mismatch between the graph-baked tiling and the replay batch.
#
# Root cause (from FA3 source, csrc/ascend910/flash_attn_npu_3):
#
#   The paged KV read in pv_matmul.hpp does:
#       nowNIdx      = kvSIdx * 512 / blockSize + curBlockIdx     // block table col
#       blockTableId = gBlockTable.GetValue(BIdx*maxNumBlocksPerBatch + nowNIdx)
#       kOffset      = blockTableId * blockSize * strideKV + ...
#   `maxNumBlocksPerBatch` is the per-sequence STRIDE the kernel uses to walk the
#   block table.  In the GRAPH path it is baked by get_scheduler_metadata from the
#   caller's `max_seqlen_k`:
#       fa_metadata.aicpu:  maxNumBlocksPerBatch = ceil(max_seqlen_k / block_size)
#   In vllm-ascend attention_v1.py the graph capture passes
#       max_seqlen_k = max(seq_lens_list)          // WARMUP batch's actual max KV len
#   which is SMALL (warmup requests are short).  At replay, requests grow toward
#   max_model_len, so the kernel reads ceil(actual_kv/128) block-table columns per
#   sequence, but the baked stride (and possibly the block-table buffer width) is
#   much smaller -> the read overruns the block-table buffer -> garbage block id ->
#   invalid K/V address -> MTE "invalid GM address" fault (SMMU, err 507011).
#
#   The EAGER path does NOT have this bug: flash_api.cpp sets
#       maxNumBlocksPerBatch = block_table.size(1)   // the ACTUAL tensor width
#   so eager is the correct reference for comparison.
#
#   This is why C6/C7/C9/C10 never reproduced it: they all baked max_seqlen_k ==
#   replay_len (32768), so the stride always matched.  C10 also proved the
#   workspace-sizing hypothesis wrong (wsSplit is a generous upper bound).
#
# Read:
#   WRONG output (graph != eager) when max_seqlen_k < replay_len  -> stride bug.
#   CRASH when block-table width also == ceil(max_seqlen_k/128) and replay_len is
#       long (last batch row reads past the buffer) -> the production fault.
#
# Usage:
#   python test_fa3_graph_bt_stride.py                              # full sweep
#   python test_fa3_graph_bt_stride.py --msk 1024 --btw 256 --replay 32768
#   python test_fa3_graph_bt_stride.py --msk 1024 --btw 8 --replay 32768

import argparse
import subprocess
import sys
from importlib import util as importlib_util

import torch

NUM_TOKENS = 128
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_SIZE = 128
BLOCK_SIZE = 128
DTYPE = torch.bfloat16
SCALE = 1.0 / (HEAD_SIZE ** 0.5)

# sweep: (max_seqlen_k, block_table_width, replay_len)
CELLS = [
    # (a) stride 8, wide buffer -> WRONG (deterministic), no fault
    (1024, 256, 32768),
    # (b) stride 256 == width 256 -> OK (control)
    (32768, 256, 32768),
    # (c) stride 8, narrow buffer == stride -> OOB at last row -> CRASH (production)
    (1024, 8, 32768),
    # (d) stride 8, narrow buffer, short replay -> OK (control)
    (1024, 8, 1024),
]


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


def build_inputs(bt_width: int, replay_len: int):
    """K/V cache has 256 blocks; block table is identity (logical j -> physical j)."""
    num_blocks = _ceil_div(32768, BLOCK_SIZE)  # 256 blocks, enough for replay 32768
    torch.manual_seed(0)
    q = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(
        num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE
    ).npu()
    v = torch.randn_like(k)

    cu_q = torch.arange(NUM_TOKENS + 1, dtype=torch.int32).npu()
    # identity block table: col j -> physical block j (valid id 0..num_blocks-1)
    rows = []
    for _ in range(NUM_TOKENS):
        rows.append([j % num_blocks for j in range(bt_width)])
    page_table = torch.tensor(rows, dtype=torch.int32).npu()
    plan_buf = torch.full((NUM_TOKENS,), replay_len, dtype=torch.int32).npu()
    cache_seqlens_buf = torch.full((NUM_TOKENS,), replay_len, dtype=torch.int32).npu()

    return q, k, v, cu_q, page_table, plan_buf, cache_seqlens_buf


def run_graph(max_seqlen_k: int, bt_width: int, replay_len: int):
    fa3_kvcache, get_scheduler_metadata, _ = load_fa3()
    q, k, v, cu_q, page_table, plan_buf, cache_seqlens_buf = build_inputs(
        bt_width, replay_len
    )

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
    return captured.clone().cpu()


def run_eager(bt_width: int, replay_len: int):
    fa3_kvcache, _, _ = load_fa3()
    q, k, v, cu_q, page_table, plan_buf, cache_seqlens_buf = build_inputs(
        bt_width, replay_len
    )
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
    return out.clone().cpu()


def run_cell(max_seqlen_k: int, bt_width: int, replay_len: int) -> int:
    fa3_kvcache, _, _ = load_fa3()
    if fa3_kvcache is None:
        print(f"[msk{max_seqlen_k} w{bt_width} r{replay_len}] SKIP", flush=True)
        return 3

    torch.npu.empty_cache()
    graph_out = run_graph(max_seqlen_k, bt_width, replay_len)
    eager_out = run_eager(bt_width, replay_len)

    diff = (graph_out.float() - eager_out.float()).abs()
    maxdiff = float(diff.max())
    # bf16 rounding ~ 1e-2 tolerance on top of numerical noise
    ok = maxdiff < 1e-1
    status = "OK" if ok else "WRONG"
    print(
        f"[msk{max_seqlen_k} w{bt_width} r{replay_len}] {status} "
        f"maxdiff={maxdiff:.4f} (graph vs eager)", flush=True,
    )
    return 0 if ok else 2


def sweep():
    print("=" * 72)
    print("C11 block-table stride sweep  batch=128 kv=8 (non-split)")
    print("=" * 72)
    for (msk, btw, rl) in CELLS:
        cmd = [
            sys.executable, __file__,
            "--msk", str(msk), "--btw", str(btw), "--replay", str(rl),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            print(f"  msk={msk:<6} w={btw:<4} r={rl:<6} -> TIMEOUT")
            continue
        if r.returncode == 0:
            tag = "OK"
        elif r.returncode == 3:
            tag = "SKIP"
        elif r.returncode == 2:
            tag = "WRONG"
        else:
            tag = f"CRASH({r.returncode})"
        detail = ""
        if r.stdout:
            detail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        print(f"  msk={msk:<6} w={btw:<4} r={rl:<6} -> {tag}  {detail}")
        if r.returncode not in (0, 2, 3) and r.stderr.strip():
            tail = "\n".join(r.stderr.splitlines()[-6:])
            print(f"         stderr tail: {tail}")
    print("=" * 72)
    print("Expect: (a) WRONG, (b) OK, (c) CRASH, (d) OK")
    print("If a CRASH leaves the device bad, reset it before re-running.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msk", type=int, default=None)
    ap.add_argument("--btw", type=int, default=None)
    ap.add_argument("--replay", type=int, default=None)
    args = ap.parse_args()

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")
    if args.msk is not None and args.btw is not None and args.replay is not None:
        sys.exit(run_cell(args.msk, args.btw, args.replay))
    sweep()


if __name__ == "__main__":
    main()
