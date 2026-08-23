# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C23: does baking scheduler_metadata at MAX cache_seqlens (the
# proposed tiling-sinking fix) produce CORRECT results when replayed with the
# REAL (shorter) cache_seqlens?
#
# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------
# C22 CONFIRMED the root cause: bake with SHORT cache_seqlens (=> maxKvSeqlen<1024
# => flashDecodeFlag=0 / non-split) then call with LONG cache_seqlens => ALL
# batches wrong (0.35-0.47).  The non-split kernel is only correct for short KV;
# production graph capture bakes it with the warmup (short) seq_lens and replays
# with real (long) seq_lens -> the deterministic graph-vs-eager decode bug.
#
# The fix is to bake at MAX capacity so flashDecodeFlag=1 (split path) and the
# split schedule (coreInfo) is computed for the worst case.  The kernel reads
# gActualKvseqlen at runtime, so it should limit itself to the real length.
#
# C23 answers:
#   (Q1) control: bake actual long   -> call actual long   (eager equivalent)
#   (Q2) fix:     bake MAX (16384)   -> call actual mixed  (the production fix)
#   (Q3) fix:     bake MAX (16384)   -> call actual SHORT  (worst-case -> short)
#   (Q4) fix:     bake MAX (16384)   -> call single long   (batch=1)
#
# Read:
#   Q1 correct, Q2/Q3/Q4 correct => "bake MAX" is the fix; implement in capture.
#   Q1 correct, Q2/Q3 wrong       => split schedule baked at MAX breaks on short
#                                   replay; need a different approach.
#   Q1 wrong                     => split path itself is broken for long KV
#                                   (separate bug, eager would be wrong too).
#
# Usage:
#   python test_fa3_graph_bakemax.py
#   KVCSV=512,1024,2048,4096 python test_fa3_graph_bakemax.py
#   BAKEMAX=16384 python test_fa3_graph_bakemax.py

import os
from importlib import util as importlib_util

import torch
import torch_npu

_HAS_FA3 = False
_fa3_kvcache = None
_get_scheduler_metadata = None

for _mod_name in ("flash_attn_npu_3", "flash_attn_npu_3"):
    if importlib_util.find_spec(_mod_name) is not None:
        try:
            _mod = __import__(
                _mod_name,
                fromlist=["flash_attn_with_kvcache", "get_scheduler_metadata"],
            )
            _fa3_kvcache = _mod.flash_attn_with_kvcache
            _get_scheduler_metadata = _mod.get_scheduler_metadata
            _HAS_FA3 = True
            print(f"[import] FA3 loaded from {_mod_name}")
            break
        except (ImportError, AttributeError) as exc:
            print(f"[import] {_mod_name} found but failed: {exc}")

if not _HAS_FA3:
    raise SystemExit("flash_attn_with_kvcache (FA3) is not installed.")

HEAD_SIZE = 128
NUM_HEADS = 32
NUM_KV_HEADS = 8
BLOCK_SIZE = 128
DTYPE = torch.bfloat16
SCALE = 1.0 / (HEAD_SIZE ** 0.5)
GROUP = NUM_HEADS // NUM_KV_HEADS


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _make_meta(batch, cache_seqlens, maxk):
    cu_q = torch.arange(batch + 1, dtype=torch.int32).npu()
    return _get_scheduler_metadata(
        batch_size=batch,
        max_seqlen_q=1,
        max_seqlen_k=maxk,
        num_heads_q=NUM_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        headdim=HEAD_SIZE,
        cache_seqlens=cache_seqlens,
        qkv_dtype=DTYPE,
        cu_seqlens_q=cu_q,
        page_size=BLOCK_SIZE,
        causal=True,
    )


def _run_fa3(q, k, v, cache_seqlens, cu_q, page_table, meta):
    return _fa3_kvcache(
        q, k, v,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_q,
        max_seqlen_q=1,
        softmax_scale=SCALE,
        causal=True,
        window_size=(-1, -1),
        scheduler_metadata=meta,
    )


def manual_ref_batch(q, k, v, block_table, seqlens):
    """float32 GQA attention, one row per batch, using each row's block ids."""
    outs = []
    for b, seq_len in enumerate(seqlens):
        nblk = _ceil_div(seq_len, BLOCK_SIZE)
        ids = block_table[b, :nblk].cpu().tolist()
        blks = [k[i] for i in ids]
        k_flat = torch.cat([blk for blk in blks], dim=0).float()[:seq_len]
        v_flat = torch.cat([blk for blk in blks], dim=0).float()[:seq_len]
        k_g = k_flat.repeat_interleave(GROUP, dim=1)  # (seq, H, D)
        v_g = v_flat.repeat_interleave(GROUP, dim=1)
        q_f = q[b].float()  # (H, D)
        scores = torch.einsum("hd,thd->ht", q_f, k_g) * SCALE
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("ht,thd->hd", attn, v_g)
        outs.append(out)
    return torch.stack(outs, dim=0)


def _mk_block_table(batch, width, seqlens, num_blocks_pool, seed):
    g = torch.Generator().manual_seed(seed)
    bt = torch.full((batch, width), -1, dtype=torch.int32)
    for b, s in enumerate(seqlens):
        nblk = _ceil_div(s, BLOCK_SIZE)
        ids = torch.randperm(num_blocks_pool, generator=g, dtype=torch.int32)[:nblk]
        bt[b, :nblk] = ids
    return bt.npu()


def _report(tag, out, ref, seqlens):
    print(f"  [{tag}]")
    for b, s in enumerate(seqlens):
        d = _max_abs_diff(out[b], ref[b])
        flag = "  <-- WRONG" if d > 0.01 else ""
        print(f"    batch {b}  seqlen={s:5d}  : {d:.6f}{flag}")


def main():
    kv_csv = os.environ.get("KVCSV", "512,1024,2048,4096")
    seqlens = [int(x) for x in kv_csv.split(",") if x.strip() != ""]
    batch = len(seqlens)
    num_blocks_pool = 128
    width = 128
    maxk = width * BLOCK_SIZE  # max_model_len equivalent
    bakemax = int(os.environ.get("BAKEMAX", str(maxk)))
    assert bakemax <= maxk, "bake length exceeds max_model_len"

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    torch.manual_seed(0)

    print("=" * 72)
    print(f"C23 bake-max   batch={batch}  seqlens={seqlens}")
    print(f"    width={width}  maxk={maxk}  BAKEMAX={bakemax}")
    print("=" * 72)

    q = torch.randn(batch, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(num_blocks_pool, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    cu_q = torch.arange(batch + 1, dtype=torch.int32).npu()

    block_table = _mk_block_table(batch, width, seqlens, num_blocks_pool, 0)
    seq_real = torch.tensor(seqlens, dtype=torch.int32).npu()

    # Q1 control: bake with the ACTUAL long cache_seqlens (eager equivalent).
    print("-" * 72)
    print("[Q1] control: bake=actual-long, call=actual-long")
    meta_actual = _make_meta(batch, seq_real, maxk)
    out = _run_fa3(q, k, v, seq_real, cu_q, block_table, meta_actual)
    torch.npu.synchronize()
    ref = manual_ref_batch(q, k, v, block_table.cpu(), seqlens)
    _report("Q1", out, ref, seqlens)

    # Q2 fix: bake with MAX cache_seqlens, call with actual (mixed short+long).
    print("-" * 72)
    print(f"[Q2] fix: bake=MAX({bakemax}), call=actual-mixed")
    seq_bake = torch.full((batch,), bakemax, dtype=torch.int32).npu()
    meta_max = _make_meta(batch, seq_bake, maxk)
    out = _run_fa3(q, k, v, seq_real, cu_q, block_table, meta_max)
    torch.npu.synchronize()
    _report("Q2", out, ref, seqlens)

    # Q3 fix: bake MAX, call SHORT (worst-case -> short, all < 1024).
    short_seqlens = [min(s, 512) for s in seqlens]  # [512, 512, 512, 512] style
    # Make them distinct and all < 1024 to stress non-split-vs-split transition.
    short_seqlens = [128, 256, 512, 768][:batch]
    while len(short_seqlens) < batch:
        short_seqlens.append(256)
    short_seqlens = short_seqlens[:batch]
    print("-" * 72)
    print(f"[Q3] fix: bake=MAX({bakemax}), call=SHORT{short_seqlens}")
    bt_short = _mk_block_table(batch, width, short_seqlens, num_blocks_pool, 1)
    seq_short = torch.tensor(short_seqlens, dtype=torch.int32).npu()
    out = _run_fa3(q, k, v, seq_short, cu_q, bt_short, meta_max)
    torch.npu.synchronize()
    ref_short = manual_ref_batch(q, k, v, bt_short.cpu(), short_seqlens)
    _report("Q3", out, ref_short, short_seqlens)

    # Q4 fix: bake MAX, call single long (batch=1, the prior standalone shape).
    print("-" * 72)
    print(f"[Q4] fix: bake=MAX({bakemax}), call=single 2048")
    b1 = 1
    q1 = q[:1].contiguous()
    cu_q1 = torch.tensor([0, 1], dtype=torch.int32).npu()
    seq_bake1 = torch.full((b1,), bakemax, dtype=torch.int32).npu()
    meta_max1 = _make_meta(b1, seq_bake1, maxk)
    seq_2048 = torch.tensor([2048], dtype=torch.int32).npu()
    bt_2048 = _mk_block_table(b1, width, [2048], num_blocks_pool, 2)
    out = _run_fa3(q1, k, v, seq_2048, cu_q1, bt_2048, meta_max1)
    torch.npu.synchronize()
    ref_2048 = manual_ref_batch(q1, k, v, bt_2048.cpu(), [2048])
    _report("Q4", out, ref_2048, [2048])

    torch.npu.synchronize()
    print("-" * 72)
    print("Read:")
    print("  Q1 OK, Q2/Q3/Q4 OK => 'bake MAX' is the fix; implement in capture.")
    print("  Q2/Q3 wrong        => split schedule baked at MAX breaks on short replay.")
    print("  Q1 wrong           => split path itself broken for long KV.")
    print("-" * 72)


if __name__ == "__main__":
    main()
