# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C24: isolate the MULTI-BATCH bug using the kernel's own batch=1
# output as ground truth (NOT the hand-written float32 reference).
#
# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------
# C22/C23 showed batch=4 is WRONG (0.3-0.8) while batch=1 was always correct
# (0.0003).  But C23 Q1 (eager-equivalent multi-batch) was ALSO wrong, which
# contradicts "production eager is correct" — so either the kernel has a genuine
# multi-batch bug, or my hand-written reference is subtly wrong for batch>1.
#
# C24 removes the reference from the equation: run batch=4 and compare each row
# against a separate batch=1 call on the SAME block-table row + SAME bake length.
# Uniform seq_lens keep the flashDecodeFlag (split vs non-split) identical
# between the batch=4 and batch=1 calls, so any diff is purely the batch>1
# effect.
#
#   Cell N: seq=[512]*4  -> flag=0 (non-split), uniform
#   Cell S: seq=[2048]*4 -> flag=1 (split),    uniform
#
# Read:
#   batch4[b] != batch1[b] => multi-batch kernel bug CONFIRMED (ref-independent).
#   batch4[b] == batch1[b] => kernel multi-batch is FINE; my reference was wrong.
#   Cell S batch=1 itself wrong/crash => split path broken even for batch=1.
#
# Usage:
#   python test_fa3_graph_multibatch_isolate.py

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
    outs = []
    for b, seq_len in enumerate(seqlens):
        nblk = _ceil_div(seq_len, BLOCK_SIZE)
        ids = block_table[b, :nblk].cpu().tolist()
        blks = [k[i] for i in ids]
        k_flat = torch.cat([blk for blk in blks], dim=0).float()[:seq_len]
        v_flat = torch.cat([blk for blk in blks], dim=0).float()[:seq_len]
        k_g = k_flat.repeat_interleave(GROUP, dim=1)
        v_g = v_flat.repeat_interleave(GROUP, dim=1)
        q_f = q[b].float()
        scores = torch.einsum("hd,thd->ht", q_f, k_g) * SCALE
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("ht,thd->hd", attn, v_g)
        outs.append(out)
    return torch.stack(outs, dim=0)


def _mk_block_table(batch, width, seqlen, num_blocks_pool, seed):
    g = torch.Generator().manual_seed(seed)
    nblk = _ceil_div(seqlen, BLOCK_SIZE)
    bt = torch.full((batch, width), -1, dtype=torch.int32)
    for b in range(batch):
        ids = torch.randperm(num_blocks_pool, generator=g, dtype=torch.int32)[:nblk]
        bt[b, :nblk] = ids
    return bt.npu()


def _cell(tag, seqlen, batch, num_blocks_pool, width, maxk):
    """Run batch=N vs N x batch=1, compare per-row."""
    print("-" * 72)
    print(f"[{tag}] seqlen={seqlen}  batch={batch}")

    # One physical K/V pool, shared by all rows (block ids are disjunct per row).
    q = torch.randn(batch, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(num_blocks_pool, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)

    seq_bake = torch.full((batch,), seqlen, dtype=torch.int32).npu()   # uniform => same flag as batch=1
    cu_q = torch.arange(batch + 1, dtype=torch.int32).npu()
    block_table = _mk_block_table(batch, width, seqlen, num_blocks_pool, 7)

    # batch=4 call
    meta4 = _make_meta(batch, seq_bake, maxk)
    out4 = _run_fa3(q, k, v, seq_bake, cu_q, block_table, meta4)
    torch.npu.synchronize()

    # per-row batch=1 calls (same row of block_table, same bake length)
    out1 = torch.zeros_like(out4)
    for b in range(batch):
        q1 = q[b:b + 1].contiguous()
        cu_q1 = torch.tensor([0, 1], dtype=torch.int32).npu()
        bt1 = block_table[b:b + 1].contiguous()
        seq1 = torch.tensor([seqlen], dtype=torch.int32).npu()
        meta1 = _make_meta(1, seq1, maxk)
        o = _run_fa3(q1, k, v, seq1, cu_q1, bt1, meta1)
        torch.npu.synchronize()
        out1[b] = o[0]

    print(f"  batch4[b] vs batch1[b] (kernel-only, no hand reference):")
    for b in range(batch):
        d = _max_abs_diff(out4[b], out1[b])
        flag = "  <-- MULTI-BATCH BUG" if d > 0.01 else ""
        print(f"    row {b}  : {d:.6f}{flag}")

    ref = manual_ref_batch(q, k, v, block_table.cpu(), [seqlen] * batch)
    print(f"  batch4 vs hand-ref:")
    for b in range(batch):
        d = _max_abs_diff(out4[b], ref[b])
        flag = "  <-- vs-ref WRONG" if d > 0.01 else ""
        print(f"    row {b}  : {d:.6f}{flag}")


def main():
    width = 128
    num_blocks_pool = 128
    maxk = width * BLOCK_SIZE

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    torch.manual_seed(0)
    print("=" * 72)
    print(f"C24 multi-batch isolate   width={width}  maxk={maxk}")
    print("=" * 72)

    _cell("N non-split (flag=0)", 512, 4, num_blocks_pool, width, maxk)
    _cell("S split (flag=1)", 2048, 4, num_blocks_pool, width, maxk)

    torch.npu.synchronize()
    print("-" * 72)
    print("Read:")
    print("  batch4[b] != batch1[b] => multi-batch kernel bug CONFIRMED.")
    print("  batch4[b] == batch1[b] => kernel fine; my reference was wrong.")
    print("  Cell S batch1 wrong/crash => split path broken even batch=1.")
    print("-" * 72)


if __name__ == "__main__":
    main()
