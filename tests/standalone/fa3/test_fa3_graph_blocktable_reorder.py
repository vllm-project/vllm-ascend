# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C14: does FA3's NON-SPLIT path stay numerically correct when the
# block_table is (a) a REORDERED permutation of physical block ids and (b) the
# unused tail of each row is filled with vllm's -1 sentinel?
#
# ---------------------------------------------------------------------------
# Why this is the #1 remaining suspect
# ---------------------------------------------------------------------------
# C13 v2 proved FA3's kernel is numerically correct (all cells ~1e-4) on an
# IDENTITY block table (physical block i == logical block i, no -1).  But the
# production graph replay feeds the kernel a REAL vllm block_table:
#   - physical block ids are a REORDERED permutation (paged KV is not contiguous)
#   - each row's unused tail is -1 (vllm's unallocated-slot sentinel)
#
# The non-split kernel walks the paged block table as
#   blockTable[BIdx * maxNumBlocksPerBatch + nowNIdx]
# with nowNIdx driven by the runtime kvSeqlen (see mha_fwd_kvcache.cpp:266/348,
# qk_matmul.hpp:241, pv_matmul.hpp:197).  If maxNumBlocksPerBatch (the row
# STRIDE) is wrong OR the -1 tail is read, the kernel reads a wrong block id or
# an invalid address -> wrong output (or the MTE crash seen with max_model_len).
#
# C13 held stride fixed (maxk=kv_long) but never tested a REORDERED / -1 table.
# C14 closes that gap.
#
# ---------------------------------------------------------------------------
# What it prints
# ---------------------------------------------------------------------------
#   [ok ] identity table, non-split  vs manual ref  -> sanity (reproduce C13)
#   [x  ] reordered table, non-split vs manual ref  -> the production shape
#   [x  ] reordered+(-1) table, non-split vs manual ref
#   [x  ] reordered+(-1), graph replay vs manual ref
#
# Read:
#   identity cell ~1e-4 but reordered cell LARGE
#       => CONFIRMED: block_table reordering / -1 is the decode-precision bug.
#   all cells ~1e-4
#       => block_table content is NOT the bug; look at the graph refresh
#          timing / stream sync instead.
#
# Usage:
#   python test_fa3_graph_blocktable_reorder.py
#   KV=2048 python test_fa3_graph_blocktable_reorder.py

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
    """Non-split meta: cache_seqlens short (<1024) forces flashDecodeFlag=0."""
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
        q,
        k,
        v,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_q,
        max_seqlen_q=1,
        softmax_scale=SCALE,
        causal=True,
        window_size=(-1, -1),
        scheduler_metadata=meta,
    )


def manual_ref_blocktable(q, k, v, block_table, seq_len):
    """float32 GQA attention over the LOGICAL KV sequence defined by block_table.

    block_table is a 1-D int32 tensor of physical block ids (length >= nblk).
    The logical KV sequence is [k[bt[0]], k[bt[1]], ..., k[bt[nblk-1]]][:seq_len],
    i.e. the paged KV as the kernel should read it.  Independent of FA3.
    """
    nblk = _ceil_div(seq_len, BLOCK_SIZE)
    ids = block_table[:nblk].cpu().tolist()
    # Gather the physical blocks in logical order, then flatten to (nblk*BS, Hkv, D).
    blks = [k[i] for i in ids]  # each (BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    k_flat = torch.cat([b for b in blks], dim=0).float()[:seq_len]
    v_flat = torch.cat([v[i] for i in ids], dim=0).float()[:seq_len]
    k_g = k_flat.repeat_interleave(GROUP, dim=1)  # (seq_len, H, D)
    v_g = v_flat.repeat_interleave(GROUP, dim=1)
    q_f = q.float()  # (1, H, D)
    scores = torch.einsum("bhd,thd->bht", q_f, k_g) * SCALE
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bht,thd->bhd", attn, v_g)  # (1, H, D)


def main():
    kv = int(os.environ.get("KV", "2048"))
    num_blocks_pool = 64  # physical block pool size (>= nblk, with slack)
    nblk = _ceil_div(kv, BLOCK_SIZE)
    assert nblk <= num_blocks_pool, "KV too long for the physical pool"

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    print("=" * 72)
    print(f"C14 block_table reorder / -1 sentinel   kv={kv}  nblk={nblk}")
    print("=" * 72)

    q = torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    # Physical pool: (num_blocks_pool, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE).
    k = torch.randn(num_blocks_pool, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 1], dtype=torch.int32).npu()
    seq = torch.tensor([kv], dtype=torch.int32).npu()

    # Meta: cache_seqlens short -> non-split.  maxk is held at
    # num_blocks_pool*BLOCK_SIZE so maxNumBlocksPerBatch (the row STRIDE)
    # == the block_table width == num_blocks_pool -- matching the production
    # "stride == block_tables.shape[1]" condition (fixed in C12).
    meta = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                      num_blocks_pool * BLOCK_SIZE)

    # ---- cell 1: identity table (reproduce C13's correct non-split) ----
    page_id = torch.arange(num_blocks_pool, dtype=torch.int32).npu().unsqueeze(0)
    ref = manual_ref_blocktable(q, k, v, page_id[0], kv)
    torch.npu.synchronize()
    out_id = _run_fa3(q, k, v, seq, cu_q, page_id, meta)

    # ---- cell 2: reordered table, no -1 (permutation of valid ids) ----
    perm = torch.randperm(num_blocks_pool, dtype=torch.int32).npu()  # deterministic? use a fixed seed
    page_reorder = perm.unsqueeze(0)
    ref_reorder = manual_ref_blocktable(q, k, v, page_reorder[0], kv)
    torch.npu.synchronize()
    out_reorder = _run_fa3(q, k, v, seq, cu_q, page_reorder, meta)

    # ---- cell 3: reordered table + -1 sentinel tail ----
    # vllm block_table: first nblk slots are the (reordered) valid ids, the rest -1.
    row = torch.full((num_blocks_pool,), -1, dtype=torch.int32).npu()
    row[:nblk] = perm[:nblk]
    page_reorder_neg = row.unsqueeze(0)
    ref_reorder_neg = manual_ref_blocktable(q, k, v, page_reorder_neg[0], kv)
    torch.npu.synchronize()
    out_reorder_neg = _run_fa3(q, k, v, seq, cu_q, page_reorder_neg, meta)

    # ---- cell 4: reordered + -1, via NPUGraph replay ----
    # Simulate the production refresh: capture with a zero/identity buffer,
    # then copy the real (reordered, -1) table in before replay.
    bt_buf = torch.zeros(1, num_blocks_pool, dtype=torch.int32).npu()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = _run_fa3(q, k, v, seq, cu_q, bt_buf, meta)
    torch.npu.synchronize()
    bt_buf.copy_(page_reorder_neg)
    graph.replay()
    torch.npu.synchronize()
    out_graph_neg = captured.clone()

    print("-" * 72)
    print(f"[ok ] identity table          vs manual ref : "
          f"{_max_abs_diff(out_id, ref):.6f}")
    print(f"[x  ] reordered table        vs manual ref : "
          f"{_max_abs_diff(out_reorder, ref_reorder):.6f}")
    print(f"[x  ] reordered + (-1) tail  vs manual ref : "
          f"{_max_abs_diff(out_reorder_neg, ref_reorder_neg):.6f}")
    print(f"[x  ] reordered + (-1) graph vs manual ref : "
          f"{_max_abs_diff(out_graph_neg, ref_reorder_neg):.6f}")

    print("-" * 72)
    print("Read:")
    print("  identity ~1e-4 but reordered cell LARGE")
    print("      => CONFIRMED: block_table reordering / -1 is the bug.")
    print("  reordered ~1e-4 but (-1) cell LARGE")
    print("      => the -1 sentinel tail is read (stride still wrong).")
    print("  graph cell LARGE while eager cells ~1e-4")
    print("      => the graph refresh (bt_buf.copy_) is stale / not synced.")
    print("  all cells ~1e-4 => block_table content is NOT the bug.")
    print("-" * 72)


if __name__ == "__main__":
    main()
