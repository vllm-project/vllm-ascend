# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C15: is C14's "identity-table WRONG / reorder-table CORRECT"
# result (a) a real content-dependent kernel bug, or (b) a FIRST-CALL artifact?
#
# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------
# C13 proved FA3's non-split kernel is numerically correct on an IDENTITY
# block table (physical block id == logical index, no -1).  C14 then found the
# OPPOSITE: identity -> 0.179 (wrong), reorder / reorder+(-1) / graph -> ~4e-4
# (correct).  That is a paradox: the kernel's block-table access
# (qk_matmul.hpp getKVOffset / nowNIdx, pv_matmul.hpp ditto) is symmetric in
# the block-table VALUES, so identity and reorder cannot diverge unless either
#   (a) the FIRST call hits a warm-up / lazy-init / workspace issue (identity
#       happened to be cell #1 in C14), or
#   (b) the kernel genuinely special-cases identity/contiguous tables.
#
# C15 discriminates with a warm-up call and re-ordered / repeated cells.
#
# ---------------------------------------------------------------------------
# What it prints
# ---------------------------------------------------------------------------
#   [warm] reorder (not compared)                    -- warm up the kernel
#   [A   ] identity #1        vs manual ref          -- first *measured* call
#   [B   ] reorder            vs manual ref
#   [C   ] reversed-identity  vs manual ref          -- same blocks, reversed
#   [D   ] identity w/ swap   vs manual ref          -- block 0 <-> block 63
#   [E   ] identity #2        vs manual ref          -- same as A, last
#
# Read:
#   A large AND E large, B/C/D ~1e-4
#       => REAL identity-specific bug (contiguous-table fast path).
#   A large but E ~1e-4
#       => FIRST-CALL artifact; C14's identity cell was just unlucky to be #1.
#   C large but D ~1e-4
#       => block-table VALUE 0 is treated as pad/skip (not the ordering).
#   all ~1e-4
#       => identity/reorder both fine; look at batch>1 / stride next.
#
# Usage:
#   python test_fa3_graph_identity_vs_reorder.py
#   KV=2048 python test_fa3_graph_identity_vs_reorder.py

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
    """Non-split meta (cache_seqlens short -> flashDecodeFlag=0), stride=maxk/128."""
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
    """float32 GQA attention over the LOGICAL KV sequence defined by block_table."""
    nblk = _ceil_div(seq_len, BLOCK_SIZE)
    ids = block_table[:nblk].cpu().tolist()
    blks = [k[i] for i in ids]
    k_flat = torch.cat([b for b in blks], dim=0).float()[:seq_len]
    v_flat = torch.cat([v[i] for i in ids], dim=0).float()[:seq_len]
    k_g = k_flat.repeat_interleave(GROUP, dim=1)
    v_g = v_flat.repeat_interleave(GROUP, dim=1)
    q_f = q.float()
    scores = torch.einsum("bhd,thd->bht", q_f, k_g) * SCALE
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bht,thd->bhd", attn, v_g)


def main():
    kv = int(os.environ.get("KV", "2048"))
    num_blocks_pool = 64
    nblk = _ceil_div(kv, BLOCK_SIZE)
    assert nblk <= num_blocks_pool, "KV too long for the physical pool"

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    torch.manual_seed(0)

    print("=" * 72)
    print(f"C15 identity vs reorder (warm-up + repeat)   kv={kv}  nblk={nblk}")
    print("=" * 72)

    q = torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(num_blocks_pool, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 1], dtype=torch.int32).npu()
    seq = torch.tensor([kv], dtype=torch.int32).npu()

    # Non-split meta, stride == num_blocks_pool (mirrors production graph bake).
    meta = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                      num_blocks_pool * BLOCK_SIZE)

    # ---- warm-up: one FA3 call with a reorder table, NOT compared ----
    warm_perm = torch.randperm(num_blocks_pool, dtype=torch.int32).npu().unsqueeze(0)
    _run_fa3(q, k, v, seq, cu_q, warm_perm, meta)
    torch.npu.synchronize()

    # ---- build the tables ----
    page_id = torch.arange(num_blocks_pool, dtype=torch.int32).npu().unsqueeze(0)  # identity

    perm = torch.randperm(num_blocks_pool, dtype=torch.int32).npu()  # reorder
    page_reorder = perm.unsqueeze(0)

    rev_cpu = torch.arange(num_blocks_pool, dtype=torch.int32)  # reversed first nblk
    rev_cpu[:nblk] = torch.arange(nblk - 1, -1, -1, dtype=torch.int32)
    page_rev = rev_cpu.npu().unsqueeze(0)

    swp_cpu = torch.arange(num_blocks_pool, dtype=torch.int32)  # swap block 0 <-> 63
    swp_cpu[0] = num_blocks_pool - 1
    swp_cpu[num_blocks_pool - 1] = 0
    page_swp = swp_cpu.npu().unsqueeze(0)

    # ---- cells ----
    out_a = _run_fa3(q, k, v, seq, cu_q, page_id, meta)        # A: identity #1
    ref_a = manual_ref_blocktable(q, k, v, page_id[0], kv)

    out_b = _run_fa3(q, k, v, seq, cu_q, page_reorder, meta)   # B: reorder
    ref_b = manual_ref_blocktable(q, k, v, page_reorder[0], kv)

    out_c = _run_fa3(q, k, v, seq, cu_q, page_rev, meta)       # C: reversed
    ref_c = manual_ref_blocktable(q, k, v, page_rev[0], kv)

    out_d = _run_fa3(q, k, v, seq, cu_q, page_swp, meta)       # D: identity w/ swap
    ref_d = manual_ref_blocktable(q, k, v, page_swp[0], kv)

    out_e = _run_fa3(q, k, v, seq, cu_q, page_id, meta)        # E: identity #2
    ref_e = manual_ref_blocktable(q, k, v, page_id[0], kv)

    torch.npu.synchronize()

    print("-" * 72)
    print(f"[warm] reorder (not compared)")
    print(f"[A   ] identity #1       vs manual ref : {_max_abs_diff(out_a, ref_a):.6f}")
    print(f"[B   ] reorder           vs manual ref : {_max_abs_diff(out_b, ref_b):.6f}")
    print(f"[C   ] reversed-identity vs manual ref : {_max_abs_diff(out_c, ref_c):.6f}")
    print(f"[D   ] identity w/ swap  vs manual ref : {_max_abs_diff(out_d, ref_d):.6f}")
    print(f"[E   ] identity #2       vs manual ref : {_max_abs_diff(out_e, ref_e):.6f}")

    print("-" * 72)
    print("Read:")
    print("  A large AND E large, B/C/D ~1e-4 => REAL identity-specific bug.")
    print("  A large but E ~1e-4           => FIRST-CALL artifact (C14 cell order).")
    print("  C large but D ~1e-4           => block-table VALUE 0 is pad/skip.")
    print("  all ~1e-4                      => both fine; try batch>1 / stride.")
    print("-" * 72)


if __name__ == "__main__":
    main()
