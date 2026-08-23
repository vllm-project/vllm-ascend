# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C16: what exactly about the block-table VALUES makes FA3 wrong?
#
# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------
# C15 (warm-up + repeat) showed, deterministically:
#   identity [0..15]        -> correct  (~4e-4)
#   reversed [15..0]        -> correct  (~4e-4)
#   reorder  (scattered)    -> wrong    (~0.16)
#   swap     ([63,1..15])   -> half-wrong (0.057)
# The diff correlates with the number of physical block ids >= 16 that the
# kernel reads.  Two candidate triggers:
#   (A) VALUE threshold: any block id >= T is mis-read.
#   (B) CONTIGUITY: contiguous (asc or desc) tables are fine, gaps are not.
# reversed [15..0] is contiguous AND low-id, so C15 cannot separate (A) from (B).
#
# C16 separates them:
#   sweep blockTable[0] = X, rest identity -> where does the diff jump?
#   blockTable[15] = 63 (high id at LAST position) -> position vs value?
#   contiguous-HIGH [48..63] and [63..48]          -> high ids but contiguous.
#
# Read:
#   diff jumps at X == 16 (or some T) in the sweep   => VALUE threshold == T.
#   contiguous-high cells ~1e-4                      => contiguity, NOT value.
#   contiguous-high cells LARGE                      => value threshold, not contiguity.
#   [15]=63 large but [0]=63 small                   => position-dependent.
#
# Usage:
#   python test_fa3_graph_blockid_threshold.py
#   KV=2048 python test_fa3_graph_blockid_threshold.py

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


def manual_ref_blocktable(q, k, v, block_table, seq_len):
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


def _table(first16, pool=64):
    """Build a 1xpool int32 table whose first 16 entries are `first16`."""
    rest = [i for i in range(pool) if i not in first16]
    row = list(first16) + rest
    return torch.tensor(row, dtype=torch.int32).npu().unsqueeze(0)


def main():
    kv = int(os.environ.get("KV", "2048"))
    num_blocks_pool = 64
    nblk = _ceil_div(kv, BLOCK_SIZE)
    assert nblk <= num_blocks_pool, "KV too long for the physical pool"

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    torch.manual_seed(0)

    print("=" * 72)
    print(f"C16 block-id threshold   kv={kv}  nblk={nblk}")
    print("=" * 72)

    q = torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(num_blocks_pool, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 1], dtype=torch.int32).npu()
    seq = torch.tensor([kv], dtype=torch.int32).npu()
    meta = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                      num_blocks_pool * BLOCK_SIZE)

    # warm-up (not compared)
    warm = _table(list(range(nblk)))
    _run_fa3(q, k, v, seq, cu_q, warm, meta)
    torch.npu.synchronize()

    print("-" * 72)
    print("[sweep] blockTable[0] = X, blocks 1..15 stay identity:")
    for X in (0, 1, 8, 15, 16, 17, 24, 32, 48, 63):
        first = [X] + list(range(1, nblk))
        tbl = _table(first)
        ref = manual_ref_blocktable(q, k, v, tbl[0], kv)
        out = _run_fa3(q, k, v, seq, cu_q, tbl, meta)
        print(f"  blockTable[0]={X:3d}  diff = {_max_abs_diff(out, ref):.6f}")

    print("-" * 72)
    print("[position] high id (63) at a specific slot:")
    for pos in (0, 7, 15):
        first = list(range(nblk))
        first[pos] = 63
        tbl = _table(first)
        ref = manual_ref_blocktable(q, k, v, tbl[0], kv)
        out = _run_fa3(q, k, v, seq, cu_q, tbl, meta)
        print(f"  blockTable[{pos:2d}]=63  diff = {_max_abs_diff(out, ref):.6f}")

    print("-" * 72)
    print("[contiguity] contiguous HIGH blocks:")
    for name, first in (("asc [48..63]", list(range(48, 64))),
                        ("desc [63..48]", list(range(63, 47, -1)))):
        tbl = _table(first)
        ref = manual_ref_blocktable(q, k, v, tbl[0], kv)
        out = _run_fa3(q, k, v, seq, cu_q, tbl, meta)
        print(f"  {name}  diff = {_max_abs_diff(out, ref):.6f}")

    torch.npu.synchronize()

    print("-" * 72)
    print("Read:")
    print("  sweep jumps at X==T            => VALUE threshold == T.")
    print("  contiguous-high ~1e-4          => CONTIGUITY trigger, not value.")
    print("  contiguous-high LARGE          => VALUE threshold, not contiguity.")
    print("  [pos]=63 large only at pos=0   => position-dependent.")
    print("-" * 72)


if __name__ == "__main__":
    main()
