# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C22: multi-batch decode with LONG KV, reordered block tables, and a
# scheduler_metadata baked at SHORT cache_seqlens (=> flashDecodeFlag=0, non-split).
#
# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------
# All C14-C21 standalone experiments were batch=1.  Production decode is
# batch=num_tokens (multiple requests, each with its own KV length = prefill
# length, possibly >= 1024) and its graph path bakes scheduler_metadata at
# capture time with the WARMUP cache_seqlens (short) -> flashDecodeFlag=0
# (non-split), while eager re-bakes at the real length -> flag=1 (split).
# The non-split kernel's MULTI-BATCH handling (blockBOffset = BIdx *
# maxNumBlocksPerBatch, totalTaskNum mapping) has never been exercised.
#
# C22 reproduces that exact shape:
#   batch = N requests, cache_seqlens = long & UNEQUAL (>= 1024),
#   block_table = [N, width] reordered rows with -1 tails,
#   metadata baked with cache_seqlens=[16]*N  (flashDecodeFlag=0 / non-split),
#   real call with cache_seqlens = the long lengths.
#
# Read:
#   any batch's diff LARGE => non-split multi-batch long-KV bug CONFIRMED.
#   all ~1e-4             => multi-batch non-split is fine; look elsewhere.
#
# Usage:
#   python test_fa3_graph_multibatch.py
#   BATCH=4 python test_fa3_graph_multibatch.py
#   KVCSV=512,1024,2048,4096 python test_fa3_graph_multibatch.py

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
        scores = torch.einsum("hd,thd->ht", q_f, k_g) * SCALE  # (H, seq)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("ht,thd->hd", attn, v_g)  # (H, D)
        outs.append(out)
    return torch.stack(outs, dim=0)  # (batch, H, D)


def main():
    kv_csv = os.environ.get("KVCSV", "512,1024,2048,4096")
    seqlens = [int(x) for x in kv_csv.split(",") if x.strip() != ""]
    batch = len(seqlens)
    num_blocks_pool = 128
    width = 128  # block-table width = max_blocks_per_seq (production: max_model_len/128)
    max_block_needed = max(_ceil_div(s, BLOCK_SIZE) for s in seqlens)
    assert max_block_needed <= num_blocks_pool, "KV too long for physical pool"

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    torch.manual_seed(0)

    print("=" * 72)
    print(f"C22 multi-batch non-split   batch={batch}  seqlens={seqlens}")
    print(f"    width={width}  pool={num_blocks_pool}  max_block_needed={max_block_needed}")
    print("=" * 72)

    q = torch.randn(batch, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(num_blocks_pool, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)

    cu_q = torch.arange(batch + 1, dtype=torch.int32).npu()

    # Reordered block table with -1 tail, one row per request.
    bt = torch.full((batch, width), -1, dtype=torch.int32)
    for b, s in enumerate(seqlens):
        nblk = _ceil_div(s, BLOCK_SIZE)
        ids = torch.randperm(num_blocks_pool, dtype=torch.int32)[:nblk]
        bt[b, :nblk] = ids
    block_table = bt.npu()

    seq_short = torch.full((batch,), 16, dtype=torch.int32).npu()   # bake flag=0
    seq_real = torch.tensor(seqlens, dtype=torch.int32).npu()        # real lengths

    meta = _make_meta(batch, seq_short, width * BLOCK_SIZE)

    # Warm up once (identity short) then run the real long-KV call.
    warm_bt = torch.full((batch, width), -1, dtype=torch.int32)
    for b, s in enumerate(seqlens):
        nblk = _ceil_div(s, BLOCK_SIZE)
        warm_bt[b, :nblk] = torch.arange(nblk, dtype=torch.int32)
    _run_fa3(q, k, v, seq_real, cu_q, warm_bt.npu(), meta)
    torch.npu.synchronize()

    out = _run_fa3(q, k, v, seq_real, cu_q, block_table, meta)
    torch.npu.synchronize()
    ref = manual_ref_batch(q, k, v, block_table.cpu(), seqlens)

    print("-" * 72)
    print("per-batch max-abs-diff (FA3 non-split vs float32 ref):")
    for b, s in enumerate(seqlens):
        d = _max_abs_diff(out[b], ref[b])
        flag = "  <-- WRONG" if d > 0.01 else ""
        print(f"  batch {b}  seqlen={s:5d}  : {d:.6f}{flag}")
    print(f"  overall               : {_max_abs_diff(out, ref):.6f}")

    torch.npu.synchronize()

    print("-" * 72)
    print("Read:")
    print("  any batch LARGE => non-split multi-batch long-KV bug CONFIRMED.")
    print("  all ~1e-4       => multi-batch non-split is fine; look elsewhere.")
    print("-" * 72)


if __name__ == "__main__":
    main()
