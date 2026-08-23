# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C13: does the graph-baked scheduler_metadata bake a
# flashDecodeFlag (split vs non-split) that depends on the WARMUP cache_seqlens,
# diverging from the eager path (which bakes it from the ACTUAL seq_lens)?
#
# ---------------------------------------------------------------------------
# Why this is the #1 remaining decode-precision suspect
# ---------------------------------------------------------------------------
# Eager and graph BOTH go through FA3's flash_attn_with_kvcache with
# scheduler_metadata (attention_v1.py:1691 eager / 1634 graph).  The kernel
# itself is correct (eager decode produces the right answer).  The only
# difference is HOW the metadata is built:
#
#   eager  -> _build_fa3_scheduler_metadata: cache_seqlens = ACTUAL batch seq_lens
#   graph  -> _get_fa3_graph_params:        cache_seqlens = WARMUP batch seq_lens
#
# ComputeFAMetadata (fa_metadata.aicpu:59-119) bakes two kinds of seq-dependent
# fields FROM the cache_seqlens it is given:
#   - maxKvSeqlen     = max over cache_seqlens
#   - flashDecodeFlag = (batch*numHeadsK <= 0.8*blockDim) && (maxKvSeqlen >= 1024)
#                       ... && (maxQ*group<=128) && (maxQ<=16) && (minQ>0)
#   and, when flashDecodeFlag=1, splitBN2S1GS2() bakes a SPLIT tile plan whose
#   coreInfo[]/splitInfo[]/needCoreNum depend on those seq_lens.
#
# The eager path (flash_api.cpp:443-448) then uses needCoreNum as launchBlockDim;
# the graph path (flash_api.cpp:310-329) hard-codes launchBlockDim = blockDim and
# NEVER reads needCoreNum.  So the graph's split tiling is doubly stale: it is
# baked from warmup seq_lens AND launched with a blockDim that ignores the
# baked needCoreNum.
#
# Prior experiments (A..C12) never exercised this: their get_scheduler_metadata
# always used batch_size == actual batch AND cache_seqlens == the replay length,
# so they baked the SAME flashDecodeFlag as a fresh eager call would.
#
# ---------------------------------------------------------------------------
# What it prints
# ---------------------------------------------------------------------------
#   [dump] tiling fields for meta built with cache_seqlens = short / 1024 / long
#          (max_seqlen_k held at KV_LONG so stride is NOT the variable)
#          -> shows flashDecodeFlag toggling 0 -> 1 and needCoreNum shifting
#   [ok ] eager(long meta)    vs manual ref  -> FA3 split path is CORRECT
#   [x  ] eager(short meta)   vs manual ref  -> non-split (stale meta) WRONG?
#   [x  ] eager(1024 meta)    vs manual ref  -> split w/ stale needCoreNum WRONG?
#   [x  ] graph(short meta)   vs manual ref  -> same divergence via NPUGraph
#
# Read:
#   flashDecodeFlag toggles 0->1, AND the short/1024-meta cells show large diff
#       => CONFIRMED: graph bakes the wrong (or stale) split plan from warmup.
#   short-meta cell large but 1024-meta cell ~1e-1
#       => non-split is the broken path; graph must be forced to split.
#   1024-meta cell large but short-meta cell ~1e-1
#       => split tiling staleness (needCoreNum/launchBlockDim) is the bug.
#   all cells ~1e-1
#       => split == non-split numerically; decode bug is elsewhere (batch
#          padding of the capture size, or CANN prefill corruption).
#
# Usage:
#   python test_fa3_graph_flashdecode_diverge.py
#   KV_SHORT=16 KV_LONG=16384 python test_fa3_graph_flashdecode_diverge.py

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

# fa_metadata::TilingOffset(causal) == MASK_BYTES (2048x2048 causal mask).
MASK_BYTES = 2048 * 2048

# FAInferTilingData byte offsets (see csrc/.../tilingdata.h).  All are relative
# to the tiling base (metaBase + TilingOffset(causal)), so reads add MASK_BYTES.
FIELDS = {
    "maxKvSeqlen": (24, "u32"),
    "batch": (32, "u32"),
    "maxNumBlocksPerBatch": (36, "u32"),
    "firstBatchTaskNum": (40, "u32"),
    "totalTaskNum": (44, "u32"),
    "flashDecodeFlag": (168, "u32"),
    "numSplits": (172, "u32"),
    # split tiling — non-zero only when flashDecodeFlag=1 (splitBN2S1GS2 ran)
    "splitLseTotalSize": (144, "u64"),
    "splitOTotalSize": (152, "u64"),
    "totalSplitNodeNum": (160, "u32"),
    "needCoreNum": (164, "u32"),
}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _make_meta(batch: int, cache_seqlens: torch.Tensor, maxk: int):
    # maxk controls ONLY the block-table row stride (maxNumBlocksPerBatch);
    # cache_seqlens controls maxKvSeqlen -> flashDecodeFlag + split tiling.
    # Production holds maxk = max_blocks_per_seq*block_size (stride==table width)
    # and varies cache_seqlens with the warmup/actual batch.
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


def _dump_meta(meta: torch.Tensor, label: str):
    torch.npu.synchronize()
    cpu = meta.cpu()
    parts = []
    for name, (off, kind) in FIELDS.items():
        if kind == "u32":
            val = int(cpu[MASK_BYTES + off: MASK_BYTES + off + 4].view(torch.int32)[0].item())
        else:  # u64
            val = int(cpu[MASK_BYTES + off: MASK_BYTES + off + 8].view(torch.int64)[0].item())
        parts.append(f"{name}={val}")
    print(f"[dump] {label:<22} " + "  ".join(parts))


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
        window_size=(-1, -1),  # production full_graph_fa3 always passes this
        scheduler_metadata=meta,
    )


def manual_ref(q, k, v, seq_len):
    """float32 GQA attention over the first seq_len KVs (identity paged).

    Decode has one query token at position seq_len-1, so causal == full
    attention over all seq_len cached KVs.  Independent of CANN V1 and FA3.
    """
    nblk = _ceil_div(seq_len, BLOCK_SIZE)
    k_flat = k[:nblk].reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:seq_len].float()
    v_flat = v[:nblk].reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:seq_len].float()
    k_g = k_flat.repeat_interleave(GROUP, dim=1)  # (seq_len, H, D)
    v_g = v_flat.repeat_interleave(GROUP, dim=1)
    q_f = q.float()  # (1, H, D)
    scores = torch.einsum("bhd,thd->bht", q_f, k_g) * SCALE  # (1, H, seq_len)
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bht,thd->bhd", attn, v_g)  # (1, H, D)


def main():
    kv_short = int(os.environ.get("KV_SHORT", "16"))
    kv_long = int(os.environ.get("KV_LONG", "16384"))

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    print("=" * 72)
    print(f"C13 flashDecodeFlag divergence  short={kv_short}  long={kv_long}")
    print("=" * 72)

    # ---- Part 1: dump tiling fields across the flash-decode threshold --------
    # maxk held at kv_long so maxNumBlocksPerBatch stays == 128 (stride fixed);
    # only cache_seqlens moves, isolating its effect on flashDecodeFlag.
    for kv in (kv_short, 1024, kv_long):
        seq = torch.tensor([kv], dtype=torch.int32).npu()
        meta = _make_meta(1, seq, kv_long)
        _dump_meta(meta, f"batch=1 kv={kv}")

    # ---- Part 2: precision of short/1024/long meta on a long replay ----------
    num_blocks = _ceil_div(kv_long, BLOCK_SIZE)
    q = torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 1], dtype=torch.int32).npu()
    page_table = torch.arange(num_blocks, dtype=torch.int32).npu().unsqueeze(0)

    ref = manual_ref(q, k, v, kv_long)
    torch.npu.synchronize()

    seq_short = torch.tensor([kv_short], dtype=torch.int32).npu()
    seq_1024 = torch.tensor([1024], dtype=torch.int32).npu()
    seq_long = torch.tensor([kv_long], dtype=torch.int32).npu()

    meta_short = _make_meta(1, seq_short, kv_long)  # flashDecodeFlag=0 (non-split)
    meta_1024 = _make_meta(1, seq_1024, kv_long)    # split, needCoreNum=16
    meta_long = _make_meta(1, seq_long, kv_long)    # split, needCoreNum=20

    # (a) eager with long meta -> SPLIT path (correct reference)
    out_eager_long = _run_fa3(q, k, v, seq_long, cu_q, page_table, meta_long)
    # (b) eager with short meta -> NON-SPLIT (flashDecodeFlag=0 from short warmup)
    out_eager_short = _run_fa3(q, k, v, seq_long, cu_q, page_table, meta_short)
    # (c) eager with 1024 meta -> SPLIT but needCoreNum baked from a different len
    out_eager_1024 = _run_fa3(q, k, v, seq_long, cu_q, page_table, meta_1024)
    torch.npu.synchronize()

    # (d) true NPUGraph path: capture with short meta, replay with long seq
    seq_buf = torch.tensor([kv_short], dtype=torch.int32).npu()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = _run_fa3(q, k, v, seq_buf, cu_q, page_table, meta_short)
    torch.npu.synchronize()
    seq_buf.copy_(seq_long)
    graph.replay()
    torch.npu.synchronize()
    out_graph_short = captured.clone()

    print("-" * 72)
    print(f"[ok ] eager(long meta)   vs manual ref : "
          f"{_max_abs_diff(out_eager_long, ref):.6f}")
    print(f"[x  ] eager(short meta)  vs manual ref : "
          f"{_max_abs_diff(out_eager_short, ref):.6f}")
    print(f"[x  ] eager(1024 meta)   vs manual ref : "
          f"{_max_abs_diff(out_eager_1024, ref):.6f}")
    print(f"[x  ] graph(short meta)  vs manual ref : "
          f"{_max_abs_diff(out_graph_short, ref):.6f}")

    print("-" * 72)
    print("Read:")
    print("  flashDecodeFlag toggles 0->1 AND short/1024 cells LARGE (>~1e-1)")
    print("      => CONFIRMED: graph bakes the wrong (or stale) split plan.")
    print("  short cell LARGE but 1024 cell ~1e-1 => non-split path is broken;")
    print("      force graph to split.")
    print("  1024 cell LARGE but short cell ~1e-1 => split tiling staleness")
    print("      (needCoreNum/launchBlockDim mismatch) is the bug.")
    print("  all cells ~1e-1 => split == non-split numerically; look elsewhere.")
    print("-" * 72)


if __name__ == "__main__":
    main()
