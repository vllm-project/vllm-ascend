# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment A (v2): where does FA3 decode precision break?
#
# v1 found that FA3 eager output was IDENTICAL for seq=16 vs seq=4096 under
# CAUSAL=1, which is impossible for a correct decode (q attends cache_seqlens
# KVs).  This v2 adds two things to pin it down:
#
#   1. window_size=(-1,-1)  -- production full_graph_fa3 always passes this
#      (no sliding window); v1 omitted it, so the FA3 default window may have
#      masked the causal behaviour.
#   2. a float32 manual attention reference (einsum GQA), independent of both
#      CANN V1 and FA3, to establish the CORRECT output for a given cache_seqlens.
#
# ---------------------------------------------------------------------------
# What it prints
# ---------------------------------------------------------------------------
#   manual short vs manual long      -> does seq affect the CORRECT output?
#   FA3 short vs FA3 long            -> does seq affect FA3 output? (must mirror)
#   FA3 short vs manual short        -> FA3 correctness @ short
#   FA3 long  vs manual long         -> FA3 correctness @ long
#   FA3(long+short meta) vs FA3(long)-> does scheduler_metadata depend on seq?
#   graph grown vs FA3 long          -> graph-replay correctness (grown seq)
#   graph short vs FA3 short         -> graph-replay positive control
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   python test_fa3_schedmeta_depends_on_seqlens.py
#   KV_SHORT=16 KV_LONG=4096 CAUSAL=1 python test_fa3_schedmeta_depends_on_seqlens.py
#   KV_SHORT=16 KV_LONG=4096 CAUSAL=0 python test_fa3_schedmeta_depends_on_seqlens.py

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


def _make_meta(kv_seqlen: int, cache_seqlens: torch.Tensor, cu_q: torch.Tensor, causal: bool):
    return _get_scheduler_metadata(
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=kv_seqlen,
        num_heads_q=NUM_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        headdim=HEAD_SIZE,
        cache_seqlens=cache_seqlens,
        qkv_dtype=DTYPE,
        cu_seqlens_q=cu_q,
        page_size=BLOCK_SIZE,
        causal=causal,
    )


def _run_fa3(q, k, v, cache_seqlens, cu_q, page_table, meta, causal):
    return _fa3_kvcache(
        q,
        k,
        v,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_q,
        max_seqlen_q=1,
        softmax_scale=SCALE,
        causal=causal,
        window_size=(-1, -1),  # production full_graph_fa3 always passes this
        scheduler_metadata=meta,
    )


def manual_ref(q, k, v, seq_len):
    """float32 GQA attention over the first seq_len KVs (paged -> flat).

    decode has a single query token at position seq_len-1, so causal == full
    attention over all seq_len cached KVs.  This is the CORRECT output for the
    given cache_seqlens, independent of CANN V1 and FA3.
    """
    nblk = _ceil_div(seq_len, BLOCK_SIZE)
    k_flat = k[:nblk].reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:seq_len].float()
    v_flat = v[:nblk].reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:seq_len].float()
    # GQA: expand each KV head to GROUP query heads.
    k_g = k_flat.repeat_interleave(GROUP, dim=1)  # (seq_len, H, D)
    v_g = v_flat.repeat_interleave(GROUP, dim=1)
    q_f = q.float()  # (1, H, D)
    scores = torch.einsum("bhd,thd->bht", q_f, k_g) * SCALE  # (1, H, seq_len)
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bht,thd->bhd", attn, v_g)  # (1, H, D)


def main():
    short = int(os.environ.get("KV_SHORT", "16"))
    long = int(os.environ.get("KV_LONG", "4096"))
    causal = os.environ.get("CAUSAL", "1") == "1"

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    print("=" * 72)
    print(f"short={short}  long={long}  causal={causal}  window_size=(-1,-1)")
    print("=" * 72)

    num_blocks = _ceil_div(long, BLOCK_SIZE)

    q = torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 1], dtype=torch.int32).npu()
    page_table = torch.arange(num_blocks, dtype=torch.int32).npu().unsqueeze(0)

    seq_short = torch.tensor([short], dtype=torch.int32).npu()
    seq_long = torch.tensor([long], dtype=torch.int32).npu()

    meta_short = _make_meta(short, seq_short, cu_q, causal)
    meta_long = _make_meta(long, seq_long, cu_q, causal)

    # ---- correct references ----
    ref_short = manual_ref(q, k, v, short)
    ref_long = manual_ref(q, k, v, long)
    torch.npu.synchronize()

    # ---- FA3 eager ----
    fa3_short = _run_fa3(q, k, v, seq_short, cu_q, page_table, meta_short, causal)
    fa3_long = _run_fa3(q, k, v, seq_long, cu_q, page_table, meta_long, causal)
    fa3_long_shortmeta = _run_fa3(q, k, v, seq_long, cu_q, page_table, meta_short, causal)
    # decode: causal and non-causal MUST agree (query attends all cached KVs;
    # there is no future KV to mask).  A mismatch => FA3 causal position wrong.
    fa3_long_noncausal = _run_fa3(
        q, k, v, seq_long, cu_q, page_table,
        _make_meta(long, seq_long, cu_q, False), False,
    )
    torch.npu.synchronize()

    # ---- graph replay ----
    seq_buf = torch.tensor([short], dtype=torch.int32).npu()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = _run_fa3(q, k, v, seq_buf, cu_q, page_table, meta_short, causal)
    torch.npu.synchronize()

    seq_buf.copy_(seq_long)
    graph.replay()
    torch.npu.synchronize()
    graph_long = captured.clone()

    seq_buf.copy_(seq_short)
    graph.replay()
    torch.npu.synchronize()
    graph_short = captured.clone()

    # ---- diffs ----
    print(f"[ref] manual short vs manual long            : "
          f"{_max_abs_diff(ref_short, ref_long):.6f}")
    print(f"[fa3] FA3 short  vs FA3 long                 : "
          f"{_max_abs_diff(fa3_short, fa3_long):.6f}")
    print(f"[ok ] FA3 short  vs manual short             : "
          f"{_max_abs_diff(fa3_short, ref_short):.6f}")
    print(f"[ok ] FA3 long   vs manual long              : "
          f"{_max_abs_diff(fa3_long, ref_long):.6f}")
    print(f"[ok ] FA3 noncausal long vs manual long      : "
          f"{_max_abs_diff(fa3_long_noncausal, ref_long):.6f}")
    print(f"[causal] FA3 long causal vs noncausal        : "
          f"{_max_abs_diff(fa3_long, fa3_long_noncausal):.6f}")
    print(f"[meta] FA3(long+short meta) vs FA3(long)      : "
          f"{_max_abs_diff(fa3_long_shortmeta, fa3_long):.6f}")
    print(f"[graph] grown-seq replay vs FA3(long)         : "
          f"{_max_abs_diff(graph_long, fa3_long):.6f}")
    print(f"[graph] short-seq replay vs FA3(short)        : "
          f"{_max_abs_diff(graph_short, fa3_short):.6f}")

    print("-" * 72)
    print("Read:")
    print("  [ref]  >0  => seq genuinely changes the correct output (expected)")
    print("  [fa3]  should mirror [ref]; ~0 while [ref]>0 => FA3 ignores cache_seqlens")
    print("  [ok ]  ~1e-2..1e-1 => FA3 matches correct output (bf16 vs fp32)")
    print("         large       => FA3 output is WRONG")
    print("  [meta] ~0 => scheduler_metadata does not depend on seq values")
    print("  [causal] ~0 => causal==noncausal (decode MUST be); large => causal pos bug")
    print("  [graph]~0 => graph replay is correct; large => graph capture/replay bug")
    print("-" * 72)


if __name__ == "__main__":
    main()
