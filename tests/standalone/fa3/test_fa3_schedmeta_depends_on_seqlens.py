# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment A: does FA3's scheduler_metadata depend on the VALUES of
# cache_seqlens (per-sequence KV lengths), or only on max_seqlen_k?
#
# ---------------------------------------------------------------------------
# Why this matters
# ---------------------------------------------------------------------------
# vllm-ascend captures the FA3 decode graph with a DUMMY cache_seqlens (the
# warmup batch, length 1..num_tokens) and builds scheduler_metadata from it.
# At replay, update_graph_params overwrites cache_seqlens with the REAL length
# (which grows to max_model_len), but does NOT rebuild scheduler_metadata.  If
# get_scheduler_metadata reads the per-sequence lengths to pre-compute tiling,
# that stale scheduler_metadata produces wrong decode output.
#
# ---------------------------------------------------------------------------
# What this script does (batch=1 decode, same q/k/v throughout)
# ---------------------------------------------------------------------------
#   [1] eager  short seq + short meta          (baseline A)
#   [2] eager  long  seq + long  meta          (baseline B = correct reference)
#   [3] eager  long  seq + short meta          (direct test, no graph involved)
#   [4] graph  capture(short seq + short meta) -> overwrite seq to long -> replay
#                                               (production path)
#   [5] graph  capture(short seq + short meta) -> replay with UNCHANGED seq
#                                               (positive control)
#
# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------
#   * [3] != [2]  -> scheduler_metadata DEPENDS on cache_seqlens values.
#                    The decode bug is a root cause (stale scheduler_metadata).
#   * [4] != [2] while [5] == [1] -> production graph path is broken by the
#     cache_seqlens growth relative to the captured scheduler_metadata, while
#     graph replay itself is fine.
#   * [4] == [2]  -> scheduler_metadata only depends on max_seqlen_k; the
#     decode bug is elsewhere (block_table/cu_seqlens_q refresh, or the
#     flash_attn_npu.patch GetCachedOutputTensor aliasing).
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   python test_fa3_schedmeta_depends_on_seqlens.py
#   KV_SHORT=128 KV_LONG=4096 CAUSAL=1 python test_fa3_schedmeta_depends_on_seqlens.py

import os
from importlib import util as importlib_util

import torch
import torch_npu

_HAS_FA3 = False
_fa3_kvcache = None
_get_scheduler_metadata = None

for _mod_name in ("flash_attn_npu_v3", "flash_attn_npu_3"):
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
        scheduler_metadata=meta,
    )


def main():
    short = int(os.environ.get("KV_SHORT", "128"))
    long = int(os.environ.get("KV_LONG", "512"))
    causal = os.environ.get("CAUSAL", "1") == "1"

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    print("=" * 72)
    print(f"short={short}  long={long}  causal={causal}")
    print("=" * 72)

    num_blocks = _ceil_div(long, BLOCK_SIZE)

    # ---- shared data: one query token, paged K/V with enough blocks for LONG --
    q = torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 1], dtype=torch.int32).npu()
    page_table = torch.arange(num_blocks, dtype=torch.int32).npu().unsqueeze(0)

    seq_short = torch.tensor([short], dtype=torch.int32).npu()
    seq_long = torch.tensor([long], dtype=torch.int32).npu()

    meta_short = _make_meta(short, seq_short, cu_q, causal)
    meta_long = _make_meta(long, seq_long, cu_q, causal)

    # ---- [1] eager short + short meta ----
    out1 = _run_fa3(q, k, v, seq_short, cu_q, page_table, meta_short, causal)
    torch.npu.synchronize()

    # ---- [2] eager long + long meta (correct reference) ----
    out2 = _run_fa3(q, k, v, seq_long, cu_q, page_table, meta_long, causal)
    torch.npu.synchronize()

    # ---- [3] eager long + short meta (direct test, no graph) ----
    out3 = _run_fa3(q, k, v, seq_long, cu_q, page_table, meta_short, causal)
    torch.npu.synchronize()

    d_short_vs_long = _max_abs_diff(out1, out2)
    d_stale_meta = _max_abs_diff(out3, out2)
    print(f"[1] eager short+shortmeta vs [2] eager long+longmeta : "
          f"max_abs_diff={d_short_vs_long:.6f}")
    print(f"[3] eager long+shortmeta  vs [2] eager long+longmeta : "
          f"max_abs_diff={d_stale_meta:.6f}")

    # ---- [4] graph: capture(short+shortmeta) -> overwrite seq to long -> replay
    seq_buf = torch.tensor([short], dtype=torch.int32).npu()  # overwritten below
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = _run_fa3(q, k, v, seq_buf, cu_q, page_table, meta_short, causal)
    torch.npu.synchronize()

    seq_buf.copy_(seq_long)  # real decode: cache_seqlens grows
    graph.replay()
    torch.npu.synchronize()
    out4 = captured.clone()
    d_graph_grown = _max_abs_diff(out4, out2)
    print(f"[4] graph grown-seq replay  vs [2] eager long+longmeta : "
          f"max_abs_diff={d_graph_grown:.6f}")

    # ---- [5] graph positive control: replay with UNCHANGED short seq ----
    seq_buf.copy_(seq_short)  # restore
    graph.replay()
    torch.npu.synchronize()
    out5 = captured.clone()
    d_graph_short = _max_abs_diff(out5, out1)
    print(f"[5] graph short-seq replay  vs [1] eager short+shortmeta: "
          f"max_abs_diff={d_graph_short:.6f}")

    print("-" * 72)
    threshold = 1e-2
    stale = d_stale_meta > threshold
    grown = d_graph_grown > threshold
    control_ok = d_graph_short <= threshold

    if stale:
        print("VERDICT: scheduler_metadata DEPENDS on cache_seqlens VALUES.")
        print("  -> [3] already diverged; graph capture is not required to")
        print("     reproduce.  Stale scheduler_metadata is a root cause.")
    else:
        print("VERDICT: scheduler_metadata does NOT depend on cache_seqlens")
        print("         values at these lengths (only on max_seqlen_k).")

    if not control_ok:
        print("  -> [5] control itself diverged: graph replay is broken even")
        print("     with unchanged seq; investigate capture/replay first.")
    elif grown:
        print("  -> [4] graph replay with grown seq diverges while [5] control")
        print("     matches: production decode is broken by cache_seqlens growth")
        print("     relative to the captured scheduler_metadata.")
    else:
        print("  -> graph replay matches in BOTH cases: decode bug is elsewhere")
        print("     (block_table/cu_seqlens_q refresh, or the flash_attn_npu.patch")
        print("     GetCachedOutputTensor aliasing).")
    print("-" * 72)


if __name__ == "__main__":
    main()
