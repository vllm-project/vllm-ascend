# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Standalone repro: does capturing an FA3 (flash_attn_with_kvcache) NPUGraph
# corrupt a SUBSEQUENTLY captured CANN V1 (npu_fused_infer_attention_score)
# graph_task_group?
#
# ---------------------------------------------------------------------------
# Hypothesis under test
# ---------------------------------------------------------------------------
# FA3 is a PyTorch CustomOp invisible to the CANN task-group mechanism.  When
# vllm-ascend captures the FA3 decode graph with torch.npu.graph() it does NOT
# wrap it in graph_task_group_begin/End.  That produces an "empty task group"
# which corrupts CANN runtime state for the NEXT graph capture -- i.e. the
# prefill CANN V1 graph captured with graph_task_group_begin/End.  The corrupted
# prefill graph then replays to wrong output.
#
# This mirrors EXACTLY the comment in vllm-ascend full_graph_fa3:
#   "an empty task group (FA3 not captured by CANN) corrupts CANN runtime
#    state for subsequent graph captures (e.g. the prefill CANN V1 graph),
#    breaking prefill accuracy."
#
# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------
#   [1] capture a CLEAN prefill graph_task_group, replay -> baseline
#   [2] capture the FA3 decode NPUGraph (no task-group wrapper)  <- the polluter
#   [3] capture a SECOND prefill graph_task_group, replay -> "polluted"
#   [4] compare baseline vs polluted.  If they differ, capture-order pollution
#       is proven (FA3 capture corrupts the later CANN V1 capture).
#
# Both graphs share ONE graph pool, as vllm-ascend's ACLGraphWrapper does
# (current_platform.get_global_graph_pool()).
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   python test_fa3_corrupts_cann.py
#
#   PREILL_SEQLEN=16384 DECODE_KV_SEQLEN=16384 python test_fa3_corrupts_cann.py

import os
from importlib import util as importlib_util

import torch
import torch_npu

# ---------------------------------------------------------------------------
# FA3 import (name differs across flash-attention-npu versions)
# ---------------------------------------------------------------------------
_HAS_FA3 = False
_fa3_kvcache = None
_get_scheduler_metadata = None

for _mod_name in ("flash_attn_npu_v3", "flash_attn_npu_3"):
    if importlib_util.find_spec(_mod_name) is not None:
        try:
            _mod = __import__(_mod_name, fromlist=["flash_attn_with_kvcache", "get_scheduler_metadata"])
            _fa3_kvcache = _mod.flash_attn_with_kvcache
            _get_scheduler_metadata = _mod.get_scheduler_metadata
            _HAS_FA3 = True
            print(f"[import] FA3 loaded from {_mod_name}")
            break
        except (ImportError, AttributeError) as exc:
            print(f"[import] {_mod_name} found but failed: {exc}")

if not _HAS_FA3:
    raise SystemExit(
        "flash_attn_with_kvcache (FA3) is not installed. "
        "Install flash-attention-npu first."
    )

# ---------------------------------------------------------------------------
# Shape / dtype knobs
# ---------------------------------------------------------------------------
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


def _mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean().item())


# ---------------------------------------------------------------------------
# CANN V1 prefill inputs (paged, TND, non-causal)
# ---------------------------------------------------------------------------
def make_prefill_inputs(seqlen: int):
    num_blocks = _ceil_div(seqlen, BLOCK_SIZE)
    q = torch.randn(seqlen, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    # CANN V1 paged layout: (num_blocks, num_kv_heads, block_size, head_size).
    k = torch.randn(num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    block_table = torch.arange(num_blocks, dtype=torch.int32).npu().unsqueeze(0)
    # MUST be Python lists: npu_fused_infer_attention_score reads these scalars
    # on host; device tensors would trigger a copy_stream sync (illegal during
    # capture).
    actual_q = [seqlen]
    actual_kv = [seqlen]
    return q, k, v, block_table, actual_q, actual_kv


def _fia_out(inp, out_buf, lse_buf):
    q, k, v, block_table, actual_q, actual_kv = inp
    torch_npu.npu_fused_infer_attention_score.out(
        query=q,
        key=k,
        value=v,
        block_table=block_table,
        input_layout="TND",
        block_size=BLOCK_SIZE,
        actual_seq_lengths=actual_q,
        actual_seq_lengths_kv=actual_kv,
        num_key_value_heads=NUM_KV_HEADS,
        num_heads=NUM_HEADS,
        scale=SCALE,
        sparse_mode=0,  # dense, non-causal
        out=[out_buf, lse_buf],
    )


def capture_prefill_graph(inp, pool=None):
    """CANN V1 prefill as a graph_task_group, matching vllm-ascend full_graph_fia."""
    q, *_ = inp
    out_buf = torch.empty(q.shape[0], NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    lse_buf = torch.empty(1, dtype=torch.float32).npu()
    graph = torch.npu.NPUGraph()
    ctx = torch.npu.graph(graph, pool=pool) if pool is not None else torch.npu.graph(graph)
    with ctx:
        stream = torch_npu.npu.current_stream()
        torch.npu.graph_task_group_begin(stream)
        _fia_out(inp, out_buf, lse_buf)
        handle = torch.npu.graph_task_group_end(stream)
    torch.npu.synchronize()
    return graph, out_buf, lse_buf, handle


def replay_prefill(graph, out_buf):
    graph.replay()
    torch.npu.synchronize()
    return out_buf.clone()


# ---------------------------------------------------------------------------
# FA3 decode (flash_attn_with_kvcache, 1 query token, long cached KV)
# ---------------------------------------------------------------------------
def make_decode_inputs(kv_seqlen: int):
    num_blocks = _ceil_div(kv_seqlen, BLOCK_SIZE)
    q = torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    cache_seqlens = torch.tensor([kv_seqlen], dtype=torch.int32).npu()
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32).npu()
    page_table = torch.arange(num_blocks, dtype=torch.int32).npu().unsqueeze(0)

    metadata = _get_scheduler_metadata(
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=kv_seqlen,
        num_heads_q=NUM_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        headdim=HEAD_SIZE,
        cache_seqlens=cache_seqlens,
        qkv_dtype=DTYPE,
        cu_seqlens_q=cu_seqlens_q,
        page_size=BLOCK_SIZE,
        causal=False,
    )
    return q, k, v, cache_seqlens, cu_seqlens_q, page_table, metadata


def run_fa3_decode(inp):
    q, k, v, cache_seqlens, cu_seqlens_q, page_table, metadata = inp
    return _fa3_kvcache(
        q,
        k,
        v,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        softmax_scale=SCALE,
        causal=False,
        scheduler_metadata=metadata,
    )


def capture_fa3_graph(decode_inp, pool=None):
    """FA3 decode NPUGraph, NO graph_task_group wrapper (the polluter)."""
    graph = torch.npu.NPUGraph()
    ctx = torch.npu.graph(graph, pool=pool) if pool is not None else torch.npu.graph(graph)
    with ctx:
        _ = run_fa3_decode(decode_inp)
    # NOTE: deliberately do NOT synchronize here -- we want any empty-task-group
    # residue to leak into the NEXT capture, exactly as vllm-ascend sees.
    return graph


def main():
    prefill_seqlen = int(os.environ.get("PREILL_SEQLEN", "16384"))
    kv_seqlen = int(os.environ.get("DECODE_KV_SEQLEN", "16384"))
    share_pool = os.environ.get("SHARE_POOL", "1") == "1"

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    print("=" * 72)
    print(f"prefill_seqlen={prefill_seqlen}  decode_kv_seqlen={kv_seqlen}  "
          f"share_pool={share_pool}")
    print("=" * 72)

    prefill_inp = make_prefill_inputs(prefill_seqlen)
    decode_inp = make_decode_inputs(kv_seqlen)

    pool = torch.npu.graph_pool_handle() if share_pool else None

    # ---- [1] clean prefill graph, captured BEFORE any FA3 capture ----------
    g_clean, out_clean, _, _ = capture_prefill_graph(prefill_inp, pool=pool)
    baseline = replay_prefill(g_clean, out_clean)
    print(f"[1] clean prefill graph replay captured "
          f"(shape={tuple(baseline.shape)})")

    # ---- [2] FA3 decode graph capture (the suspected polluter) -------------
    fa3_graph = capture_fa3_graph(decode_inp, pool=pool)
    print("[2] FA3 decode graph captured (no task-group wrapper)")

    # ---- [3] second prefill graph, captured AFTER FA3 ----------------------
    g_polluted, out_polluted, _, _ = capture_prefill_graph(prefill_inp, pool=pool)
    polluted = replay_prefill(g_polluted, out_polluted)
    d = _max_abs_diff(baseline, polluted)
    d_mean = _mean_abs_diff(baseline, polluted)
    print(f"[3] prefill graph captured AFTER FA3 -> diff vs clean = "
          f"{d:.6f} (mean {d_mean:.6f})")

    # ---- [4] control: replay the clean graph again (still clean?) ----------
    baseline2 = replay_prefill(g_clean, out_clean)
    print(f"[4] clean graph re-replay self-consistency = "
          f"{_max_abs_diff(baseline, baseline2):.6f}")

    # ---- verdict -----------------------------------------------------------
    print("-" * 72)
    threshold = 1e-2
    if d > threshold:
        print("VERDICT: REPRODUCED.")
        print("  Capturing the FA3 decode graph corrupted the LATER CANN V1")
        print("  prefill graph capture -> its replay output diverges.  This is")
        print("  the 'empty task group corrupts CANN runtime state' mechanism.")
    else:
        print("VERDICT: NOT reproduced.")
        print("  The FA3 capture did NOT change a later prefill capture.  Next")
        print("  levers: (a) larger PREILL_SEQLEN, (b) also replay the FA3 graph")
        print("  before capturing the second prefill graph, (c) add the")
        print("  graph_task_update_begin/End replay path used by vllm-ascend.")
    print("-" * 72)


if __name__ == "__main__":
    main()
