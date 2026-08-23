# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Standalone repro: does capturing/replaying an FA3 (flash_attn_with_kvcache)
# NPUGraph corrupt the CANN graph_task_group machinery that a LATER CANN V1
# prefill graph depends on?
#
# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------
# In vllm-ascend, the CANN V1 prefill graph is captured with
# graph_task_group_begin/End and, on every replay, re-bound via
# graph_task_update_begin/End BEFORE aclgraph.replay().  The FA3 decode graph
# (a PyTorch CustomOp invisible to task-group) is captured in the SAME graph
# pool.  That FA3 capture/replay leaves residue in the CANN task-group runtime
# state, which corrupts the LATER prefill graph's task_group_update + replay.
#
# Earlier standalone attempts used a bare graph.replay() for the prefill and
# never reproduced, because the task_group_update path (the actual corruption
# carrier) was missing.  This version replicates the full
#   capture -> graph_task_update_begin/End -> replay
# cycle, matching vllm-ascend full_graph_fia + update_graph_params.
#
# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------
#   [1] capture CLEAN prefill task-group graph; update+replay -> baseline
#   [2] capture the FA3 decode NPUGraph (no task-group wrapper)  <- polluter
#   [3] capture SECOND prefill task-group graph; update+replay -> "polluted"
#   [4] compare baseline vs polluted
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   python test_fa3_corrupts_cann.py
#
#   PREILL_SEQLEN=16384 DECODE_KV_SEQLEN=16384 REPLAY_FA3=1 python ...

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
            _mod = __import__(_mod_name, fromlist=["flash_attn_with_kvcache", "get_scheduler_metadata"])
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
SWA_INT_MAX = 2147483647


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean().item())


# ---------------------------------------------------------------------------
# CANN V1 prefill (paged, TND, non-causal)
# ---------------------------------------------------------------------------
def make_prefill_inputs(seqlen: int):
    num_blocks = _ceil_div(seqlen, BLOCK_SIZE)
    q = torch.randn(seqlen, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    block_table = torch.arange(num_blocks, dtype=torch.int32).npu().unsqueeze(0)
    actual_q = [seqlen]  # MUST be Python list (see LocalScalarDenseNpu issue)
    actual_kv = [seqlen]
    return q, k, v, block_table, actual_q, actual_kv


def get_workspace(inp):
    q, k, v, block_table, actual_q, actual_kv = inp
    return torch_npu._npu_fused_infer_attention_score_get_max_workspace(
        query=q,
        key=k,
        value=v,
        atten_mask=None,
        block_table=block_table,
        input_layout="TND",
        block_size=BLOCK_SIZE,
        actual_seq_lengths=actual_q,
        actual_seq_lengths_kv=actual_kv,
        num_key_value_heads=NUM_KV_HEADS,
        num_heads=NUM_HEADS,
        sparse_mode=0,
        pre_tokens=SWA_INT_MAX,
        next_tokens=SWA_INT_MAX,
        scale=SCALE,
    )


def _fia_out(inp, workspace, out_buf, lse_buf):
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
        sparse_mode=0,
        workspace=workspace,
        out=[out_buf, lse_buf],
    )


def capture_prefill_graph(inp, pool=None):
    """Capture the prefill FIA as a task-group graph (matches full_graph_fia)."""
    q, *_ = inp
    workspace = get_workspace(inp)
    out_buf = torch.empty(q.shape[0], NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    lse_buf = torch.empty(1, dtype=torch.float32).npu()
    graph = torch.npu.NPUGraph()
    ctx = torch.npu.graph(graph, pool=pool) if pool is not None else torch.npu.graph(graph)
    with ctx:
        stream = torch_npu.npu.current_stream()
        torch.npu.graph_task_group_begin(stream)
        _fia_out(inp, workspace, out_buf, lse_buf)
        handle = torch.npu.graph_task_group_end(stream)
    torch.npu.synchronize()
    return graph, handle, workspace


def replay_prefill(graph, handle, inp, workspace):
    """Full task-group replay: graph_task_update_begin/End, then graph.replay."""
    q, *_ = inp
    out_buf = torch.empty(q.shape[0], NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    lse_buf = torch.empty(1, dtype=torch.float32).npu()
    stream = torch_npu.npu.current_stream()
    torch.npu.graph_task_update_begin(stream, handle)
    _fia_out(inp, workspace, out_buf, lse_buf)
    torch.npu.graph_task_update_end(stream)
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    return out_buf.clone()


# ---------------------------------------------------------------------------
# FA3 decode
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
    graph = torch.npu.NPUGraph()
    ctx = torch.npu.graph(graph, pool=pool) if pool is not None else torch.npu.graph(graph)
    with ctx:
        _ = run_fa3_decode(decode_inp)
    return graph


def main():
    prefill_seqlen = int(os.environ.get("PREILL_SEQLEN", "16384"))
    kv_seqlen = int(os.environ.get("DECODE_KV_SEQLEN", "16384"))
    replay_fa3 = os.environ.get("REPLAY_FA3", "0") == "1"

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    print("=" * 72)
    print(f"prefill_seqlen={prefill_seqlen}  decode_kv_seqlen={kv_seqlen}  "
          f"replay_fa3={replay_fa3}")
    print("=" * 72)

    prefill_inp = make_prefill_inputs(prefill_seqlen)
    decode_inp = make_decode_inputs(kv_seqlen)
    pool = torch.npu.graph_pool_handle()

    # ---- [1] clean prefill graph (captured BEFORE FA3) ----------------------
    g_clean, h_clean, ws_clean = capture_prefill_graph(prefill_inp, pool=pool)
    baseline = replay_prefill(g_clean, h_clean, prefill_inp, ws_clean)
    print(f"[1] clean prefill task-group update+replay captured "
          f"(shape={tuple(baseline.shape)})")

    # ---- [2] FA3 decode graph capture (+optional replay) --------------------
    fa3_graph = capture_fa3_graph(decode_inp, pool=pool)
    if replay_fa3:
        for _ in range(3):
            fa3_graph.replay()
        torch.npu.synchronize()
    print(f"[2] FA3 decode graph captured{' and replayed x3' if replay_fa3 else ''}")

    # ---- [3] second prefill graph (captured AFTER FA3) ----------------------
    g_poll, h_poll, ws_poll = capture_prefill_graph(prefill_inp, pool=pool)
    polluted = replay_prefill(g_poll, h_poll, prefill_inp, ws_poll)
    d = _max_abs_diff(baseline, polluted)
    d_mean = _mean_abs_diff(baseline, polluted)
    print(f"[3] prefill captured AFTER FA3 -> diff vs clean = "
          f"{d:.6f} (mean {d_mean:.6f})")

    # ---- [4] control: clean graph update+replay again (still clean?) -------
    baseline2 = replay_prefill(g_clean, h_clean, prefill_inp, ws_clean)
    print(f"[4] clean graph re-replay self-consistency = "
          f"{_max_abs_diff(baseline, baseline2):.6f}")

    print("-" * 72)
    threshold = 1e-2
    if d > threshold:
        print("VERDICT: REPRODUCED.")
        print("  FA3 capture/replay corrupted the LATER prefill task-group's")
        print("  graph_task_update + replay -> output diverges.")
    else:
        print("VERDICT: NOT reproduced.")
        print("  Next levers: larger PREILL_SEQLEN, REPLAY_FA3=1, or run the")
        print("  E48/E53 diagnostics inside vllm-ascend to pin capture-vs-replay.")
    print("-" * 72)


if __name__ == "__main__":
    main()
