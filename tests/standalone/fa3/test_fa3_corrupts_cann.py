# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Standalone repro: does FA3 (flash_attn_with_kvcache) NPUGraph replay corrupt
# a subsequent CANN V1 attention (npu_fused_infer_attention_score) GRAPH replay?
#
# ---------------------------------------------------------------------------
# Hypothesis under test
# ---------------------------------------------------------------------------
# FA3's SplitFuse::FAInfer kernel UNCONDITIONALLY uses FFTS cross-core sync
# (AscendC::SetSyncBaseAddr + CrossCoreSetFlag/WaitFlag) whose base address is
# the DEVICE-GLOBAL C2C control region (rtGetC2cCtrlAddr).  CANN V1 long-sequence
# prefill also splits KV across AI cores and uses the same FFTS machinery.
#
#   * eager prefill: the CANN runtime manages/resets the shared FFTS region at
#     kernel boundaries, so FA3's leftover FFTS state is cleared before prefill.
#   * graph-replayed prefill (task-group): replay bypasses that per-kernel
#     management, so FA3's FFTS state leaks into prefill -> wrong result.
#
# This is EXACTLY why the first version of this repro (eager prefill) did NOT
# reproduce: prefill must be graph-replayed too.  This version captures the
# CANN V1 prefill as a graph_task_group and replays it.
#
# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------
#   [0a] prefill graph-replay == eager prefill (sanity: graph replay works)
#   [0]  prefill graph-replay deterministic (twice)
#   [1]  baseline = clean prefill graph-replay
#   [2]  FA3 EAGER        -> prefill graph-replay (expect ~0)
#   [3]  FA3 GRAPH CAPTURE -> prefill graph-replay
#   [4]  FA3 GRAPH REPLAY xN -> prefill graph-replay (expect > 0 = reproduced)
#   [4b] FA3 GRAPH REPLAY xN -> prefill EAGER      (expect ~0 = control)
#
#   If [4] > threshold and [4b] ~ 0, the hypothesis holds.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   python test_fa3_corrupts_cann.py
#
#   PREILL_SEQLEN=16384 DECODE_KV_SEQLEN=16384 N_REPLAYS=3 \
#       python test_fa3_corrupts_cann.py
#
#   PREILL_SEQLEN controls whether the CANN V1 prefill splits across cores
#   (and thus uses FFTS).  If 16384 does not reproduce, try 32768.

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
    # actual_seq_lengths / actual_seq_lengths_kv MUST be Python lists (CPU):
    # npu_fused_infer_attention_score reads these scalars on the host to decide
    # whether to split KV across cores.  Passing device tensors makes it do a
    # copy_stream sync (LocalScalarDenseNpu) which is illegal during graph
    # capture ("stream is captured").  vllm-ascend passes seq_lens_list (a
    # Python list) for exactly this reason.
    actual_q = [seqlen]  # cumulative WITHOUT leading 0
    actual_kv = [seqlen]  # per-sequence KV length
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
        sparse_mode=0,  # dense, non-causal -> no atten_mask
        out=[out_buf, lse_buf],
    )


def run_prefill_eager(inp):
    """CANN V1 prefill EAGER (control: runtime manages FFTS)."""
    q, *_ = inp
    out = torch.empty(q.shape[0], NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    lse = torch.empty(1, dtype=torch.float32).npu()
    _fia_out(inp, out, lse)
    torch.npu.synchronize()
    return out.clone()


def capture_prefill_graph(inp):
    """CANN V1 prefill GRAPH (task-group), matching vllm-ascend full_graph_fia."""
    q, *_ = inp
    out_buf = torch.empty(q.shape[0], NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    lse_buf = torch.empty(1, dtype=torch.float32).npu()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
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
    # FA3 paged layout: (num_blocks, block_size, num_kv_heads, head_size).
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


def capture_fa3_graph(decode_inp):
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        _ = run_fa3_decode(decode_inp)
    torch.npu.synchronize()
    return graph


def main():
    prefill_seqlen = int(os.environ.get("PREILL_SEQLEN", "16384"))
    kv_seqlen = int(os.environ.get("DECODE_KV_SEQLEN", "16384"))
    n_replays = int(os.environ.get("N_REPLAYS", "3"))

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    print("=" * 72)
    print(f"prefill_seqlen={prefill_seqlen}  decode_kv_seqlen={kv_seqlen}  "
          f"n_replays={n_replays}")
    print("=" * 72)

    prefill_inp = make_prefill_inputs(prefill_seqlen)
    decode_inp = make_decode_inputs(kv_seqlen)

    # Capture the CANN V1 prefill as a task-group graph.
    prefill_graph, prefill_out, _, handle = capture_prefill_graph(prefill_inp)

    # ---- [0a] sanity: graph-replayed prefill == eager prefill --------------
    eager_ref = run_prefill_eager(prefill_inp)
    g0 = replay_prefill(prefill_graph, prefill_out)
    print(f"[0a] prefill graph-vs-eager max_abs_diff = "
          f"{_max_abs_diff(g0, eager_ref):.6f} (should be ~0)")

    # ---- [0] sanity: prefill graph replay is deterministic ----------------
    r0 = replay_prefill(prefill_graph, prefill_out)
    r1 = replay_prefill(prefill_graph, prefill_out)
    print(f"[0]  prefill graph self-consistency   = "
          f"{_max_abs_diff(r0, r1):.6f} (should be ~0)")

    baseline = r1
    print(f"[1]  baseline prefill graph-replay captured "
          f"(shape={tuple(baseline.shape)})")

    # ---- [2] FA3 EAGER, then prefill graph-replay -------------------------
    _ = run_fa3_decode(decode_inp)
    torch.npu.synchronize()
    r2 = replay_prefill(prefill_graph, prefill_out)
    print(f"[2]  after FA3 EAGER       -> prefill GRAPH = "
          f"{_max_abs_diff(baseline, r2):.6f} (should be ~0)")

    # ---- [3] FA3 GRAPH CAPTURE, then prefill graph-replay ------------------
    fa3_graph = capture_fa3_graph(decode_inp)
    r3 = replay_prefill(prefill_graph, prefill_out)
    print(f"[3]  after FA3 CAPTURE     -> prefill GRAPH = "
          f"{_max_abs_diff(baseline, r3):.6f}")

    # ---- [4] FA3 GRAPH REPLAY xN, then prefill graph-replay ----------------
    for _ in range(n_replays):
        fa3_graph.replay()
    torch.npu.synchronize()
    r4 = replay_prefill(prefill_graph, prefill_out)
    d4 = _max_abs_diff(baseline, r4)
    d4_mean = _mean_abs_diff(baseline, r4)
    print(f"[4]  after FA3 REPLAY x{n_replays} -> prefill GRAPH = "
          f"{d4:.6f} (mean {d4_mean:.6f})")

    # ---- [4b] FA3 GRAPH REPLAY xN, then prefill EAGER (control) ------------
    for _ in range(n_replays):
        fa3_graph.replay()
    torch.npu.synchronize()
    r4b = run_prefill_eager(prefill_inp)
    d4b = _max_abs_diff(baseline, r4b)
    print(f"[4b] after FA3 REPLAY x{n_replays} -> prefill EAGER  = "
          f"{d4b:.6f} (should be ~0)")

    # ---- verdict -----------------------------------------------------------
    print("-" * 72)
    threshold = 1e-2  # well above bf16 rounding for identical inputs
    if d4 > threshold and d4b < threshold:
        print("VERDICT: REPRODUCED.")
        print("  FA3 graph replay corrupted the CANN V1 prefill GRAPH replay")
        print("  but NOT the eager prefill -> FFTS state leaks only across")
        print("  graph replay (matches vllm-ascend 'eager ok, graph broken').")
    elif d4 <= threshold:
        print("VERDICT: NOT reproduced.")
        print("  Try a larger PREILL_SEQLEN / DECODE_KV_SEQLEN so the kernels")
        print("  actually split KV across cores (and thus use FFTS).")
    else:
        print("VERDICT: INCONCLUSIVE.")
        print("  Eager prefill was also affected; the corruption is not specific")
        print("  to graph replay. Investigate FFTS sharing in eager mode too.")
    print("-" * 72)


if __name__ == "__main__":
    main()
