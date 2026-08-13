# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Standalone repro: does FA3 (flash_attn_with_kvcache) NPUGraph replay corrupt
# a subsequent CANN V1 attention (npu_fused_infer_attention_score)?
#
# ---------------------------------------------------------------------------
# Hypothesis under test
# ---------------------------------------------------------------------------
# FA3's SplitFuse::FAInfer kernel UNCONDITIONALLY uses FFTS cross-core sync
# (AscendC::SetSyncBaseAddr + CrossCoreSetFlag/WaitFlag) whose base address is
# the DEVICE-GLOBAL C2C control region returned by rtGetC2cCtrlAddr.  CANN V1
# long-sequence prefill (npu_fused_infer_attention_score) also splits the KV
# sequence across AI cores and uses the same FFTS machinery.
#
#   * eager mode: the CANN runtime manages/resets that shared FFTS region at
#     kernel boundaries, so FA3 and CANN V1 do not corrupt each other.
#   * graph mode: NPUGraph replay bypasses that per-kernel management, so the
#     FFTS state FA3 leaves behind leaks into the next CANN V1 prefill.
#
# This script reproduces exactly that mixed-eager/graph sequence and measures
# whether the CANN V1 prefill result changes after an FA3 decode graph replay.
#
# ---------------------------------------------------------------------------
# What it does
# ---------------------------------------------------------------------------
#   0. sanity: the CANN V1 prefill result is deterministic (same input, twice).
#   1. baseline  = clean CANN V1 prefill (nothing ran before it).
#   2. after FA3 EAGER     -> prefill again, compare to baseline (expect ~0).
#   3. after FA3 GRAPH CAPTURE -> prefill again, compare to baseline.
#   4. after FA3 GRAPH REPLAY (N times) -> prefill again, compare to baseline.
#
#   If (4) diverges significantly from baseline while (2) stays ~0, the
#   hypothesis holds: FA3 graph replay corrupts CANN V1 prefill.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   python test_fa3_corrupts_cann.py
#
#   # tune the sequence lengths / replay count
#   PREILL_SEQLEN=8192 DECODE_KV_SEQLEN=8192 N_REPLAYS=3 \
#       python test_fa3_corrupts_cann.py
#
#   PREILL_SEQLEN controls whether CANN V1 prefill actually splits across cores
#   (and therefore uses FFTS).  If 8192 does not reproduce, try larger values
#   (e.g. 16384, 32768).  A short prefill (e.g. 256) should NOT reproduce,
#   because it stays on a single core and never touches FFTS.

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
# CANN V1 prefill (npu_fused_infer_attention_score, paged, TND, non-causal)
# ---------------------------------------------------------------------------
def make_prefill_inputs(seqlen: int):
    num_blocks = _ceil_div(seqlen, BLOCK_SIZE)
    q = torch.randn(seqlen, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    # CANN V1 paged layout is (num_blocks, num_kv_heads, block_size, head_size).
    k = torch.randn(num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    block_table = torch.arange(num_blocks, dtype=torch.int32).npu().unsqueeze(0)
    # actual_seq_lengths  : cumulative WITHOUT leading 0.
    # actual_seq_lengths_kv: per-sequence KV length.
    actual_q = torch.tensor([seqlen], dtype=torch.int32).npu()
    actual_kv = torch.tensor([seqlen], dtype=torch.int32).npu()
    return q, k, v, block_table, actual_q, actual_kv


def run_prefill(inp):
    q, k, v, block_table, actual_q, actual_kv = inp
    out = torch.empty(q.shape[0], NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    lse = torch.empty(1, dtype=torch.float32).npu()
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
        sparse_mode=0,  # dense, non-causal -> no atten_mask needed
        out=[out, lse],
    )
    torch.npu.synchronize()
    return out.clone()


# ---------------------------------------------------------------------------
# FA3 decode (flash_attn_with_kvcache, 1 query token, long cached KV)
# ---------------------------------------------------------------------------
def make_decode_inputs(kv_seqlen: int):
    num_blocks = _ceil_div(kv_seqlen, BLOCK_SIZE)
    q = torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    # FA3 paged layout is (num_blocks, block_size, num_kv_heads, head_size).
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


def main():
    prefill_seqlen = int(os.environ.get("PREILL_SEQLEN", "8192"))
    kv_seqlen = int(os.environ.get("DECODE_KV_SEQLEN", "8192"))
    n_replays = int(os.environ.get("N_REPLAYS", "3"))

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    print("=" * 72)
    print(f"prefill_seqlen={prefill_seqlen}  decode_kv_seqlen={kv_seqlen}  "
          f"n_replays={n_replays}")
    print("=" * 72)

    prefill_inp = make_prefill_inputs(prefill_seqlen)
    decode_inp = make_decode_inputs(kv_seqlen)

    # ---- 0. sanity: prefill is deterministic under clean state -------------
    p0 = run_prefill(prefill_inp)
    p1 = run_prefill(prefill_inp)
    self_diff = _max_abs_diff(p0, p1)
    print(f"[0] prefill self-consistency max_abs_diff = {self_diff:.6f} "
          f"(should be ~0)")

    # ---- 1. baseline -------------------------------------------------------
    baseline = p1
    print(f"[1] baseline prefill captured (shape={tuple(baseline.shape)})")

    # ---- 2. FA3 EAGER, then prefill ---------------------------------------
    _ = run_fa3_decode(decode_inp)
    torch.npu.synchronize()
    p_eager = run_prefill(prefill_inp)
    d_eager = _max_abs_diff(baseline, p_eager)
    print(f"[2] after FA3 EAGER        : prefill diff = {d_eager:.6f} "
          f"(should be ~0)")

    # ---- 3. FA3 GRAPH CAPTURE (no replay yet), then prefill ----------------
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        _ = run_fa3_decode(decode_inp)
    torch.npu.synchronize()
    p_capture = run_prefill(prefill_inp)
    d_capture = _max_abs_diff(baseline, p_capture)
    print(f"[3] after FA3 CAPTURE      : prefill diff = {d_capture:.6f}")

    # ---- 4. FA3 GRAPH REPLAY (N times), then prefill -----------------------
    for i in range(n_replays):
        graph.replay()
    torch.npu.synchronize()
    p_replay = run_prefill(prefill_inp)
    d_replay = _max_abs_diff(baseline, p_replay)
    d_replay_mean = _mean_abs_diff(baseline, p_replay)
    print(f"[4] after FA3 REPLAY x{n_replays}  : prefill diff = {d_replay:.6f} "
          f"(mean {d_replay_mean:.6f})")

    # ---- verdict -----------------------------------------------------------
    print("-" * 72)
    threshold = 1e-2  # well above bf16 rounding for identical inputs
    if d_eager < threshold and d_replay > threshold:
        print("VERDICT: REPRODUCED.")
        print("  FA3 graph replay changed the CANN V1 prefill result while FA3")
        print("  eager did not -> FA3 NPUGraph replay corrupts prefill (FFTS).")
    elif d_replay <= threshold:
        print("VERDICT: NOT reproduced.")
        print("  Try a larger PREILL_SEQLEN / DECODE_KV_SEQLEN so the kernels")
        print("  actually split KV across cores (and thus use FFTS).")
    else:
        print("VERDICT: INCONCLUSIVE.")
        print("  FA3 EAGER already shifted the prefill result; the corruption")
        print("  is not specific to graph replay. Investigate FFTS sharing in")
        print("  eager mode too.")
    print("-" * 72)


if __name__ == "__main__":
    main()
