# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C17: does the FIRST (warm-up) block table affect the correctness
# of subsequent FA3 calls?  (a cross-call "state" effect)
#
# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------
# C14/C15/C16 flip-flop in a way that no block-table *content* rule explains:
#   C14 (no warm-up):            identity WRONG, reorder correct
#   C15 (warm-up = reorder):     identity correct, reorder WRONG, swap half-wrong
#   C16 (warm-up = identity):    EVERYTHING correct
# The only clean correlate is the WARM-UP table.  That smells like a cross-call
# state: whatever the first call bakes (metadata? workspace? HW cache?) affects
# later calls.
#
# Production graph capture feeds block_table_buf = ALL-ZEROS (attention_v1.py
# creates torch.zeros(...) and captures FA3 on it), then replay refreshes to the
# real reordered table.  So "warm-up = zeros" is the production case.
#
# C17 isolates it: for each warm-up table {identity, reorder, zeros}, build a
# FRESH scheduler_metadata (resets any metadata-side state), warm up once, then
# measure identity/reorder/swap.  If fresh metadata does NOT erase the effect,
# the state lives in the FA3 op's workspace / kernel / hardware, not metadata.
#
# Read:
#   warm-up row matters (reorder row wrong, others right) but fresh meta ~same
#       => state is in workspace/kernel/HW, NOT metadata.
#   warm-up row matters AND fresh meta changes it
#       => metadata is being mutated by the kernel.
#   zeros warm-up -> reorder WRONG
#       => REPRODUCED production (capture-zeros / replay-reorder) precision bug.
#
# Usage:
#   python test_fa3_graph_warmup_state.py
#   KV=2048 python test_fa3_graph_warmup_state.py

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


def main():
    kv = int(os.environ.get("KV", "2048"))
    num_blocks_pool = 64
    nblk = _ceil_div(kv, BLOCK_SIZE)
    assert nblk <= num_blocks_pool, "KV too long for the physical pool"

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    torch.manual_seed(0)

    print("=" * 72)
    print(f"C17 warm-up state   kv={kv}  nblk={nblk}")
    print("=" * 72)

    q = torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(num_blocks_pool, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 1], dtype=torch.int32).npu()
    seq = torch.tensor([kv], dtype=torch.int32).npu()

    # tables
    identity = torch.arange(num_blocks_pool, dtype=torch.int32).npu().unsqueeze(0)
    zeros = torch.zeros(1, num_blocks_pool, dtype=torch.int32).npu()
    perm = torch.randperm(num_blocks_pool, dtype=torch.int32).npu().unsqueeze(0)  # reorder

    swp_cpu = torch.arange(num_blocks_pool, dtype=torch.int32)  # swap block 0 <-> 63
    swp_cpu[0] = num_blocks_pool - 1
    swp_cpu[num_blocks_pool - 1] = 0
    swap = swp_cpu.npu().unsqueeze(0)

    test_tables = [("identity", identity), ("reorder", perm), ("swap", swap)]

    for warmup_name, warmup_table in (("identity", identity),
                                      ("reorder", perm),
                                      ("zeros", zeros)):
        meta = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                          num_blocks_pool * BLOCK_SIZE)  # FRESH metadata
        _run_fa3(q, k, v, seq, cu_q, warmup_table, meta)  # warm-up (not measured)
        torch.npu.synchronize()

        line = f"[warmup={warmup_name:<8}] "
        for name, table in test_tables:
            ref = manual_ref_blocktable(q, k, v, table[0], kv)
            out = _run_fa3(q, k, v, seq, cu_q, table, meta)
            line += f" {name}={_max_abs_diff(out, ref):.6f}"
        print(line)

    torch.npu.synchronize()

    print("-" * 72)
    print("Read:")
    print("  warm-up row matters (reorder row wrong), fresh meta does NOT fix it")
    print("      => state lives in workspace/kernel/HW, NOT metadata.")
    print("  fresh meta DOES change the row => metadata is mutated by the kernel.")
    print("  zeros warm-up -> reorder WRONG  => reproduced production precision bug.")
    print("-" * 72)


if __name__ == "__main__":
    main()
