# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C27: multi-seq SPLIT-path (flashDecodeFlag=1) with a STALE split
# schedule — the one combination C13 and C26 left open.
#
# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------
# C13 (batch=1) proved that baking scheduler_metadata from WARMUP cache_seqlens
# (short=16 -> flag=0, 1024 -> flag=1/needCoreNum=16, 16384 -> flag=1/needCoreNum=20)
# is numerically harmless for SINGLE-sequence decode: all cells 0.000121.
#
# C26 proved the kernel is correct for VARIABLE seq_lens, but its split cells
# (V1/V4/V8 "long") all baked cache_seqlens == the ACTUAL replay seq_lens
# (fresh schedule).  The only "bake-short" cell (G4) baked flag=0 (non-split).
#
# The production graph bakes metadata ONCE from the WARMUP cache_seqlens and
# replays with a DIFFERENT (variable) batch.  If the warmup max KV length >= 1024,
# flashDecodeFlag=1 and splitBN2S1GS2() bakes a per-core split plan (coreInfo[]/
# needCoreNum) for the WARMUP lengths.  Replaying with different lengths then runs
# the SPLIT path against a STALE split plan — a combination NO prior experiment
# exercised (C13 only did batch=1; C26's split cells used fresh == actual lengths).
#
# This experiment bakes metadata with cache_seqlens = LONG (flag=1, split) and
# replays with DIFFERENT variable lengths, against the CPU float64 reference.
#
# Read:
#   any cell > ~0.05 => SPLIT path with stale plan is WRONG for multi-seq
#                       (this is the production decode bug if warmup maxKv>=1024).
#   all ~1e-3        => split stale-plan is harmless; bug is elsewhere.
#
# Usage:
#   python test_fa3_graph_varlen_split.py

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
    return float((a.float().cpu() - b.float().cpu()).abs().max().item())


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


def cpu_ref_f64(q_cpu, k_cpu, v_cpu, block_table_cpu, seqlens):
    outs = []
    for b, seq_len in enumerate(seqlens):
        nblk = _ceil_div(seq_len, BLOCK_SIZE)
        ids = block_table_cpu[b, :nblk].tolist()
        k_flat = torch.cat([k_cpu[i] for i in ids], dim=0)[:seq_len]
        v_flat = torch.cat([v_cpu[i] for i in ids], dim=0)[:seq_len]
        k_g = k_flat.repeat_interleave(GROUP, dim=1)
        v_g = v_flat.repeat_interleave(GROUP, dim=1)
        q_f = q_cpu[b]
        scores = torch.einsum("hd,thd->ht", q_f, k_g) * SCALE
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("ht,thd->hd", attn, v_g)
        outs.append(out)
    return torch.stack(outs, dim=0)


def _mk_block_table(batch, width, seqlens, num_blocks_pool, seed):
    g = torch.Generator().manual_seed(seed)
    bt = torch.full((batch, width), -1, dtype=torch.int32)
    for b, s in enumerate(seqlens):
        nblk = _ceil_div(s, BLOCK_SIZE)
        ids = torch.randperm(num_blocks_pool, generator=g, dtype=torch.int32)[:nblk]
        bt[b, :nblk] = ids
    return bt


def _cell(tag, seqlens, bake_lens, num_blocks_pool, width, maxk):
    batch = len(seqlens)
    print("-" * 72)
    print(f"[{tag}] batch={batch} call={seqlens} bake={bake_lens}")

    q = torch.randn(batch, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(num_blocks_pool, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)

    block_table = _mk_block_table(batch, width, seqlens, num_blocks_pool, 7)
    seq_call = torch.tensor(seqlens, dtype=torch.int32).npu()
    seq_bake = torch.tensor(bake_lens, dtype=torch.int32).npu()
    cu_q = torch.arange(batch + 1, dtype=torch.int32).npu()

    meta = _make_meta(batch, seq_bake, maxk)
    out = _run_fa3(q, k, v, seq_call, cu_q, block_table.npu(), meta)
    torch.npu.synchronize()

    ref = cpu_ref_f64(q.cpu().double(), k.cpu().double(), v.cpu().double(),
                      block_table, seqlens)
    for b in range(batch):
        d = _max_abs_diff(out[b], ref[b])
        flag = "  <-- KERNEL WRONG" if d > 0.05 else ""
        print(f"    row {b}  seq={seqlens[b]:5d}  : kernel-vs-cpuF64 = {d:.6f}{flag}")


def main():
    width = 128
    num_blocks_pool = 128
    maxk = width * BLOCK_SIZE

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    torch.manual_seed(0)
    print("=" * 72)
    print(f"C27 varlen+split   width={width}  maxk={maxk}")
    print("=" * 72)

    # Control: bake SHORT (flag=0 non-split) + call variable -> C26 G4 already OK.
    _cell("ctrl bake-short", [512, 1024, 2048, 4096], [16] * 4,
          num_blocks_pool, width, maxk)
    # The untested cells: bake LONG (flag=1 split) + call DIFFERENT variable.
    _cell("split bake-1024", [512, 1024, 2048, 4096], [1024] * 4,
          num_blocks_pool, width, maxk)
    _cell("split bake-2048", [512, 1024, 2048, 4096], [2048] * 4,
          num_blocks_pool, width, maxk)
    _cell("split bake-4096", [512, 1024, 2048, 4096], [4096] * 4,
          num_blocks_pool, width, maxk)
    # Bake long, call a mix that is ALL shorter than the bake (max mismatch).
    _cell("split bake-4096 call-short", [256, 512, 768, 1024], [4096] * 4,
          num_blocks_pool, width, maxk)

    torch.npu.synchronize()
    print("-" * 72)
    print("Read:")
    print("  any split cell >0.05 => SPLIT stale-plan WRONG for multi-seq (production bug).")
    print("  all ~1e-3            => split stale-plan harmless; look elsewhere.")
    print("-" * 72)


if __name__ == "__main__":
    main()
