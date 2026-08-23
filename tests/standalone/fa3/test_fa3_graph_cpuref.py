# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C25: settle kernel-vs-reference with an UNambiguous float64 CPU
# reference (no NPU einsum/softmax, which may itself misbehave).
#
# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------
# C24 showed the FA3 kernel is SELF-consistent (batch4 == batch1, 0.000000) yet
# disagreed with my hand-written NPU float32 reference (manual_ref_batch) by
# 0.35-0.67.  Two candidate explanations:
#   (a) the kernel is wrong for short/split cases, or
#   (b) my NPU-side reference (einsum+softmax in float32 on NPU) is buggy.
#
# C25 removes ambiguity: compute the reference on CPU in float64 (torch CPU,
# fully deterministic, no NPU ops) and compare against the kernel output.
#
# Tests each (seq, bake) cell at batch=4:
#   seq=512  bake=[16]*4   (non-split, short)
#   seq=512  bake=[512]*4  (non-split, short)
#   seq=2048 bake=[16]*4   (non-split, long)
#   seq=2048 bake=[2048]*4 (split,     long)
#
# Read:
#   kernel ~= cpu-f64 ref (<=~0.01) => kernel CORRECT; NPU ref was buggy.
#   kernel != cpu-f64 ref (>~0.05)  => kernel WRONG for that (seq,bake) cell.
#
# Usage:
#   python test_fa3_graph_cpuref.py

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
    """float64 CPU reference — unambiguous, no NPU ops."""
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


def _mk_block_table(batch, width, seqlen, num_blocks_pool, seed):
    g = torch.Generator().manual_seed(seed)
    nblk = _ceil_div(seqlen, BLOCK_SIZE)
    bt = torch.full((batch, width), -1, dtype=torch.int32)
    for b in range(batch):
        ids = torch.randperm(num_blocks_pool, generator=g, dtype=torch.int32)[:nblk]
        bt[b, :nblk] = ids
    return bt


def _cell(tag, seqlen, bake_len, batch, num_blocks_pool, width, maxk):
    print("-" * 72)
    print(f"[{tag}] seq={seqlen} bake={bake_len} batch={batch}")

    q = torch.randn(batch, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(num_blocks_pool, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)

    block_table = _mk_block_table(batch, width, seqlen, num_blocks_pool, 7)
    seq_bake = torch.full((batch,), bake_len, dtype=torch.int32).npu()
    seq_call = torch.full((batch,), seqlen, dtype=torch.int32).npu()
    cu_q = torch.arange(batch + 1, dtype=torch.int32).npu()

    meta = _make_meta(batch, seq_bake, maxk)
    out = _run_fa3(q, k, v, seq_call, cu_q, block_table.npu(), meta)
    torch.npu.synchronize()

    # CPU float64 reference from the SAME inputs.
    q_cpu = q.cpu().double()
    k_cpu = k.cpu().double()
    v_cpu = v.cpu().double()
    ref = cpu_ref_f64(q_cpu, k_cpu, v_cpu, block_table, [seqlen] * batch)

    for b in range(batch):
        d = _max_abs_diff(out[b], ref[b])
        flag = "  <-- KERNEL WRONG" if d > 0.05 else ""
        print(f"    row {b}  : kernel-vs-cpuF64 = {d:.6f}{flag}")


def main():
    width = 128
    num_blocks_pool = 128
    maxk = width * BLOCK_SIZE
    batch = 4

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    torch.manual_seed(0)
    print("=" * 72)
    print(f"C25 CPU-f64 ref   batch={batch}  width={width}  maxk={maxk}")
    print("=" * 72)

    _cell("N1 non-split short (bake16)", 512, 16, batch, num_blocks_pool, width, maxk)
    _cell("N2 non-split short (bake512)", 512, 512, batch, num_blocks_pool, width, maxk)
    _cell("N3 non-split long (bake16)", 2048, 16, batch, num_blocks_pool, width, maxk)
    _cell("S  split long (bake2048)", 2048, 2048, batch, num_blocks_pool, width, maxk)

    torch.npu.synchronize()
    print("-" * 72)
    print("Read:")
    print("  kernel ~= cpu-f64 (<=0.01) => kernel CORRECT; NPU ref was buggy.")
    print("  kernel != cpu-f64 (>0.05)  => kernel WRONG for that cell.")
    print("-" * 72)


if __name__ == "__main__":
    main()
