# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C26: VARIABLE seq_lens multi-batch decode, validated against the
# CPU float64 reference (the one gap C25 left open).
#
# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------
# C25 proved the FA3 kernel is correct for UNIFORM seq_lens (all [512] or all
# [2048]).  But production decode has VARIABLE seq_lens per request (each
# request's prefill length differs, e.g. [512, 1024, 2048, 4096]).  C22/C23
# tested variable lengths but with a buggy NPU-side reference, so the variable-
# length multi-batch case was never validated against a correct reference.
#
# C26 closes that gap: variable seq_lens, batch = 1/2/4/8, both bake=actual
# (eager-equivalent, flag depends on max) and bake=short (graph non-split),
# all against the CPU float64 reference.
#
# Read:
#   any cell kernel-vs-cpuF64 > ~0.05 => FA3 kernel WRONG for variable seq_lens
#                                      (this is the production decode bug).
#   all ~1e-3                        => kernel correct for variable lengths;
#                                      look elsewhere (glue / capture).
#
# Usage:
#   python test_fa3_graph_varlen.py

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


def _cell(tag, seqlens, bake_short, num_blocks_pool, width, maxk):
    batch = len(seqlens)
    print("-" * 72)
    print(f"[{tag}] batch={batch} seqlens={seqlens} bake={'short' if bake_short else 'actual'}")

    q = torch.randn(batch, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(num_blocks_pool, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)

    block_table = _mk_block_table(batch, width, seqlens, num_blocks_pool, 7)
    seq_call = torch.tensor(seqlens, dtype=torch.int32).npu()
    seq_bake = torch.full((batch,), 16, dtype=torch.int32).npu() if bake_short else seq_call
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
    print(f"C26 variable-seqlen   width={width}  maxk={maxk}")
    print("=" * 72)

    # Control: single long (flag likely split if bd large enough).
    _cell("V1 single long", [2048], False, num_blocks_pool, width, maxk)
    # Variable multi-batch.
    _cell("V2 var", [512, 2048], False, num_blocks_pool, width, maxk)
    _cell("V4 var", [512, 1024, 2048, 4096], False, num_blocks_pool, width, maxk)
    _cell("V8 var", [256, 512, 768, 1024, 1536, 2048, 2560, 3072], False, num_blocks_pool, width, maxk)
    # Variable but all short (flag=0 non-split) and all long (flag likely split).
    _cell("V4 short", [128, 256, 384, 512], False, num_blocks_pool, width, maxk)
    _cell("V4 long", [1024, 2048, 3072, 4096], False, num_blocks_pool, width, maxk)
    # Graph-equivalent: bake short [16], call variable.
    _cell("G4 bake-short", [512, 1024, 2048, 4096], True, num_blocks_pool, width, maxk)

    torch.npu.synchronize()
    print("-" * 72)
    print("Read:")
    print("  any cell >0.05 => FA3 kernel WRONG for variable seq_lens (production bug).")
    print("  all ~1e-3      => kernel correct; look in glue/capture code.")
    print("-" * 72)


if __name__ == "__main__":
    main()
