# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C19: characterize the "first-call is a scattered table" trigger.
#
# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------
# C18 nailed the trigger shape: perm_B (2nd randperm) is WRONG only when the
# FIRST FA3 call of the process used perm_A (1st randperm):
#   S1 perm_B first-call            -> 0.000351  (clean)
#   S3 perm_B after warmup=identity -> 0.000351  (clean)
#   S4 perm_B after [perm_A,identity]-> 0.149113 (WRONG)
# and swap right after the errored perm_B is still correct -> the error does NOT
# spread downstream (so it is not a naive "write garbage into workspace" chain).
# The only shared state across calls is the thread_local GetCachedWorkspace
# buffer (same size -> same bytes, never cleared).  meta is read-only.
#
# C19 answers:
#   (Q1) determinism: is perm_A -> perm_B wrong every single time?
#   (Q2) self vs other: does perm_A -> perm_A (same scattered table twice) fail?
#   (Q3) any scattered first-call: does perm_C -> perm_B fail too (not perm_A-specific)?
#   (Q4) spatial pattern: WHICH heads/rows are wrong in the perm_B output?  A
#        full-slate vs a few heads vs a few rows tells us the mechanism.
#
# Read:
#   Q1 flaky              => a race (stream/pipe), not a deterministic state bug.
#   Q2 perm_A->perm_A OK  => needs TWO DIFFERENT scattered tables.
#   Q2 perm_A->perm_A BAD => "first call scattered" poisons all later scattered.
#   Q3 perm_C->perm_B BAD => generic scattered first-call, not perm_A-specific.
#   Q4 contiguous 4-head groups wrong => a whole kv-head's blocks mis-read.
#   Q4 all heads uniformly off  => a global softmax/lse normalization error.
#
# Usage:
#   python test_fa3_graph_scatter_chain.py
#   KV=2048 python test_fa3_graph_scatter_chain.py

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


def _per_head_diff(out, ref):
    """(1, H, D) -> list of H max-abs-diffs over D."""
    d = (out.float() - ref.float()).abs().amax(dim=-1)  # (1, H)
    return d[0].cpu().tolist()


def _head_group_map():
    """head -> kv-head (for interpreting which heads went wrong)."""
    return [h // GROUP for h in range(NUM_HEADS)]


def main():
    kv = int(os.environ.get("KV", "2048"))
    num_blocks_pool = 64
    nblk = _ceil_div(kv, BLOCK_SIZE)
    assert nblk <= num_blocks_pool, "KV too long for the physical pool"

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    torch.manual_seed(0)

    print("=" * 72)
    print(f"C19 scattered-chain   kv={kv}  nblk={nblk}")
    print("=" * 72)

    q = torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(num_blocks_pool, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 1], dtype=torch.int32).npu()
    seq = torch.tensor([kv], dtype=torch.int32).npu()

    perm_a_cpu = torch.randperm(num_blocks_pool, dtype=torch.int32)   # 1st randperm
    perm_b_cpu = torch.randperm(num_blocks_pool, dtype=torch.int32)   # 2nd
    perm_c_cpu = torch.randperm(num_blocks_pool, dtype=torch.int32)   # 3rd
    identity_cpu = torch.arange(num_blocks_pool, dtype=torch.int32)

    perm_a = perm_a_cpu.npu().unsqueeze(0)
    perm_b = perm_b_cpu.npu().unsqueeze(0)
    perm_c = perm_c_cpu.npu().unsqueeze(0)
    identity = identity_cpu.npu().unsqueeze(0)

    ref_b = manual_ref_blocktable(q, k, v, perm_b_cpu, kv)
    ref_a = manual_ref_blocktable(q, k, v, perm_a_cpu, kv)

    # ---- Q1: determinism of perm_A -> perm_B (3 repeats, fresh meta each) ----
    print("-" * 72)
    print("[Q1] perm_A -> perm_B, 3 repeats (fresh meta each):")
    for r in range(3):
        meta = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                          num_blocks_pool * BLOCK_SIZE)
        _run_fa3(q, k, v, seq, cu_q, perm_a, meta)   # first call scattered
        out = _run_fa3(q, k, v, seq, cu_q, perm_b, meta)
        torch.npu.synchronize()
        print(f"  repeat {r}: diff = {_max_abs_diff(out, ref_b):.6f}")

    # ---- Q2: perm_A -> perm_A (same scattered table twice) ----
    print("-" * 72)
    print("[Q2] perm_A -> perm_A (same scattered table twice):")
    meta = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                      num_blocks_pool * BLOCK_SIZE)
    _run_fa3(q, k, v, seq, cu_q, perm_a, meta)
    out = _run_fa3(q, k, v, seq, cu_q, perm_a, meta)
    torch.npu.synchronize()
    print(f"  diff = {_max_abs_diff(out, ref_a):.6f}")

    # ---- Q3: perm_C -> perm_B (generic scattered first-call) ----
    print("-" * 72)
    print("[Q3] perm_C -> perm_B (third scattered table as first call):")
    meta = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                      num_blocks_pool * BLOCK_SIZE)
    _run_fa3(q, k, v, seq, cu_q, perm_c, meta)
    out = _run_fa3(q, k, v, seq, cu_q, perm_b, meta)
    torch.npu.synchronize()
    print(f"  diff = {_max_abs_diff(out, ref_b):.6f}")

    # ---- Q4: spatial pattern of the errored perm_B output ----
    print("-" * 72)
    print("[Q4] per-head diff of perm_B under perm_A -> perm_B:")
    meta = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                      num_blocks_pool * BLOCK_SIZE)
    _run_fa3(q, k, v, seq, cu_q, perm_a, meta)
    out_bad = _run_fa3(q, k, v, seq, cu_q, perm_b, meta)
    torch.npu.synchronize()
    hd = _per_head_diff(out_bad, ref_b)
    groups = _head_group_map()
    print("  head -> kv-head : max-abs-diff")
    for h, d in enumerate(hd):
        flag = "  <-- WRONG" if d > 0.01 else ""
        print(f"    {h:2d}  ->  kv{groups[h]}      : {d:.6f}{flag}")

    # control: clean perm_B per-head diff (identity first-call)
    print("-" * 72)
    print("[Q4c] per-head diff of perm_B under identity -> perm_B (control):")
    meta = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                      num_blocks_pool * BLOCK_SIZE)
    _run_fa3(q, k, v, seq, cu_q, identity, meta)
    out_good = _run_fa3(q, k, v, seq, cu_q, perm_b, meta)
    torch.npu.synchronize()
    hd_good = _per_head_diff(out_good, ref_b)
    for h, d in enumerate(hd_good):
        flag = "  <-- WRONG" if d > 0.01 else ""
        print(f"    {h:2d}  ->  kv{groups[h]}      : {d:.6f}{flag}")

    torch.npu.synchronize()

    print("-" * 72)
    print("Read:")
    print("  Q1 flaky            => race, not deterministic state.")
    print("  Q2 OK               => needs TWO DIFFERENT scattered tables.")
    print("  Q2 BAD              => 'first scattered' poisons all scattered.")
    print("  Q3 BAD              => generic scattered first-call, not perm_A-specific.")
    print("  Q4 whole 4-head groups wrong => a whole kv-head's blocks mis-read.")
    print("  Q4 all heads uniformly off   => global softmax/lse error.")
    print("-" * 72)


if __name__ == "__main__":
    main()
