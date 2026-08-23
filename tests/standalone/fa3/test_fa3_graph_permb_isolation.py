# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C18: is perm_B wrong on its OWN (a table-content kernel bug), or
# is it wrong only because of a cross-call "pollution chain" in the workspace?
#
# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------
# C17's three rows were bit-identical and all correct — but that was because its
# "reorder" test table was perm_A (seed's 1st randperm), NOT perm_B (2nd).
# Comparing C14/C15/C17 by table identity:
#   perm_A (1st randperm)  -> correct everywhere (C14, C17)
#   perm_B (2nd randperm)  -> WRONG 0.16 (C15 only place it was tested)
#   swap  [63,1..15,..,0]  -> 0.057 in C15, 0.000464 in C17  <-- SAME table!
# The swap discrepancy proves a table is NOT wrong purely by its content: C15's
# swap ran AFTER perm_B (which errored), C17's swap never saw perm_B.
#
# Two hypotheses to separate:
#   (H1) perm_B is a table-content kernel bug: perm_B wrong even as a clean
#        first call with fresh metadata.
#   (H2) pollution chain: the errored perm_B call leaves bad data in the shared
#        workspace, which corrupts the NEXT call (swap).  perm_B itself may be
#        wrong only under a specific warm-up.
#
# C18 prints perm_A[:nblk] and perm_B[:nblk] (the only ids the kernel should
# read) so we can see the structural difference between them directly.
#
# Read:
#   perm_B first-call LARGE, perm_A first-call ~1e-4  => H1: table-content bug.
#   perm_B first-call ~1e-4 but [perm_A,identity,perm_B,swap] chain has
#       swap LARGE while clean swap ~1e-4             => H2: pollution chain.
#   perm_A[:nblk] vs perm_B[:nblk] differ in a visible way (e.g. count of
#       ids >= 16, or # of adjacent pairs)             => the trigger feature.
#
# Usage:
#   python test_fa3_graph_permb_isolation.py
#   KV=2048 python test_fa3_graph_permb_isolation.py

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
    print(f"C18 perm_B isolation   kv={kv}  nblk={nblk}")
    print("=" * 72)

    q = torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(num_blocks_pool, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 1], dtype=torch.int32).npu()
    seq = torch.tensor([kv], dtype=torch.int32).npu()

    # Fixed tables (CPU first so we can print them, then .npu()).
    perm_a_cpu = torch.randperm(num_blocks_pool, dtype=torch.int32)   # 1st randperm
    perm_b_cpu = torch.randperm(num_blocks_pool, dtype=torch.int32)   # 2nd randperm
    identity_cpu = torch.arange(num_blocks_pool, dtype=torch.int32)
    swap_cpu = torch.arange(num_blocks_pool, dtype=torch.int32)
    swap_cpu[0] = num_blocks_pool - 1
    swap_cpu[num_blocks_pool - 1] = 0

    perm_a = perm_a_cpu.npu().unsqueeze(0)
    perm_b = perm_b_cpu.npu().unsqueeze(0)
    identity = identity_cpu.npu().unsqueeze(0)
    swap = swap_cpu.npu().unsqueeze(0)

    print("-" * 72)
    print(f"perm_A[:{nblk}] = {perm_a_cpu[:nblk].tolist()}")
    print(f"perm_B[:{nblk}] = {perm_b_cpu[:nblk].tolist()}")
    a16 = perm_a_cpu[:nblk]
    b16 = perm_b_cpu[:nblk]
    print(f"perm_A[:nblk] ids>=16 count = {(a16 >= 16).sum().item()}")
    print(f"perm_B[:nblk] ids>=16 count = {(b16 >= 16).sum().item()}")
    print("-" * 72)

    # ---- scenario 1: perm_B as a CLEAN first call (fresh meta, no warm-up) ----
    meta1 = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                       num_blocks_pool * BLOCK_SIZE)
    ref_b = manual_ref_blocktable(q, k, v, perm_b_cpu, kv)
    out_b = _run_fa3(q, k, v, seq, cu_q, perm_b, meta1)
    torch.npu.synchronize()
    print(f"[S1] perm_B  first-call (fresh meta, no warm-up) : {_max_abs_diff(out_b, ref_b):.6f}")

    # ---- scenario 2: perm_A as a CLEAN first call (fresh meta) — control ----
    meta2 = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                       num_blocks_pool * BLOCK_SIZE)
    ref_a = manual_ref_blocktable(q, k, v, perm_a_cpu, kv)
    out_a = _run_fa3(q, k, v, seq, cu_q, perm_a, meta2)
    torch.npu.synchronize()
    print(f"[S2] perm_A  first-call (fresh meta, no warm-up) : {_max_abs_diff(out_a, ref_a):.6f}")

    # ---- scenario 3: perm_B after warm-up=identity (fresh meta) ----
    meta3 = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                       num_blocks_pool * BLOCK_SIZE)
    _run_fa3(q, k, v, seq, cu_q, identity, meta3)          # warm-up identity
    out_b3 = _run_fa3(q, k, v, seq, cu_q, perm_b, meta3)
    torch.npu.synchronize()
    print(f"[S3] perm_B  after warm-up=identity            : {_max_abs_diff(out_b3, ref_b):.6f}")

    # ---- scenario 4: full C15 pollution chain (shared meta) ----
    meta4 = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                       num_blocks_pool * BLOCK_SIZE)
    _run_fa3(q, k, v, seq, cu_q, perm_a, meta4)            # warm-up perm_A
    _run_fa3(q, k, v, seq, cu_q, identity, meta4)
    out_b4 = _run_fa3(q, k, v, seq, cu_q, perm_b, meta4)   # perm_B (errored in C15)
    ref_swap = manual_ref_blocktable(q, k, v, swap_cpu, kv)
    out_swap4 = _run_fa3(q, k, v, seq, cu_q, swap, meta4)  # swap right after perm_B
    torch.npu.synchronize()
    print(f"[S4] perm_B  after [perm_A, identity]          : {_max_abs_diff(out_b4, ref_b):.6f}")
    print(f"[S4] swap    after [perm_A,identity,perm_B]    : {_max_abs_diff(out_swap4, ref_swap):.6f}")

    # ---- scenario 5: clean swap (fresh meta, warm-up identity, NO perm_B) ----
    meta5 = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                       num_blocks_pool * BLOCK_SIZE)
    _run_fa3(q, k, v, seq, cu_q, identity, meta5)
    out_swap5 = _run_fa3(q, k, v, seq, cu_q, swap, meta5)
    torch.npu.synchronize()
    print(f"[S5] swap    clean (fresh meta, warm-up identity): {_max_abs_diff(out_swap5, ref_swap):.6f}")

    torch.npu.synchronize()

    print("-" * 72)
    print("Read:")
    print("  S1 large AND S2 small      => H1: perm_B is a table-content kernel bug.")
    print("  S1 small, but S4-swap large while S5-swap small")
    print("                            => H2: pollution chain (perm_B corrupts workspace).")
    print("  S1 small AND S3 large      => perm_B needs warm-up=identity to fail (odd).")
    print("  S4 == S1                   => warm-up/context has NO effect (pure content).")
    print("-" * 72)


if __name__ == "__main__":
    main()
