# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C21: is the inter-cell DEVICE SYNC the missing variable?
#
# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------
# C15 (WRONG) and C18 (WRONG) both have an implicit device sync between FA3
# calls: manual_ref_blocktable does `block_table[:nblk].cpu()`, which flushes the
# whole NPU stream.  C19/C20 (ALL CORRECT) issue FA3 calls back-to-back with no
# sync.  The three-step chain perm_A->identity->perm_B was wrong in C18 (0.149)
# but the *same* chain is correct in C20 (0.000351) — the only code-level
# difference is the sync structure around the calls.
#
# C21 runs the EXACT C15 call sequence (warm-up perm_A -> sync -> A identity ->
# B perm_B -> C reversed -> D swap -> E identity, with manual_ref's implicit
# sync between cells) and compares it against a NO-SYNC variant of the same
# tables.  Run each MODE as a separate process:
#   MODE=sync    python test_fa3_graph_sync.py   # reproduces C15
#   MODE=nosync  python test_fa3_graph_sync.py   # control
#
# Read:
#   MODE=sync B large, MODE=nosync B small => inter-cell sync is the trigger.
#   both modes small                         => sync is NOT it; the C15/C18 error
#                                               was environmental/one-off.
#
# Usage:
#   MODE=sync python test_fa3_graph_sync.py
#   MODE=nosync python test_fa3_graph_sync.py
#   KV=2048 MODE=sync python test_fa3_graph_sync.py

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
    mode = os.environ.get("MODE", "sync")
    kv = int(os.environ.get("KV", "2048"))
    num_blocks_pool = 64
    nblk = _ceil_div(kv, BLOCK_SIZE)
    assert nblk <= num_blocks_pool, "KV too long for the physical pool"

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    torch.manual_seed(0)

    print("=" * 72)
    print(f"C21 sync-structure   MODE={mode}  kv={kv}  nblk={nblk}")
    print("=" * 72)

    q = torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(num_blocks_pool, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 1], dtype=torch.int32).npu()
    seq = torch.tensor([kv], dtype=torch.int32).npu()

    meta = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                      num_blocks_pool * BLOCK_SIZE)

    # C15's exact tables, in the exact construction order.
    warm_perm = torch.randperm(num_blocks_pool, dtype=torch.int32).npu().unsqueeze(0)  # perm_A
    page_id = torch.arange(num_blocks_pool, dtype=torch.int32).npu().unsqueeze(0)      # identity
    perm = torch.randperm(num_blocks_pool, dtype=torch.int32).npu()                     # perm_B
    page_reorder = perm.unsqueeze(0)
    rev_cpu = torch.arange(num_blocks_pool, dtype=torch.int32)
    rev_cpu[:nblk] = torch.arange(nblk - 1, -1, -1, dtype=torch.int32)
    page_rev = rev_cpu.npu().unsqueeze(0)
    swp_cpu = torch.arange(num_blocks_pool, dtype=torch.int32)
    swp_cpu[0] = num_blocks_pool - 1
    swp_cpu[num_blocks_pool - 1] = 0
    page_swp = swp_cpu.npu().unsqueeze(0)

    # Warm-up (perm_A), then sync — exactly C15.
    _run_fa3(q, k, v, seq, cu_q, warm_perm, meta)
    torch.npu.synchronize()

    cells = [("A identity", page_id), ("B reorder", page_reorder),
             ("C reversed", page_rev), ("D swap", page_swp),
             ("E identity", page_id)]

    if mode == "sync":
        # C15's exact per-cell structure: run FA3, then manual_ref (whose .cpu()
        # implicitly syncs the device before the next cell).
        print("  [mode=sync] FA3 -> manual_ref (implicit sync) between cells:")
        for name, tbl in cells:
            out = _run_fa3(q, k, v, seq, cu_q, tbl, meta)
            ref = manual_ref_blocktable(q, k, v, tbl[0], kv)
            torch.npu.synchronize()
            print(f"    {name:12s} : diff = {_max_abs_diff(out, ref):.6f}")
    else:
        # Same tables, back-to-back with NO inter-cell sync; measure B only.
        print("  [mode=nosync] back-to-back FA3, no inter-cell sync:")
        ref_b = manual_ref_blocktable(q, k, v, page_reorder[0], kv)
        torch.npu.synchronize()   # compute ref up front so its .cpu() sync is NOT between cells
        _run_fa3(q, k, v, seq, cu_q, page_id, meta)
        out_b = _run_fa3(q, k, v, seq, cu_q, page_reorder, meta)
        torch.npu.synchronize()
        print(f"    B reorder     : diff = {_max_abs_diff(out_b, ref_b):.6f}")

    torch.npu.synchronize()

    print("-" * 72)
    print("Read:")
    print("  sync B large / nosync B small => inter-cell sync is the trigger.")
    print("  both small                    => sync is NOT it; C15/C18 error was one-off.")
    print("-" * 72)


if __name__ == "__main__":
    main()
