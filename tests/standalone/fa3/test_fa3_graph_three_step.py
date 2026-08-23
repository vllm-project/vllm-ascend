# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C20: the trigger is the THREE-step chain  perm_A -> identity -> perm_B.
#
# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------
# C19 falsified "first-call scattered poisons later scattered": perm_A->perm_B
# (two steps) is correct.  Re-aligning ALL data, the ONLY wrong case is the
# three-step chain  perm_A -> identity -> perm_B:
#   identity -> perm_B            OK   (C18 S3)
#   perm_A   -> perm_B            OK   (C19 Q1/Q4)
#   perm_C   -> perm_B            OK   (C19 Q3)
#   perm_A -> identity -> perm_B  BAD  (C18 S4 = 0.149, C15 = 0.16)
# Workspace S/P uses PRE_LAUNCH+1 = 4 slots with curStackTileMod = stackSeqCount % 4;
# three calls is exactly one slot-wrap period, so call #3 can read residue call #1
# left in the un-overwritten (rowNum=4 << 128) tail of a slot.
#
# C20 pins down:
#   (Q1) determinism: perm_A -> identity -> perm_B wrong every time?
#   (Q2) middle table: identity / perm_A / swap in the middle — which trigger?
#   (Q3) first table:  perm_A / perm_B / identity first — which trigger?
#   (Q4) meta shared vs fresh: does the meta OBJECT matter (vs pure call sequence)?
#   (Q5) spatial: which kv-heads go wrong in the errored perm_B output?
#
# Read:
#   Q1 flaky               => race, not deterministic state.
#   Q2 only identity middles => identity in the middle is special (contiguous).
#   Q2 any middle triggers  => just "three calls", content-independent.
#   Q3 any scattered first  => generic scattered-first.
#   Q4 shared BAD / fresh OK => the meta object is mutated after all.
#   Q5 whole 4-head groups  => a kv-head's blocks mis-read.
#
# Usage:
#   python test_fa3_graph_three_step.py
#   KV=2048 python test_fa3_graph_three_step.py

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
    d = (out.float() - ref.float()).abs().amax(dim=-1)
    return d[0].cpu().tolist()


def main():
    kv = int(os.environ.get("KV", "2048"))
    num_blocks_pool = 64
    nblk = _ceil_div(kv, BLOCK_SIZE)
    assert nblk <= num_blocks_pool, "KV too long for the physical pool"

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    torch.manual_seed(0)

    print("=" * 72)
    print(f"C20 three-step chain   kv={kv}  nblk={nblk}")
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
    swap_cpu = torch.arange(num_blocks_pool, dtype=torch.int32)
    swap_cpu[0] = num_blocks_pool - 1
    swap_cpu[num_blocks_pool - 1] = 0

    perm_a = perm_a_cpu.npu().unsqueeze(0)
    perm_b = perm_b_cpu.npu().unsqueeze(0)
    perm_c = perm_c_cpu.npu().unsqueeze(0)
    identity = identity_cpu.npu().unsqueeze(0)
    swap = swap_cpu.npu().unsqueeze(0)

    ref_b = manual_ref_blocktable(q, k, v, perm_b_cpu, kv)
    ref_a = manual_ref_blocktable(q, k, v, perm_a_cpu, kv)

    def run_seq(tables, shared_meta):
        """Run a list of tables (each already .npu().unsqueeze(0)) with one meta,
        return the last output tensor."""
        out = None
        for t in tables:
            out = _run_fa3(q, k, v, seq, cu_q, t, shared_meta)
        torch.npu.synchronize()
        return out

    # ---- Q1: determinism of the 3-step chain (fresh meta each repeat) ----
    print("-" * 72)
    print("[Q1] perm_A -> identity -> perm_B, 3 repeats (shared meta each):")
    for r in range(3):
        meta = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                          num_blocks_pool * BLOCK_SIZE)
        out = run_seq([perm_a, identity, perm_b], meta)
        print(f"  repeat {r}: diff = {_max_abs_diff(out, ref_b):.6f}")

    # ---- Q2: middle table variation: perm_A -> X -> perm_B ----
    print("-" * 72)
    print("[Q2] perm_A -> X -> perm_B  (X varies):")
    for name, x in (("identity", identity), ("perm_A", perm_a),
                    ("swap", swap), ("perm_C", perm_c)):
        meta = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                          num_blocks_pool * BLOCK_SIZE)
        out = run_seq([perm_a, x, perm_b], meta)
        print(f"  X={name:<8} : diff = {_max_abs_diff(out, ref_b):.6f}")

    # ---- Q3: first-table variation: X -> identity -> perm_B ----
    print("-" * 72)
    print("[Q3] X -> identity -> perm_B  (X varies):")
    for name, x in (("perm_A", perm_a), ("perm_B", perm_b),
                    ("identity", identity), ("swap", swap)):
        meta = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                          num_blocks_pool * BLOCK_SIZE)
        out = run_seq([x, identity, perm_b], meta)
        print(f"  X={name:<8} : diff = {_max_abs_diff(out, ref_b):.6f}")

    # ---- Q4: fresh meta per call vs shared meta ----
    print("-" * 72)
    print("[Q4] perm_A -> identity -> perm_B, fresh meta per call:")
    meta_a = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                        num_blocks_pool * BLOCK_SIZE)
    meta_i = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                        num_blocks_pool * BLOCK_SIZE)
    meta_b = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                        num_blocks_pool * BLOCK_SIZE)
    _run_fa3(q, k, v, seq, cu_q, perm_a, meta_a)
    _run_fa3(q, k, v, seq, cu_q, identity, meta_i)
    out = _run_fa3(q, k, v, seq, cu_q, perm_b, meta_b)
    torch.npu.synchronize()
    print(f"  diff = {_max_abs_diff(out, ref_b):.6f}")

    # ---- Q5: per-head diff of the errored perm_B (shared meta 3-step) ----
    print("-" * 72)
    print("[Q5] per-head diff of perm_B under perm_A -> identity -> perm_B:")
    meta = _make_meta(1, torch.tensor([16], dtype=torch.int32).npu(),
                      num_blocks_pool * BLOCK_SIZE)
    out_bad = run_seq([perm_a, identity, perm_b], meta)
    hd = _per_head_diff(out_bad, ref_b)
    for h, d in enumerate(hd):
        flag = "  <-- WRONG" if d > 0.01 else ""
        print(f"    {h:2d}  ->  kv{h // GROUP}      : {d:.6f}{flag}")

    torch.npu.synchronize()

    print("-" * 72)
    print("Read:")
    print("  Q1 flaky            => race, not deterministic state.")
    print("  Q2 only identity middle => contiguous middle is special.")
    print("  Q2 any middle       => just 'three calls', content-independent.")
    print("  Q3 any scattered first => generic scattered first-call.")
    print("  Q4 fresh OK / shared BAD => meta object is mutated after all.")
    print("  Q5 whole 4-head groups  => a kv-head's blocks mis-read.")
    print("-" * 72)


if __name__ == "__main__":
    main()
