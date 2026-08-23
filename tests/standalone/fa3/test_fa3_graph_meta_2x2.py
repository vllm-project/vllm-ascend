# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C7: what drives the FA3 tile plan — cache_seqlens or max_seqlen_k?
#
# C6 falsified the "workspace ~ batch x max_seqlen_k -> 2GB -> crash" theory:
# every cell was OK (up to batch=256/maxk=32768) and the measured device-memory
# delta was flat ~55MB.  So the production crash (after setting max_seqlen_k =
# max_model_len) must come from something else.
#
# The production fix changed ONLY max_seqlen_k -> max_model_len, while leaving
# cache_seqlens = the warmup batch's small real lengths (1-16).  That is an
# INCONSISTENT get_scheduler_metadata(max_seqlen_k=32768, cache_seqlens=[16,...])
# input, which C5/C6 never tested (they used cache_seqlens == max_seqlen_k == maxk).
#
# This experiment is a 2x2 over (plan cache_seqlens, max_seqlen_k):
#   A  small,small : cache_seqlens=[16]*b , maxk=16   -> baseline under-cover (WRONG, no crash)
#   B  small,large : cache_seqlens=[16]*b , maxk=2048 -> THE PRODUCTION FIX inconsistency (??)
#   C  large,small : cache_seqlens=[2048]*b, maxk=16  -> reverse inconsistency (??)
#   D  large,large : cache_seqlens=[2048]*b, maxk=2048-> C5/C6 consistent over-cover (OK)
#
# Every case replays with cache_seqlens=[64,128,200] (batch=3) over a page_table
# of width 16 (contiguous blocks), so the block table is never the limiting
# factor.  Read:
#   B crashes / wrong  -> max_seqlen_k alone does NOT over-cover; the fix must
#                         also set cache_seqlens = max_model_len in the plan.
#   C crashes          -> cache_seqlens drives the tile plan but max_seqlen_k
#                         sizes the workspace; a too-small max_seqlen_k OOBs.
#   B OK + D OK        -> tile plan is driven by max_seqlen_k, cache_seqlens only
#                         masks; then the crash is elsewhere (stale block table).
#
# Each case runs in a fresh subprocess so an MTE fault cannot kill the sweep.
#
# Usage:
#   python test_fa3_graph_meta_2x2.py                 # all 4 cases, batch=3
#   python test_fa3_graph_meta_2x2.py --batch 64      # scale up (crash may be size-dependent)

import argparse
import subprocess
import sys
from importlib import util as importlib_util

import torch

HEAD_SIZE = 128
NUM_HEADS = 32
NUM_KV_HEADS = 8
BLOCK_SIZE = 128
DTYPE = torch.bfloat16
SCALE = 1.0 / (HEAD_SIZE ** 0.5)
GROUP = NUM_HEADS // NUM_KV_HEADS

MAXK_LARGE = 2048
MAXK_SMALL = 16
SEQLEN_LARGE = MAXK_LARGE
SEQLEN_SMALL = MAXK_SMALL

def _replay_seqlens(batch: int) -> list[int]:
    # lengths spanning 1-2 blocks, cycling so batch > 3 is well-defined
    # (64, 132, 200, 64, 132, 200, ...)  -> seq2/seq5/... span 2 blocks
    return [64 + (i % 3) * 68 for i in range(batch)]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def manual_ref(q_seq, k, v, page_row, seq_len):
    nblk = _ceil_div(seq_len, BLOCK_SIZE)
    blocks = page_row[:nblk]
    k_flat = k[blocks].reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:seq_len].float()
    v_flat = v[blocks].reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:seq_len].float()
    k_g = k_flat.repeat_interleave(GROUP, dim=1)
    v_g = v_flat.repeat_interleave(GROUP, dim=1)
    scores = torch.einsum("bhd,thd->bht", q_seq.float().unsqueeze(0), k_g) * SCALE
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bht,thd->bhd", attn, v_g).squeeze(0)


def load_fa3():
    for mod_name in ("flash_attn_npu_3", "flash_attn_npu_3"):
        if importlib_util.find_spec(mod_name) is not None:
            mod = __import__(
                mod_name,
                fromlist=["flash_attn_with_kvcache", "get_scheduler_metadata"],
            )
            return mod.flash_attn_with_kvcache, mod.get_scheduler_metadata
    return None, None


def run_case(batch: int, plan_mode: str, maxk_mode: str) -> int:
    fa3_kvcache, get_scheduler_metadata = load_fa3()
    if fa3_kvcache is None:
        print(f"[{plan_mode},{maxk_mode}] SKIP (FA3 not installed)", flush=True)
        return 3

    plan_seqlens = [SEQLEN_LARGE if plan_mode == "large" else SEQLEN_SMALL] * batch
    maxk = MAXK_LARGE if maxk_mode == "large" else MAXK_SMALL

    # page_table width = 16 (ceil(MAXK_LARGE/128)), contiguous per sequence, so
    # the block table is never the limiting factor.
    blocks_per_seq = _ceil_div(MAXK_LARGE, BLOCK_SIZE)
    num_blocks = batch * blocks_per_seq
    page_table = [
        list(range(i * blocks_per_seq, (i + 1) * blocks_per_seq))
        for i in range(batch)
    ]

    q = torch.randn(batch, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(
        num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE
    ).npu()
    v = torch.randn_like(k)

    cu_q = torch.arange(batch + 1, dtype=torch.int32).npu()
    page_table_buf = torch.tensor(page_table, dtype=torch.int32).npu()
    plan_buf = torch.tensor(plan_seqlens, dtype=torch.int32).npu()

    meta = get_scheduler_metadata(
        batch_size=batch,
        max_seqlen_q=1,
        max_seqlen_k=maxk,
        num_heads_q=NUM_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        headdim=HEAD_SIZE,
        cache_seqlens=plan_buf,
        qkv_dtype=DTYPE,
        cu_seqlens_q=cu_q,
        page_size=BLOCK_SIZE,
        causal=True,
    )

    replay_seqlens = _replay_seqlens(batch)
    refs = [
        manual_ref(q[i], k, v, page_table[i], replay_seqlens[i])
        for i in range(batch)
    ]

    cache_seqlens_buf = torch.tensor(plan_seqlens, dtype=torch.int32).npu()

    def run():
        return fa3_kvcache(
            q, k, v,
            cache_seqlens=cache_seqlens_buf,
            page_table=page_table_buf,
            cu_seqlens_q=cu_q,
            max_seqlen_q=1,
            softmax_scale=SCALE,
            causal=True,
            window_size=(-1, -1),
            scheduler_metadata=meta,
        )

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    torch.npu.synchronize()

    cache_seqlens_buf.copy_(
        torch.tensor(replay_seqlens, dtype=torch.int32).npu()
    )
    graph.replay()
    torch.npu.synchronize()
    out = captured.clone()

    errs = [_max_abs_diff(out[i], refs[i]) for i in range(batch)]
    ok = all(e <= 0.1 for e in errs)
    tag = "OK" if ok else "WRONG"
    print(
        f"[{plan_mode},{maxk_mode}] batch={batch} plan={plan_seqlens[0]} maxk={maxk} "
        f"seq0/seq1/seq2 = {errs[0]:.6f} {errs[1]:.6f} {errs[2]:.6f} -> {tag}",
        flush=True,
    )
    return 0 if ok else 1


CASES = [
    ("small", "small"),
    ("small", "large"),
    ("large", "small"),
    ("large", "large"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--case", type=int, default=None)
    args = ap.parse_args()

    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    if args.case is not None:
        plan_mode, maxk_mode = CASES[args.case]
        sys.exit(run_case(args.batch, plan_mode, maxk_mode))

    print("=" * 78)
    print(f"C7 2x2 (batch={args.batch}, replay={_replay_seqlens(args.batch)})")
    print("=" * 78)
    for i, (plan_mode, maxk_mode) in enumerate(CASES):
        cmd = [sys.executable, __file__, "--batch", str(args.batch), "--case", str(i)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        line = next((l for l in r.stdout.splitlines() if "->" in l), r.stdout.strip())
        result = "OK" if r.returncode == 0 else f"CRASH({r.returncode})"
        print(f"  {plan_mode:>6},{maxk_mode:<6} -> {result}   {line}")
        if r.returncode not in (0, 1):
            tail = "\n".join(r.stderr.splitlines()[-6:])
            if tail.strip():
                print(f"         stderr tail: {tail}")
    print("=" * 78)
    print("Read:")
    print("  B(small,large) CRASH/WRONG -> max_seqlen_k alone does not over-cover;")
    print("       fix must also set cache_seqlens = max_model_len in the plan.")
    print("  C(large,small) CRASH       -> cache_seqlens drives tiles, maxk sizes")
    print("       workspace; too-small maxk OOBs.")
    print("  B OK and D OK              -> tiles driven by maxk; crash elsewhere.")
    print("=" * 78)


if __name__ == "__main__":
    main()
