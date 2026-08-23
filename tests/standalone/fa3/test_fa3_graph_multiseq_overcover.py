# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C4: does "over-cover" work?  I.e. does FA3 multi-seq decode stay
# correct when the scheduler_metadata plan is built for MORE KV tiles per
# sequence than the replay actually needs?
#
# C3 established the trigger precisely: the plan bakes a per-sequence KV tile
# count (from the build-time cache_seqlens).  Multi-seq decode breaks when any
# sequence's replay length needs MORE tiles than the plan allows (cache_seqlens
# > block_size, i.e. spans 2 blocks).  Single-seq under-cover is fine (kernel
# re-derives tiles), but multi-seq under-cover is not.
#
# The production fix is to build the plan for the MAX config (every request at
# its max KV length), then replay with shorter actual lengths.  That is
# "over-cover": plan has 2 (or more) tiles/seq, replay needs only 1 or 2.
# This experiment checks whether over-cover is correct:
#
#   under-cover  : plan=[16,16,16]     replay=[64,128,200]  (== C3 stale, BROKEN)
#   over-cover   : plan=[200,200,200]  replay=[64,128,200]  (2 tiles/seq planned)
#   over-cover-2k: plan=[2048,2048,2048] replay=[64,128,200] (16 tiles/seq planned)
#
# Read:
#   under-cover BROKEN, over-cover OK    -> production fix is valid:
#                                            build plan for max config, replay shorter
#   under-cover BROKEN, over-cover BROKEN -> kernel uses baked tile count;
#                                            need to rebuild metadata per replay
#                                            (not possible in graph) -> different approach
#
# Usage:
#   python test_fa3_graph_multiseq_overcover.py

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

BATCH = 3
PAGE_TABLE = [[0, 1], [2, 3], [4, 5]]  # width 2, same as experiments C/C3
REPLAY_SEQLENS = [64, 128, 200]  # seq0:1 block, seq1:1 block, seq2:2 blocks


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


def run_case(name, plan_seqlens, replay_seqlens):
    q = torch.randn(BATCH, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(6, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)

    cu_q = torch.tensor([0, 1, 2, 3], dtype=torch.int32).npu()
    page_table = torch.tensor(PAGE_TABLE, dtype=torch.int32).npu()

    # metadata plan is built from *plan_seqlens* (the baked tile count).
    plan_buf = torch.tensor(plan_seqlens, dtype=torch.int32).npu()
    meta = _get_scheduler_metadata(
        batch_size=BATCH,
        max_seqlen_q=1,
        max_seqlen_k=max(plan_seqlens),
        num_heads_q=NUM_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        headdim=HEAD_SIZE,
        cache_seqlens=plan_buf,
        qkv_dtype=DTYPE,
        cu_seqlens_q=cu_q,
        page_size=BLOCK_SIZE,
        causal=True,
    )

    refs = [
        manual_ref(q[i], k, v, PAGE_TABLE[i], replay_seqlens[i])
        for i in range(BATCH)
    ]

    # mutable cache_seqlens buffer whose address the graph captures.
    cache_seqlens_buf = torch.tensor(plan_seqlens, dtype=torch.int32).npu()

    def run():
        return _fa3_kvcache(
            q, k, v,
            cache_seqlens=cache_seqlens_buf,
            page_table=page_table,
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

    errs = [_max_abs_diff(out[i], refs[i]) for i in range(BATCH)]
    ok = all(e <= 0.1 for e in errs)
    print(f"[{name}] seq0/seq1/seq2 = {errs[0]:.6f}  {errs[1]:.6f}  {errs[2]:.6f}"
          f"  -> {'OK' if ok else 'BROKEN'}")
    return ok


def main():
    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    print("=" * 72)
    print(f"batch={BATCH}  page_table={PAGE_TABLE}  replay={REPLAY_SEQLENS}")
    print("=" * 72)

    under = run_case("under-cover ", [16, 16, 16], REPLAY_SEQLENS)
    over = run_case("over-cover  ", [200, 200, 200], REPLAY_SEQLENS)
    over2k = run_case("over-cover-2k", [2048, 2048, 2048], REPLAY_SEQLENS)

    print("-" * 72)
    if under:
        print("VERDICT: under-cover already OK -> cannot reproduce C3; check setup")
    elif over and over2k:
        print("VERDICT: over-cover OK -> production fix is valid:")
        print("         build the metadata plan for the MAX config, replay shorter")
    elif over and not over2k:
        print("VERDICT: over-cover OK but over-cover-2k BROKEN -> size-dependent;")
        print("         plan must cover replay but not overshoot too far")
    else:
        print("VERDICT: over-cover BROKEN -> kernel uses baked tile count;")
        print("         cannot just over-cover; need a different approach")
    print("-" * 72)


if __name__ == "__main__":
    main()
