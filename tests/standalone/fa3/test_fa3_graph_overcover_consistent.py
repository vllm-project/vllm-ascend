# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C5: production-faithful over-cover.  In C4, over-cover at 200
# tokens worked but over-cover at 2048 failed ONLY because the page_table width
# (2) was smaller than the plan's tile count (16) -> out-of-bounds block reads.
# Production keeps these consistent: block_table width == max_blocks_per_seq ==
# ceil(max_model_len / block_size), and the plan is built for the same max.
#
# This experiment builds the plan for the MAX config and gives the page_table
# the MATCHING width, exactly like the planned production fix:
#
#   plan cache_seqlens = [MAX, MAX, MAX]   (max_seqlen_k = MAX)
#   page_table width   = ceil(MAX / 128)
#   replay cache_seqlens = [64, 128, 200]  (shorter than the plan)
#
# If over-cover with consistent width is OK, the production fix
# (build metadata plan for max_model_len) is validated.
#
# Usage:
#   python test_fa3_graph_overcover_consistent.py

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
REPLAY_SEQLENS = [64, 128, 200]


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


def run_case(name, max_len):
    # page_table width == plan tile count == ceil(max_len / 128)
    blocks_per_seq = _ceil_div(max_len, BLOCK_SIZE)
    num_blocks = BATCH * blocks_per_seq
    # seq i -> contiguous range of blocks [i*bps, (i+1)*bps)
    page_table = [
        list(range(i * blocks_per_seq, (i + 1) * blocks_per_seq))
        for i in range(BATCH)
    ]

    q = torch.randn(BATCH, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)

    cu_q = torch.tensor([0, 1, 2, 3], dtype=torch.int32).npu()
    page_table_buf = torch.tensor(page_table, dtype=torch.int32).npu()

    plan_seqlens = [max_len] * BATCH
    plan_buf = torch.tensor(plan_seqlens, dtype=torch.int32).npu()
    meta = _get_scheduler_metadata(
        batch_size=BATCH,
        max_seqlen_q=1,
        max_seqlen_k=max_len,
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
        manual_ref(q[i], k, v, page_table[i], REPLAY_SEQLENS[i])
        for i in range(BATCH)
    ]

    # capture with the max config (like production), replay with shorter lengths
    cache_seqlens_buf = torch.tensor(plan_seqlens, dtype=torch.int32).npu()

    def run():
        return _fa3_kvcache(
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
        torch.tensor(REPLAY_SEQLENS, dtype=torch.int32).npu()
    )
    graph.replay()
    torch.npu.synchronize()
    out = captured.clone()

    errs = [_max_abs_diff(out[i], refs[i]) for i in range(BATCH)]
    ok = all(e <= 0.1 for e in errs)
    print(f"[{name}] max_len={max_len}  blocks/seq={blocks_per_seq}  "
          f"seq0/seq1/seq2 = {errs[0]:.6f}  {errs[1]:.6f}  {errs[2]:.6f}"
          f"  -> {'OK' if ok else 'BROKEN'}")
    return ok


def main():
    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    print("=" * 72)
    print(f"batch={BATCH}  replay={REPLAY_SEQLENS}  (plan built for MAX)")
    print("=" * 72)

    r16 = run_case("over-16", 2048)   # 16 blocks/seq
    r32 = run_case("over-32", 4096)   # 32 blocks/seq

    print("-" * 72)
    if r16 and r32:
        print("VERDICT: consistent over-cover OK -> production fix validated:")
        print("         build metadata plan for max_model_len, replay shorter")
    elif r16 and not r32:
        print("VERDICT: over-16 OK, over-32 BROKEN -> size-dependent; the max")
        print("         config plan degrades for larger max_seqlen_k")
    else:
        print("VERDICT: over-cover BROKEN even with consistent width -> the")
        print("         fix needs a different approach (not just over-cover)")
    print("-" * 72)


if __name__ == "__main__":
    main()
