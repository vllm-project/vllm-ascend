# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C3: is the multi-seq FA3 graph bug caused by a STALE
# scheduler_metadata (built from the capture-time cache_seqlens/max_seqlen_k),
# or is it inherent to FA3's multi-seq + multi-block decode?
#
# Experiment C (test_fa3_graph_multiseq.py) found seq1 (len 128) and seq2
# (len 200, 2 blocks) WRONG, seq0 (len 64) correct, with:
#     page_table = [[0,1],[2,3],[4,5]]   (width 2)
#     metadata   built from cache_seqlens=[16,16,16], max_seqlen_k=16
#     replay     cache_seqlens=[64,128,200]   (seq2 now spans 2 KV blocks)
# Experiment C2 ruled out "kernel uses block 0" and "kernel uses cache_seqlens[0]"
# (its width-1 / all-single-block batch was fully correct).  So the trigger is
# either (a) the metadata plan is too small to cover the replay lengths, or
# (b) FA3's multi-seq path breaks whenever any sequence spans 2 blocks.
#
# This experiment distinguishes (a) from (b) by rebuilding the metadata to
# COVER the replay lengths while keeping the identical kernel call:
#
#   case  stale-meta : meta(cache_seqlens=[16,16,16],  maxk=16)   <- == C (control)
#   case  fresh-meta : meta(cache_seqlens=[64,128,200],maxk=200)  <- plan covers replay
#   case  fresh-all  : meta(cache_seqlens=[64,128,200],maxk=200) + capture=[64,128,200]
#
# Read:
#   stale-meta BROKEN and fresh-meta OK     -> metadata staleness is the bug (fixable)
#   stale-meta BROKEN and fresh-meta BROKEN -> FA3 multi-seq multi-block kernel bug
#   fresh-all  BROKEN                       -> FA3 multi-seq multi-block is
#                                              fundamentally wrong (no staleness at all)
#
# Usage:
#   python test_fa3_graph_multiseq_meta.py

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
NUM_BLOCKS = 6
PAGE_TABLE = [[0, 1], [2, 3], [4, 5]]  # width 2, same as experiment C
REPLAY_SEQLENS = [64, 128, 200]  # seq2 spans 2 blocks
CAPTURE_SEQLENS = [16, 16, 16]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def manual_ref(q_seq, k, v, page_row, seq_len):
    """float32 GQA attention over the first seq_len KVs of the sequence's blocks."""
    nblk = _ceil_div(seq_len, BLOCK_SIZE)
    blocks = page_row[:nblk]
    k_flat = k[blocks].reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:seq_len].float()
    v_flat = v[blocks].reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:seq_len].float()
    k_g = k_flat.repeat_interleave(GROUP, dim=1)
    v_g = v_flat.repeat_interleave(GROUP, dim=1)
    scores = torch.einsum("bhd,thd->bht", q_seq.float().unsqueeze(0), k_g) * SCALE
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bht,thd->bhd", attn, v_g).squeeze(0)  # (H, D)


def run_case(name, meta_cache_seqlens, meta_maxk, capture_seqlens, replay_seqlens):
    q = torch.randn(BATCH, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)

    cu_q = torch.tensor([0, 1, 2, 3], dtype=torch.int32).npu()
    page_table = torch.tensor(PAGE_TABLE, dtype=torch.int32).npu()

    # metadata is built from *meta_cache_seqlens* (the plan), independent of
    # the mutable capture buffer below.
    meta_buf = torch.tensor(meta_cache_seqlens, dtype=torch.int32).npu()
    meta = _get_scheduler_metadata(
        batch_size=BATCH,
        max_seqlen_q=1,
        max_seqlen_k=meta_maxk,
        num_heads_q=NUM_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        headdim=HEAD_SIZE,
        cache_seqlens=meta_buf,
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
    cache_seqlens_buf = torch.tensor(capture_seqlens, dtype=torch.int32).npu()

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

    stale = run_case(
        "stale-meta",
        meta_cache_seqlens=CAPTURE_SEQLENS,      # [16,16,16]
        meta_maxk=max(CAPTURE_SEQLENS),           # 16
        capture_seqlens=CAPTURE_SEQLENS,
        replay_seqlens=REPLAY_SEQLENS,
    )
    fresh = run_case(
        "fresh-meta",
        meta_cache_seqlens=REPLAY_SEQLENS,        # [64,128,200]
        meta_maxk=max(REPLAY_SEQLENS),            # 200
        capture_seqlens=CAPTURE_SEQLENS,
        replay_seqlens=REPLAY_SEQLENS,
    )
    fresh_all = run_case(
        "fresh-all ",
        meta_cache_seqlens=REPLAY_SEQLENS,        # [64,128,200]
        meta_maxk=max(REPLAY_SEQLENS),            # 200
        capture_seqlens=REPLAY_SEQLENS,
        replay_seqlens=REPLAY_SEQLENS,
    )

    print("-" * 72)
    if not stale:
        print("VERDICT: stale-meta already OK -> cannot reproduce C; check setup")
    elif fresh:
        print("VERDICT: (a) STALE scheduler_metadata is the bug -> fixable:")
        print("         build metadata so max_seqlen_k covers the max replay length")
    elif not fresh_all:
        print("VERDICT: (b) FA3 multi-seq + multi-block is fundamentally broken,")
        print("         even with fresh metadata -> need kernel workaround")
    else:
        print("VERDICT: (c) inconsistent -> investigate further")
    print("-" * 72)


if __name__ == "__main__":
    main()
