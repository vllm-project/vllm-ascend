# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C2: discriminate the multi-sequence FA3 graph bug.
#
# Experiment C found: seq0 (block 0, len 64) correct; seq1 (len 128) and seq2
# (len 200) WRONG.  Three hypotheses explain "seq0 correct, rest wrong":
#   H1 page_table bug    : kernel uses block 0 for every sequence.
#   H2 cache_seqlens bug : kernel uses cache_seqlens[0] for every sequence.
#   H3 both              : kernel uses block 0 AND cache_seqlens[0] for all.
#
# This experiment isolates page_table vs cache_seqlens with 3 sequences:
#   seq0: len=64,  block=0   (baseline)
#   seq1: len=64,  block=1   (same length, different block -> tests page_table)
#   seq2: len=128, block=0   (same block,  different length -> tests cache_seqlens)
#
# Prediction matrix (WRONG = mismatch vs manual reference):
#   seq1 wrong, seq2 correct  -> H1 (page_table bug)
#   seq1 correct, seq2 wrong  -> H2 (cache_seqlens bug)
#   seq1 wrong, seq2 wrong    -> H3 (both)
#
# Usage:
#   python test_fa3_graph_multiseq_discrim.py

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
NUM_BLOCKS = 2
PAGE_TABLE = [[0], [1], [0]]  # seq0->blk0, seq1->blk1, seq2->blk0
CAPTURE_SEQLENS = [16, 16, 16]
REPLAY_SEQLENS = [64, 64, 128]  # seq0:64, seq1:64, seq2:128


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def manual_ref(q_seq, page_row, seq_len):
    nblk = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    blocks = page_row[:nblk]
    k_flat = k[blocks].reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:seq_len].float()
    v_flat = v[blocks].reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:seq_len].float()
    k_g = k_flat.repeat_interleave(GROUP, dim=1)
    v_g = v_flat.repeat_interleave(GROUP, dim=1)
    scores = torch.einsum("bhd,thd->bht", q_seq.float().unsqueeze(0), k_g) * SCALE
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bht,thd->bhd", attn, v_g).squeeze(0)


def main():
    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    print("=" * 72)
    print(f"batch={BATCH}  capture={CAPTURE_SEQLENS}  replay={REPLAY_SEQLENS}")
    print(f"page_table={PAGE_TABLE}")
    print("=" * 72)

    global k, v
    q = torch.randn(BATCH, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)

    cu_q = torch.tensor([0, 1, 2, 3], dtype=torch.int32).npu()
    page_table = torch.tensor(PAGE_TABLE, dtype=torch.int32).npu()
    cache_seqlens_buf = torch.tensor(CAPTURE_SEQLENS, dtype=torch.int32).npu()

    meta = _get_scheduler_metadata(
        batch_size=BATCH,
        max_seqlen_q=1,
        max_seqlen_k=max(CAPTURE_SEQLENS),
        num_heads_q=NUM_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        headdim=HEAD_SIZE,
        cache_seqlens=cache_seqlens_buf,
        qkv_dtype=DTYPE,
        cu_seqlens_q=cu_q,
        page_size=BLOCK_SIZE,
        causal=True,
    )

    def run_fa3(cache_seqlens, page_table):
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

    refs = [
        manual_ref(q[i], PAGE_TABLE[i], REPLAY_SEQLENS[i]) for i in range(BATCH)
    ]

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run_fa3(cache_seqlens_buf, page_table)
    torch.npu.synchronize()

    cache_seqlens_buf.copy_(
        torch.tensor(REPLAY_SEQLENS, dtype=torch.int32).npu()
    )
    graph.replay()
    torch.npu.synchronize()
    out = captured.clone()

    errs = [_max_abs_diff(out[i], refs[i]) for i in range(BATCH)]
    print(f"[ok ] seq0 (len=64, blk=0)  vs manual : {errs[0]:.6f}")
    print(f"[ok ] seq1 (len=64, blk=1)  vs manual : {errs[1]:.6f}")
    print(f"[ok ] seq2 (len=128, blk=0) vs manual : {errs[2]:.6f}")

    print("-" * 72)
    if errs[1] > 0.1 and errs[2] <= 0.1:
        print("VERDICT: H1 — page_table bug (kernel uses block 0 for all)")
    elif errs[1] <= 0.1 and errs[2] > 0.1:
        print("VERDICT: H2 — cache_seqlens bug (kernel uses cache_seqlens[0] for all)")
    elif errs[1] > 0.1 and errs[2] > 0.1:
        print("VERDICT: H3 — BOTH page_table and cache_seqlens wrong")
    else:
        print("VERDICT: none of H1/H2/H3 (all correct) — bug is elsewhere")
    print("-" * 72)


if __name__ == "__main__":
    main()
