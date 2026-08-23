# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C: multi-sequence decode (batch > 1) in the FA3 graph path.
#
# Experiments A/B proved the FA3 graph is correct for a SINGLE sequence:
#   A: cache_seqlens refresh (16 -> 4096) works, scheduler_metadata irrelevant.
#   B: page_table (block_table) refresh works.
# Both were batch=1.  Production decode is batch=N with DIFFERENT per-sequence
# cache_seqlens that GROW each step.  If FA3 mishandles multiple sequences of
# different lengths, decode is wrong even though single-seq is correct.
#
# Design (graph path only — immune to the eager "first-call bakes" bug):
#   - batch=3, distinct cache_seqlens that grow between capture and replay.
#   - page_table: seq i -> blocks [2i, 2i+1] (distinct physical blocks).
#   - capture cache_seqlens = [16, 16, 16]; replay = [64, 128, 200]
#     (seq 0 -> 1 block, seq 1 -> 1 block, seq 2 -> 2 blocks).
#   - manual_ref computes the CORRECT output per sequence (float32 GQA).
#
# Read:
#   [seq ] replay seq_i vs seq_j        LARGE => per-sequence outputs differ
#   [ok  ] replay seq_i vs manual seq_i ~1e-2 => each sequence is CORRECT
#   [grow] seq2 replay grown vs manual  ~1e-2 => growing to 2 blocks is correct
#
# Usage:
#   python test_fa3_graph_multiseq.py

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
MAX_BLOCKS_PER_SEQ = 2
# seq i -> blocks [2i, 2i+1] (distinct physical blocks per sequence)
PAGE_TABLE = [[0, 1], [2, 3], [4, 5]]
CAPTURE_SEQLENS = [16, 16, 16]
REPLAY_SEQLENS = [64, 128, 200]  # seq 0: 1 block, seq 1: 1 block, seq 2: 2 blocks


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def manual_ref(q_seq, page_row, seq_len):
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


def main():
    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    print("=" * 72)
    print(f"batch={BATCH}  capture_seqlens={CAPTURE_SEQLENS}  "
          f"replay_seqlens={REPLAY_SEQLENS}")
    print("=" * 72)

    global k, v
    q = torch.randn(BATCH, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)

    cu_q = torch.tensor([0, 1, 2, 3], dtype=torch.int32).npu()
    page_table = torch.tensor(PAGE_TABLE, dtype=torch.int32).npu()

    # mutable cache_seqlens buffer whose address the graph captures
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

    # references for the replay lengths
    refs = [
        manual_ref(q[i], PAGE_TABLE[i], REPLAY_SEQLENS[i]) for i in range(BATCH)
    ]

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run_fa3(cache_seqlens_buf, page_table)
    torch.npu.synchronize()

    # replay with capture seqlens (should match capture-length references)
    graph.replay()
    torch.npu.synchronize()
    out_cap = captured.clone()

    # mutate cache_seqlens to the replay (grown) lengths, replay the SAME graph
    cache_seqlens_buf.copy_(
        torch.tensor(REPLAY_SEQLENS, dtype=torch.int32).npu()
    )
    graph.replay()
    torch.npu.synchronize()
    out_rep = captured.clone()

    for i in range(BATCH):
        print(f"[ok ] replay seq{i} (len={REPLAY_SEQLENS[i]}) vs manual : "
              f"{_max_abs_diff(out_rep[i], refs[i]):.6f}")
    print(f"[seq ] replay seq0 vs seq1 (should differ)          : "
          f"{_max_abs_diff(out_rep[0], out_rep[1]):.6f}")
    print(f"[seq ] replay seq1 vs seq2 (should differ)          : "
          f"{_max_abs_diff(out_rep[1], out_rep[2]):.6f}")
    print(f"[grow] seq2 replay grown vs capture (should differ)  : "
          f"{_max_abs_diff(out_rep[2], out_cap[2]):.6f}")

    print("-" * 72)
    print("Read:")
    print("  [ok  ] ~1e-2 per sequence => multi-seq FA3 graph is CORRECT")
    print("         large on ANY seq  => multi-seq FA3 graph is WRONG (bug found)")
    print("  [seq ] LARGE => outputs actually differ per sequence (sanity)")
    print("  [grow] LARGE => seq2 output changed after cache_seqlens grew")
    print("-" * 72)


if __name__ == "__main__":
    main()
