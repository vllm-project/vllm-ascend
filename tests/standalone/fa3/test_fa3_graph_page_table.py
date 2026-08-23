# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment B: does the FA3 graph re-read a MUTATED page_table at replay?
#
# Experiment A proved the graph correctly re-reads a mutated cache_seqlens,
# but it used a STATIC identity page_table (arange).  Production's block_table
# GROWS during decode and is refreshed by update_graph_params before each
# replay.  If the FA3 graph bakes the page_table at capture (does NOT re-read
# the mutated buffer), decode reads STALE blocks -> wrong output that worsens
# with sequence length.  That is the #1 remaining suspect for the production
# decode-precision bug.
#
# Design (all on the graph path, which experiment A proved is immune to the
# eager "first-call bakes cache_seqlens" bug):
#   - k/v cache: 8 blocks, each filled with DIFFERENT randn, so block 0 vs
#     block 5 give clearly different outputs.
#   - cache_seqlens = 16 (exactly one 128-token block).
#   - page_table_buf is a mutable (1,1) int32 buffer.
#   - Capture the graph reading page_table_buf == [0].
#   - Replay (block 0) -> out_a; mutate page_table_buf to [5]; replay -> out_b.
#
# Read:
#   [ref]  manual block0 vs block5  LARGE  -> the two blocks genuinely differ
#   [block] replay blk0 vs blk5     LARGE  -> graph re-reads page_table (GOOD)
#                                   ~0     -> graph bakes page_table (BUG FOUND)
#   [ok ]   out vs manual ref       ~1e-2  -> output matches the intended block
#
# Usage:
#   python test_fa3_graph_page_table.py

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
SEQ = 16  # cache_seqlens == 16 -> exactly one block

NUM_BLOCKS = 8
BLK_A = 0
BLK_B = 5


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def manual_ref(q, k, v, blk: int):
    """float32 GQA attention over the first SEQ KVs of block ``blk``.

    Decode query attends all SEQ cached KVs; causal == full attention here.
    """
    k_flat = k[blk].reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:SEQ].float()
    v_flat = v[blk].reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:SEQ].float()
    k_g = k_flat.repeat_interleave(GROUP, dim=1)
    v_g = v_flat.repeat_interleave(GROUP, dim=1)
    scores = torch.einsum("bhd,thd->bht", q.float(), k_g) * SCALE
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bht,thd->bhd", attn, v_g)


def main():
    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    print("=" * 72)
    print(f"blocks={NUM_BLOCKS}  block_size={BLOCK_SIZE}  seq={SEQ}  "
          f"blk_a={BLK_A}  blk_b={BLK_B}")
    print("=" * 72)

    q = torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)

    cu_q = torch.tensor([0, 1], dtype=torch.int32).npu()
    cache_seqlens = torch.tensor([SEQ], dtype=torch.int32).npu()

    meta = _get_scheduler_metadata(
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=SEQ,
        num_heads_q=NUM_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        headdim=HEAD_SIZE,
        cache_seqlens=cache_seqlens,
        qkv_dtype=DTYPE,
        cu_seqlens_q=cu_q,
        page_size=BLOCK_SIZE,
        causal=True,
    )

    def run_fa3(page_table):
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

    ref_a = manual_ref(q, k, v, BLK_A)
    ref_b = manual_ref(q, k, v, BLK_B)

    # Mutable page_table buffer whose address the graph will capture.
    page_buf = torch.zeros(1, 1, dtype=torch.int32).npu()
    page_buf[0, 0] = BLK_A

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run_fa3(page_buf)
    torch.npu.synchronize()

    # Replay reading BLK_A.
    graph.replay()
    torch.npu.synchronize()
    out_a = captured.clone()

    # Mutate page_table -> BLK_B, replay the SAME graph.
    page_buf[0, 0] = BLK_B
    graph.replay()
    torch.npu.synchronize()
    out_b = captured.clone()

    print(f"[ref  ] manual blk{BLK_A} vs blk{BLK_B}          : "
          f"{_max_abs_diff(ref_a, ref_b):.6f}")
    print(f"[block] replay blk{BLK_A} vs blk{BLK_B}          : "
          f"{_max_abs_diff(out_a, out_b):.6f}")
    print(f"[ok   ] out(blk{BLK_A}) vs ref(blk{BLK_A})       : "
          f"{_max_abs_diff(out_a, ref_a):.6f}")
    print(f"[ok   ] out(blk{BLK_B}) vs ref(blk{BLK_B})       : "
          f"{_max_abs_diff(out_b, ref_b):.6f}")

    print("-" * 72)
    print("Read:")
    print("  [ref  ] LARGE  => the two blocks genuinely differ (sanity)")
    print("  [block] LARGE  => graph re-reads page_table (GOOD, block refresh OK)")
    print("          ~0    => graph BAKES page_table (production bug FOUND)")
    print("  [ok   ] ~1e-2 => output matches the intended block (bf16 vs fp32)")
    print("-" * 72)


if __name__ == "__main__":
    main()
