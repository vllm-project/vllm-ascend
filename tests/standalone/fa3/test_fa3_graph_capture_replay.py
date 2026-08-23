# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C28: the one thing C1..C27 never exercised — the actual
# NPUGraph/ACLGraph CAPTURE + REPLAY of `flash_attn_with_kvcache` with static
# buffers refreshed between replays.
#
# ---------------------------------------------------------------------------
# Why this exists
# ---------------------------------------------------------------------------
# C25/C26/C27 all call `_fa3_kvcache` directly with FRESH tensors and only ever
# varied the (pre-computed, possibly stale) scheduler_metadata.  Every one was
# correct.  But the production decode bug is graph-mode ONLY: eager FA3 decode is
# correct, graph FA3 decode is wrong.  The production graph:
#   * pre-computes scheduler_metadata ONCE,
#   * allocates fixed NPU buffers (cache_seqlens / cu_seqlens_q / block_table),
#   * captures `fa3_kvcache` INSIDE torch.npu.graph(),
#   * replays, refreshing those buffers between replays via copy_ on a side
#     `update_stream` (with torch.npu.current_stream().wait_stream(update_stream)).
#
# This experiment reproduces that capture+replay and checks each replay against
# the CPU float64 reference.  It is the FIRST experiment that actually runs the
# graph path, so it can distinguish:
#   (a) capture/replay is fine, refresh works        -> bug is vllm-ascend glue
#       ordering (update-after-replay -> first decode reads capture-time zeros),
#   (b) capture bakes stale values / replay reads wrong buffers -> bug is in the
#       FA3 graph capture itself,
#   (c) the side-stream refresh races the replay     -> wait_stream insufficient.
#
# Read:
#   T1 (no refresh)  wrong   -> confirms first-token stale-buffer is a REAL bug.
#   T2 (same-stream refresh) wrong -> capture/replay itself broken (b).
#   T3 (side-stream + wait) wrong  -> stream sync still insufficient (c).
#   all ~1e-3 -> the graph capture/replay path is sound; look elsewhere.
#
# Usage:
#   python test_fa3_graph_capture_replay.py

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
WIDTH = 128
NUM_BLOCKS_POOL = 128


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float().cpu() - b.float().cpu()).abs().max().item())


def _mk_block_table(batch, seqlens, seed):
    g = torch.Generator().manual_seed(seed)
    bt = torch.full((batch, WIDTH), -1, dtype=torch.int32)
    for b, s in enumerate(seqlens):
        nblk = _ceil_div(s, BLOCK_SIZE)
        ids = torch.randperm(NUM_BLOCKS_POOL, generator=g, dtype=torch.int32)[:nblk]
        bt[b, :nblk] = ids
    return bt


def cpu_ref_f64(q_cpu, k_cpu, v_cpu, block_table_cpu, seqlens):
    outs = []
    for b, seq_len in enumerate(seqlens):
        nblk = _ceil_div(seq_len, BLOCK_SIZE)
        ids = block_table_cpu[b, :nblk].tolist()
        k_flat = torch.cat([k_cpu[i] for i in ids], dim=0)[:seq_len]
        v_flat = torch.cat([v_cpu[i] for i in ids], dim=0)[:seq_len]
        k_g = k_flat.repeat_interleave(GROUP, dim=1)
        v_g = v_flat.repeat_interleave(GROUP, dim=1)
        scores = torch.einsum("hd,thd->ht", q_cpu[b], k_g) * SCALE
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("ht,thd->hd", attn, v_g)
        outs.append(out)
    return torch.stack(outs, dim=0)


def _report(tag, out, q, k, v, bt, seqlens, batch):
    ref = cpu_ref_f64(q.cpu().double(), k.cpu().double(), v.cpu().double(), bt, seqlens)
    for b in range(batch):
        d = _max_abs_diff(out[b], ref[b])
        flag = "  <-- WRONG" if d > 0.05 else ""
        print(f"    [{tag}] row {b} seq={seqlens[b]:5d} : {d:.6f}{flag}")


def main():
    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")

    torch.manual_seed(0)
    batch = 4
    print("=" * 72)
    print(f"C28 capture+replay   batch={batch} width={WIDTH} maxk={WIDTH*BLOCK_SIZE}")
    print("=" * 72)

    # ---- static buffers (addresses captured) ----
    q_buf = torch.zeros(batch, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    k = torch.randn(NUM_BLOCKS_POOL, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    cache_seqlens_buf = torch.zeros(batch, dtype=torch.int32).npu()
    cu_q_buf = torch.arange(batch + 1, dtype=torch.int32).npu()
    block_table_buf = torch.zeros(batch, WIDTH, dtype=torch.int32).npu()

    # ---- pre-computed scheduler_metadata (like production, built from warmup) ----
    # bake with SHORT warmup lengths so the metadata differs from the replay.
    warmup_seqlens = torch.tensor([16, 16, 16, 16], dtype=torch.int32).npu()
    cache_seqlens_buf.copy_(warmup_seqlens)
    meta = _get_scheduler_metadata(
        batch_size=batch,
        max_seqlen_q=1,
        max_seqlen_k=WIDTH * BLOCK_SIZE,
        num_heads_q=NUM_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        headdim=HEAD_SIZE,
        cache_seqlens=cache_seqlens_buf,
        qkv_dtype=DTYPE,
        cu_seqlens_q=cu_q_buf,
        page_size=BLOCK_SIZE,
        causal=True,
    )

    def run_fn():
        return _fa3_kvcache(
            q_buf, k, v,
            cache_seqlens=cache_seqlens_buf,
            page_table=block_table_buf,
            cu_seqlens_q=cu_q_buf,
            max_seqlen_q=1,
            softmax_scale=SCALE,
            causal=True,
            window_size=(-1, -1),
            scheduler_metadata=meta,
        )

    # ---- capture ----
    print("[capture] torch.npu.graph begin")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out_capture = run_fn()
    torch.npu.synchronize()
    print("[capture] done")

    # ---- the replay batch (variable lengths, differs from warmup) ----
    seqlens = [512, 1024, 2048, 4096]
    q = torch.randn(batch, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    bt = _mk_block_table(batch, seqlens, 7)
    seq_call = torch.tensor(seqlens, dtype=torch.int32).npu()

    # ---- T1: replay with NO refresh (capture-time: cache_seqlens=[16]*4,
    # block_table=zeros) -> this is what the FIRST decode sees in production
    # (update-after-replay ordering).  Expect WRONG if the stale-buffer bug is
    # real.
    print("-" * 72)
    print("[T1] replay, no refresh (capture-time buffers)")
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    _report("T1", out_capture, q, k, v, bt, seqlens, batch)

    # ---- T2: refresh on the SAME stream, then replay -> should be CORRECT if
    # capture/replay + refresh is sound.
    print("-" * 72)
    print("[T2] refresh on current stream, then replay")
    q_buf.copy_(q)
    cache_seqlens_buf.copy_(seq_call)
    block_table_buf.copy_(bt.npu())
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    _report("T2", out_capture, q, k, v, bt, seqlens, batch)

    # ---- T3: refresh on a SIDE stream + wait_stream (production pattern) ----
    print("-" * 72)
    print("[T3] refresh on side stream + wait_stream, then replay")
    update_stream = torch.npu.Stream()
    q_buf.copy_(torch.zeros_like(q))  # poison so a failed refresh is visible
    cache_seqlens_buf.fill_(0)
    block_table_buf.fill_(-1)
    with torch.npu.stream(update_stream):
        q_buf.copy_(q, non_blocking=True)
        cache_seqlens_buf.copy_(seq_call, non_blocking=True)
        block_table_buf.copy_(bt.npu(), non_blocking=True)
    torch.npu.current_stream().wait_stream(update_stream)
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    _report("T3", out_capture, q, k, v, bt, seqlens, batch)

    # ---- T4: a second, DIFFERENT batch to confirm refresh keeps working ----
    seqlens2 = [256, 768, 1536, 3072]
    q2 = torch.randn(batch, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    bt2 = _mk_block_table(batch, seqlens2, 8)
    seq_call2 = torch.tensor(seqlens2, dtype=torch.int32).npu()
    print("-" * 72)
    print("[T4] second batch, same-stream refresh, replay")
    q_buf.copy_(q2)
    cache_seqlens_buf.copy_(seq_call2)
    block_table_buf.copy_(bt2.npu())
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    _report("T4", out_capture, q2, k, v, bt2, seqlens2, batch)

    print("-" * 72)
    print("Read:")
    print("  T1 wrong + T2/T3/T4 correct -> stale first-token is the bug (ordering).")
    print("  T2/T3 wrong                 -> graph capture/replay itself broken.")
    print("  T3 wrong, T2 correct        -> side-stream refresh races replay.")
    print("-" * 72)


if __name__ == "__main__":
    main()
