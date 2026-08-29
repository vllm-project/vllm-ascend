# Copyright (c) 2026.
"""UT: FA3 decode graph capture + replay with refreshed static buffers.

This is the core mechanism behind vllm-ascend's FA3 FULL decode graphs:
``flash_attn_with_kvcache`` is captured inside ``torch.npu.NPUGraph`` with
fixed-address buffers (cache_seqlens / cu_seqlens_q / block_table) and a
scheduler metadata baked for the max config. NPUGraph records addresses,
not values — before every replay the buffers are overwritten in-place on
the current stream (the production refresh pattern; a cross-stream
``wait_stream`` fails under FULL aclgraph replay) and the graph computes
from the new data.

Validated against the float64 CPU reference:
  - replay of a batch that differs completely from the warmup batch
    (variable long KV lengths, reordered block ids);
  - a second replay of yet another batch (refresh keeps working);
  - a padded batch: fewer real requests than the capture size, padding
    rows zeroed and padding cache_seqlens set to 1, mirroring
    AscendAttentionBackendImpl.refresh_fa3_graph_params.
"""

import pytest
import torch

from _util import (
    BLOCK_SIZE, DTYPE, HEAD_SIZE, HAS_FA3, NUM_HEADS, NUM_KV_HEADS,
    SCALE, cpu_ref_decode, fa3_kvcache, get_scheduler_metadata,
    make_block_table,
)

pytestmark = pytest.mark.skipif(not HAS_FA3, reason="flash-attention-npu not installed")

BATCH = 4
WIDTH = 128  # block-table width == baked row stride (max 16384-token KV)
POOL = 128
WARMUP_SEQLENS = [16, 16, 16, 16]  # deliberately unlike the replay batches
ATOL = 5e-2


@pytest.fixture(scope="module")
def pools():
    torch.manual_seed(0)
    k = torch.randn(POOL, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    return k, v


@pytest.fixture(scope="module")
def captured_graph(pools):
    """Capture FA3 decode with max-config static buffers (warmup data)."""
    q_buf = torch.zeros(BATCH, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    cache_seqlens_buf = torch.tensor(WARMUP_SEQLENS, dtype=torch.int32).npu()
    cu_q_buf = torch.arange(BATCH + 1, dtype=torch.int32).npu()
    block_table_buf = torch.zeros(BATCH, WIDTH, dtype=torch.int32).npu()

    # max_seqlen_k = width * block_size bakes the block-table row stride
    # equal to the buffer width — the constraint production relies on.
    meta = get_scheduler_metadata(
        batch_size=BATCH,
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

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = fa3_kvcache(
            q_buf, pools[0], pools[1],
            cache_seqlens=cache_seqlens_buf,
            page_table=block_table_buf,
            cu_seqlens_q=cu_q_buf,
            max_seqlen_q=1,
            softmax_scale=SCALE,
            causal=True,
            window_size=(-1, -1),
            scheduler_metadata=meta,
        )
    torch.npu.synchronize()
    # meta and every captured input buffer must stay alive for the whole
    # module: the captured kernel keeps reading them, and freeing any of
    # them makes replays fault (507011).
    return graph, out, q_buf, cache_seqlens_buf, cu_q_buf, block_table_buf, meta


def _replay_and_check(captured_graph, pools, q, bt, seqlens, real_batch=None):
    graph, out, q_buf, cache_seqlens_buf, _cu_q, block_table_buf, _meta = captured_graph

    # Production refresh: in-place copies on the current stream, then a
    # sync before the replay — without it the replay's host-args kernel
    # launches overtake the pending copies and read a half-written
    # block_table (MTE fault 507011). A padded batch (real_batch < BATCH)
    # zeroes its padding block rows and gives padding requests a dummy KV
    # length of 1.
    n = BATCH if real_batch is None else real_batch
    q_buf.copy_(q)
    cache_seqlens_buf[:n].copy_(torch.tensor(seqlens, dtype=torch.int32).npu())
    if n < BATCH:
        cache_seqlens_buf[n:].fill_(1)
    block_table_buf.copy_(bt.npu())
    if n < BATCH:
        block_table_buf[n:].zero_()
    torch.npu.synchronize()

    graph.replay()
    torch.npu.synchronize()

    ref = cpu_ref_decode(
        q.double().cpu(), pools[0].double().cpu(), pools[1].double().cpu(),
        bt[:n], seqlens,
    )
    for b, s in enumerate(seqlens):
        diff = (out[b].float().cpu() - ref[b]).abs().max().item()
        assert diff < ATOL, f"replay seq={s} row {b}: max abs diff {diff:.6f}"


def test_replay_batch_differs_from_warmup(captured_graph, pools):
    torch.manual_seed(1)
    seqlens = [512, 1024, 2048, 4096]
    q = torch.randn(BATCH, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    bt = make_block_table(BATCH, seqlens, WIDTH, POOL, seed=7)
    _replay_and_check(captured_graph, pools, q, bt, seqlens)


def test_replay_second_batch(captured_graph, pools):
    torch.manual_seed(2)
    seqlens = [256, 768, 1536, 3072]
    q = torch.randn(BATCH, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    bt = make_block_table(BATCH, seqlens, WIDTH, POOL, seed=8)
    _replay_and_check(captured_graph, pools, q, bt, seqlens)


def test_replay_padded_batch(captured_graph, pools):
    """Two real requests in a graph captured for four (padding semantics of
    refresh_fa3_graph_params: padding rows -> block 0, cache_seqlens -> 1)."""
    torch.manual_seed(3)
    seqlens = [1024, 3584]
    q = torch.randn(BATCH, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    bt = make_block_table(BATCH, seqlens, WIDTH, POOL, seed=9)
    _replay_and_check(captured_graph, pools, q, bt, seqlens, real_batch=2)
