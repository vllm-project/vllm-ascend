# Copyright (c) 2026.
"""UT: FA3 eager decode attention vs a float64 CPU reference.

Covers the eager path production uses for DecodeOnly batches (PIECEWISE
mode): multi-request batches with variable KV lengths, reordered (non-
identity) block tables with vllm's -1 sentinel tail, GQA, long-KV split-K
behaviour, and the sliding-window mapping. Both with and without a
pre-computed scheduler_metadata are exercised.
"""

import pytest
import torch

from _util import (
    BLOCK_SIZE, DTYPE, HEAD_SIZE, HAS_FA3, NUM_HEADS, NUM_KV_HEADS,
    SCALE, cpu_ref_decode, fa3_kvcache, get_scheduler_metadata,
    make_block_table,
)

pytestmark = pytest.mark.skipif(not HAS_FA3, reason="flash-attention-npu not installed")

# Long enough seqlens to cross several 128-token blocks and to make the FA3
# kernel take its flash-decode (split-K) path in addition to the plain path.
SEQLENS = [512, 1024, 2048, 4096]
WIDTH = 128  # block-table width == max blocks per request
POOL = 128
# bf16 kernel output vs float64 reference: 0.05 abs is the empirically
# validated threshold for these magnitudes.
ATOL = 5e-2


def _run_decode(q, k_pool, v_pool, bt, seqlens, window=None, with_metadata=False):
    cache_seqlens = torch.tensor(seqlens, dtype=torch.int32).npu()
    cu_q = torch.arange(len(seqlens) + 1, dtype=torch.int32).npu()
    scheduler_metadata = None
    if with_metadata:
        scheduler_metadata = get_scheduler_metadata(
            batch_size=len(seqlens),
            max_seqlen_q=1,
            max_seqlen_k=bt.shape[1] * BLOCK_SIZE,
            num_heads_q=NUM_HEADS,
            num_heads_kv=NUM_KV_HEADS,
            headdim=HEAD_SIZE,
            cache_seqlens=cache_seqlens,
            qkv_dtype=DTYPE,
            cu_seqlens_q=cu_q,
            page_size=BLOCK_SIZE,
            causal=True,
        )
    return fa3_kvcache(
        q, k_pool, v_pool,
        cache_seqlens=cache_seqlens,
        page_table=bt.npu(),
        cu_seqlens_q=cu_q,
        max_seqlen_q=1,
        softmax_scale=SCALE,
        causal=True,
        window_size=(window, 0) if window is not None else (-1, -1),
        scheduler_metadata=scheduler_metadata,
    )


@pytest.fixture()
def pools():
    torch.manual_seed(0)
    k = torch.randn(POOL, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    v = torch.randn_like(k)
    return k, v


@pytest.fixture()
def q():
    torch.manual_seed(1)
    return torch.randn(len(SEQLENS), NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()


@pytest.mark.parametrize("with_metadata", [False, True])
def test_decode_variable_lengths_matches_reference(pools, q, with_metadata):
    bt = make_block_table(len(SEQLENS), SEQLENS, WIDTH, POOL, seed=7)
    out = _run_decode(q, *pools, bt, SEQLENS, with_metadata=with_metadata)
    ref = cpu_ref_decode(
        q.double().cpu(), pools[0].double().cpu(), pools[1].double().cpu(), bt, SEQLENS
    )
    for b, s in enumerate(SEQLENS):
        diff = (out[b].float().cpu() - ref[b]).abs().max().item()
        assert diff < ATOL, f"seq={s} row {b}: max abs diff {diff:.6f}"


def test_decode_sliding_window_matches_reference(pools, q):
    window = 768
    bt = make_block_table(len(SEQLENS), SEQLENS, WIDTH, POOL, seed=9)
    out = _run_decode(q, *pools, bt, SEQLENS, window=window)
    ref = cpu_ref_decode(
        q.double().cpu(), pools[0].double().cpu(), pools[1].double().cpu(),
        bt, SEQLENS, window=window,
    )
    for b, s in enumerate(SEQLENS):
        diff = (out[b].float().cpu() - ref[b]).abs().max().item()
        assert diff < ATOL, f"window={window} seq={s} row {b}: max abs diff {diff:.6f}"
