# Copyright (c) 2026.
"""UT: FA3 prefill-state semantics against a plain-torch reference.

Covers the three prefill attention states that the vllm-ascend FA3 refactor
routes to flash-attention-npu (see design docs 01/02):

  * PrefillNoCache      -> flash_attn_varlen_func  (dense, no KV cache)
  * PrefillCacheHit     -> flash_attn_with_kvcache (paged, q_len == 1 per seq)
  * ChunkedPrefill      -> flash_attn_with_kvcache (paged, variable q_len,
                            causal mask aligned bottom-right against the
                            history + current chunk)

The reference implementation is plain float32 torch on NPU (gather from the
paged cache, explicit bottom-right causal mask, GQA head expansion), so these
tests validate *semantics*, independent of any CANN baseline.

test_metadata_bake_* additionally verify the graph-mode contract documented in
the design docs: scheduler_metadata baked with the paged-cache CAPACITY
(max_seqlen_k = page_size * page_table_width) stays correct for batches whose
real cache_seqlens are much shorter, and for block tables whose physical ids
are scattered (vllm allocator order) with a -1 tail.
"""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from flash_attn_npu_3 import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)

from vllm_ascend.attention.fa3_adapter import fa3_forward

BLOCK_SIZE = 128
_HEAD_DIM = 128
_TOL = 2e-2  # bf16 attention vs fp32 reference


def _cu_to_list(cu):
    return [int(x) for x in cu]


def ref_varlen_attention(q, k, v, cu_q, cu_k, num_kv_heads, scale, window_left=None):
    """Dense varlen reference with bottom-right causal alignment.

    q: (total_q, H, D) fp32; k/v: (total_k, Hkv, D) fp32.
    cu_q/cu_k: cumulative lengths WITH leading zero.
    Batch i's query token t attends to keys
    [0, ki - qi + t] (causal) intersected with the sliding window.
    """
    outs = []
    nheads = q.shape[1]
    for i in range(len(cu_q) - 1):
        q0, q1 = cu_q[i], cu_q[i + 1]
        k0, k1 = cu_k[i], cu_k[i + 1]
        qi, ki = q1 - q0, k1 - k0
        qb = q[q0:q1]  # (qi, H, D)
        kb = k[k0:k1]  # (ki, Hkv, D)
        vb = v[k0:k1]
        # GQA expand
        rep = nheads // num_kv_heads
        kb = kb.repeat_interleave(rep, dim=1)
        vb = vb.repeat_interleave(rep, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qb, kb) * scale  # (H, qi, ki)
        kv_offset = ki - qi
        q_idx = torch.arange(qi, device=q.device).unsqueeze(1)
        k_idx = torch.arange(ki, device=q.device).unsqueeze(0)
        keep = k_idx <= (q_idx + kv_offset)
        if window_left is not None:
            keep &= k_idx >= (q_idx + kv_offset - window_left)
        scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("hqk,khd->qhd", probs, vb)
        outs.append(out)
    return torch.cat(outs, dim=0)


def ref_paged_attention(q, k_cache, v_cache, block_table, cache_seqlens, cu_q, num_kv_heads, scale):
    """Paged reference: gather each seq's KV by block table, then bottom-right causal."""
    nheads = q.shape[1]
    page = k_cache.shape[1]
    outs = []
    for i in range(len(cache_seqlens)):
        ki = int(cache_seqlens[i])
        q0, q1 = cu_q[i], cu_q[i + 1]
        qi = q1 - q0
        blocks = [int(b) for b in block_table[i][: (ki + page - 1) // page]]
        k_seq = torch.cat([k_cache[b] for b in blocks], dim=0)[:ki]
        v_seq = torch.cat([v_cache[b] for b in blocks], dim=0)[:ki]
        rep = nheads // num_kv_heads
        k_seq = k_seq.repeat_interleave(rep, dim=1)
        v_seq = v_seq.repeat_interleave(rep, dim=1)
        qb = q[q0:q1]
        scores = torch.einsum("qhd,khd->hqk", qb, k_seq) * scale
        kv_offset = ki - qi
        q_idx = torch.arange(qi, device=q.device).unsqueeze(1)
        k_idx = torch.arange(ki, device=q.device).unsqueeze(0)
        keep = k_idx <= (q_idx + kv_offset)
        scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", probs, v_seq))
    return torch.cat(outs, dim=0)


def _rand_kv_cache(num_blocks, num_kv_heads, device):
    shape = (num_blocks, BLOCK_SIZE, num_kv_heads, _HEAD_DIM)
    return torch.randn(*shape, dtype=torch.float32, device=device) * 0.05


def _scatter_block_table(seq_lens, num_blocks, gen, device, tail_value=-1):
    """vllm-like block table: scattered physical ids, -1 in the unused tail."""
    max_blocks = (max(seq_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    perm = torch.randperm(num_blocks, generator=gen)[: max_blocks * len(seq_lens) + 4]
    table = torch.full((len(seq_lens), max_blocks + 1), tail_value, dtype=torch.int32)
    it = iter(perm.tolist())
    for i, L in enumerate(seq_lens):
        for j in range((L + BLOCK_SIZE - 1) // BLOCK_SIZE):
            table[i, j] = next(it)
    return table.to(device)


@pytest.fixture(autouse=True)
def _np_device():
    torch.npu.set_device(0)


@pytest.mark.parametrize("num_heads,num_kv", [(16, 4), (8, 1)])
def test_prefill_no_cache_dense_varlen(num_heads, num_kv):
    """PrefillNoCache: dense varlen, multi-batch variable lengths, causal GQA."""
    torch.manual_seed(0)
    device = "npu"
    seq_lens = [37, 129, 256]
    total_q = sum(seq_lens)
    q = torch.randn(total_q, num_heads, _HEAD_DIM, dtype=torch.float32, device=device) * 0.05
    k = torch.randn(total_q, num_kv, _HEAD_DIM, dtype=torch.float32, device=device) * 0.05
    v = torch.randn(total_q, num_kv, _HEAD_DIM, dtype=torch.float32, device=device) * 0.05
    cu = [0]
    for L in seq_lens:
        cu.append(cu[-1] + L)
    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)

    out = flash_attn_varlen_func(
        q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
        cu_t, cu_t, max(seq_lens), max(seq_lens),
        softmax_scale=_HEAD_DIM ** -0.5, causal=True,
    )
    ref = ref_varlen_attention(q, k, v, cu, cu, num_kv, _HEAD_DIM ** -0.5)
    diff = (out.float() - ref).abs().max().item()
    assert diff < _TOL, f"max diff {diff}"


@pytest.mark.parametrize("q_lens", [[1, 1, 1], [3, 1, 17]])
def test_chunked_prefill_kvcache(q_lens):
    """ChunkedPrefill / PrefillCacheHit: paged KV with history, variable q_len.

    history_i is nonzero so the causal mask must align bottom-right against
    history + current chunk (the chunked-prefill semantics of vllm).
    """
    torch.manual_seed(1)
    device = "npu"
    num_heads, num_kv = 16, 4
    history = [128, 65, 300]
    cache_seqlens = [h + q for h, q in zip(history, q_lens)]
    num_blocks = (sum(cache_seqlens) + BLOCK_SIZE - 1) // BLOCK_SIZE + 4
    gen = torch.Generator().manual_seed(7)
    k_cache = _rand_kv_cache(num_blocks, num_kv, device)
    v_cache = _rand_kv_cache(num_blocks, num_kv, device)
    table = _scatter_block_table(cache_seqlens, num_blocks, gen, device, tail_value=-1)

    total_q = sum(q_lens)
    q = torch.randn(total_q, num_heads, _HEAD_DIM, dtype=torch.float32, device=device) * 0.05
    # write the current chunk into the paged cache (as vllm reshape_and_cache does)
    q_off = 0
    for i, (h, ql) in enumerate(zip(history, q_lens)):
        flat = h
        for t in range(ql):
            block = int(table[i, flat // BLOCK_SIZE])
            slot = flat % BLOCK_SIZE
            k_cache[block, slot] = torch.randn(num_kv, _HEAD_DIM, device=device) * 0.05
            v_cache[block, slot] = torch.randn(num_kv, _HEAD_DIM, device=device) * 0.05
            flat += 1
        q_off += ql
    cu = [0]
    for ql in q_lens:
        cu.append(cu[-1] + ql)
    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
    cs_t = torch.tensor(cache_seqlens, dtype=torch.int32, device=device)

    out = flash_attn_with_kvcache(
        q.to(torch.bfloat16),
        k_cache.to(torch.bfloat16), v_cache.to(torch.bfloat16),
        cache_seqlens=cs_t,
        page_table=table,
        cu_seqlens_q=cu_t,
        max_seqlen_q=max(q_lens),
        softmax_scale=_HEAD_DIM ** -0.5,
        causal=True,
    )
    ref = ref_paged_attention(
        q, k_cache, v_cache, table, cache_seqlens, cu, num_kv, _HEAD_DIM ** -0.5,
    )
    diff = (out.float() - ref).abs().max().item()
    assert diff < _TOL, f"max diff {diff} (q_lens={q_lens})"


def test_metadata_bake_capacity_vs_actual():
    """Graph contract: metadata baked at paged CAPACITY is correct for short
    real cache_seqlens (and scattered block ids with -1 tail)."""
    torch.manual_seed(2)
    device = "npu"
    num_heads, num_kv = 16, 4
    cache_seqlens = [130, 64]  # far below capacity
    q_lens = [1, 1]
    num_blocks = 16  # capacity = table width * BLOCK_SIZE >> real lens
    gen = torch.Generator().manual_seed(11)
    k_cache = _rand_kv_cache(num_blocks, num_kv, device)
    v_cache = _rand_kv_cache(num_blocks, num_kv, device)
    table = _scatter_block_table(cache_seqlens, num_blocks, gen, device, tail_value=-1)

    q = torch.randn(2, num_heads, _HEAD_DIM, dtype=torch.float32, device=device) * 0.05
    for i, L in enumerate(cache_seqlens):
        for pos in range(L):
            block = int(table[i, pos // BLOCK_SIZE])
            k_cache[block, pos % BLOCK_SIZE] = torch.randn(num_kv, _HEAD_DIM, device=device) * 0.05
            v_cache[block, pos % BLOCK_SIZE] = torch.randn(num_kv, _HEAD_DIM, device=device) * 0.05

    cu_t = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    cs_t = torch.tensor(cache_seqlens, dtype=torch.int32, device=device)
    scale = _HEAD_DIM ** -0.5
    capacity = table.shape[1] * BLOCK_SIZE

    meta = get_scheduler_metadata(
        batch_size=len(cache_seqlens), max_seqlen_q=1, max_seqlen_k=capacity,
        num_heads_q=num_heads, num_heads_kv=num_kv, headdim=_HEAD_DIM,
        cache_seqlens=cs_t, qkv_dtype=torch.bfloat16, cu_seqlens_q=cu_t,
        page_size=BLOCK_SIZE, causal=True,
    )
    out_baked = flash_attn_with_kvcache(
        q.to(torch.bfloat16),
        k_cache.to(torch.bfloat16), v_cache.to(torch.bfloat16),
        cache_seqlens=cs_t, page_table=table, cu_seqlens_q=cu_t,
        max_seqlen_q=1, softmax_scale=scale, causal=True,
        scheduler_metadata=meta,
    )
    out_plain = flash_attn_with_kvcache(
        q.to(torch.bfloat16),
        k_cache.to(torch.bfloat16), v_cache.to(torch.bfloat16),
        cache_seqlens=cs_t, page_table=table, cu_seqlens_q=cu_t,
        max_seqlen_q=1, softmax_scale=scale, causal=True,
    )
    ref = ref_paged_attention(
        q, k_cache, v_cache, table, cache_seqlens, [0, 1, 2], num_kv, scale,
    )
    assert (out_baked.float() - out_plain.float()).abs().max().item() < _TOL
    assert (out_baked.float() - ref).abs().max().item() < _TOL


def test_adapter_forward_all_states():
    """fa3_forward routes the three prefill states with a mock attn_metadata."""
    torch.manual_seed(3)
    device = "npu"
    num_heads, num_kv = 16, 4
    history, q_len = [128, 65], [17, 3]
    cache_seqlens = [h + q for h, q in zip(history, q_len)]
    num_blocks = (sum(cache_seqlens) + BLOCK_SIZE - 1) // BLOCK_SIZE + 4
    gen = torch.Generator().manual_seed(13)
    k_cache = _rand_kv_cache(num_blocks, num_kv, device)
    v_cache = _rand_kv_cache(num_blocks, num_kv, device)
    table = _scatter_block_table(cache_seqlens, num_blocks, gen, device)

    total_q = sum(q_len)
    q = torch.randn(total_q, num_heads, _HEAD_DIM, dtype=torch.float32, device=device) * 0.05
    off = 0
    for i, (h, ql) in enumerate(zip(history, q_len)):
        for t in range(ql):
            pos = h + t
            block = int(table[i, pos // BLOCK_SIZE])
            k_cache[block, pos % BLOCK_SIZE] = torch.randn(num_kv, _HEAD_DIM, device=device) * 0.05
            v_cache[block, pos % BLOCK_SIZE] = torch.randn(num_kv, _HEAD_DIM, device=device) * 0.05
        off += ql

    cu = [0]
    for ql in q_len:
        cu.append(cu[-1] + ql)
    meta = SimpleNamespace(actual_seq_lengths_q=cu[1:])
    scale = _HEAD_DIM ** -0.5

    # ChunkedPrefill: paged, variable q_len, no scheduler_metadata (adapter builds internally)
    out = fa3_forward(
        q.to(torch.bfloat16),
        k_cache.to(torch.bfloat16).view(num_blocks, BLOCK_SIZE, -1),
        v_cache.to(torch.bfloat16).view(num_blocks, BLOCK_SIZE, -1),
        attn_metadata=meta, scale=scale, num_heads=num_heads, num_kv_heads=num_kv,
        head_size=_HEAD_DIM, causal=True, cache_mode=True,
        block_table=table, seq_lens_list=cache_seqlens,
    )
    ref = ref_paged_attention(q, k_cache, v_cache, table, cache_seqlens, cu, num_kv, scale)
    assert (out.float() - ref).abs().max().item() < _TOL

    # PrefillNoCache: dense varlen through the same adapter entry
    dense_k = torch.randn(total_q, num_kv, _HEAD_DIM, dtype=torch.float32, device=device) * 0.05
    dense_v = torch.randn(total_q, num_kv, _HEAD_DIM, dtype=torch.float32, device=device) * 0.05
    out2 = fa3_forward(
        q.to(torch.bfloat16), dense_k.to(torch.bfloat16), dense_v.to(torch.bfloat16),
        attn_metadata=meta, scale=scale, num_heads=num_heads, num_kv_heads=num_kv,
        head_size=_HEAD_DIM, causal=True, cache_mode=False,
    )
    ref2 = ref_varlen_attention(q, dense_k, dense_v, cu, cu, num_kv, scale)
    assert (out2.float() - ref2).abs().max().item() < _TOL


def test_sliding_window_chunked():
    """Sliding-window semantics on the paged path (window left, causal)."""
    torch.manual_seed(4)
    device = "npu"
    num_heads, num_kv = 16, 4
    window = 64
    history, q_len = [200, 100], [5, 2]
    cache_seqlens = [h + q for h, q in zip(history, q_len)]
    num_blocks = (sum(cache_seqlens) + BLOCK_SIZE - 1) // BLOCK_SIZE + 4
    gen = torch.Generator().manual_seed(17)
    k_cache = _rand_kv_cache(num_blocks, num_kv, device)
    v_cache = _rand_kv_cache(num_blocks, num_kv, device)
    table = _scatter_block_table(cache_seqlens, num_blocks, gen, device)

    total_q = sum(q_len)
    q = torch.randn(total_q, num_heads, _HEAD_DIM, dtype=torch.float32, device=device) * 0.05
    off = 0
    for i, (h, ql) in enumerate(zip(history, q_len)):
        for t in range(ql):
            pos = h + t
            block = int(table[i, pos // BLOCK_SIZE])
            k_cache[block, pos % BLOCK_SIZE] = torch.randn(num_kv, _HEAD_DIM, device=device) * 0.05
            v_cache[block, pos % BLOCK_SIZE] = torch.randn(num_kv, _HEAD_DIM, device=device) * 0.05
        off += ql
    cu = [0]
    for ql in q_len:
        cu.append(cu[-1] + ql)
    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
    cs_t = torch.tensor(cache_seqlens, dtype=torch.int32, device=device)
    scale = _HEAD_DIM ** -0.5

    out = flash_attn_with_kvcache(
        q.to(torch.bfloat16),
        k_cache.to(torch.bfloat16), v_cache.to(torch.bfloat16),
        cache_seqlens=cs_t, page_table=table, cu_seqlens_q=cu_t,
        max_seqlen_q=max(q_len), softmax_scale=scale, causal=True,
        window_size=(window, 0),
    )
    # reference with window: reuse dense ref on the gathered sequences
    outs = []
    for i, (h, ql) in enumerate(zip(history, q_len)):
        ki = cache_seqlens[i]
        blocks = [int(b) for b in table[i][: (ki + BLOCK_SIZE - 1) // BLOCK_SIZE]]
        k_seq = torch.cat([k_cache[b] for b in blocks], dim=0)[:ki].repeat_interleave(4, dim=1)
        v_seq = torch.cat([v_cache[b] for b in blocks], dim=0)[:ki].repeat_interleave(4, dim=1)
        qb = q[cu[i]:cu[i + 1]]
        scores = torch.einsum("qhd,khd->hqk", qb, k_seq) * scale
        kv_off = ki - ql
        qi = torch.arange(ql, device=q.device).unsqueeze(1)
        ki_idx = torch.arange(ki, device=q.device).unsqueeze(0)
        keep = (ki_idx <= qi + kv_off) & (ki_idx >= qi + kv_off - window)
        scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", probs, v_seq))
    ref = torch.cat(outs, dim=0)
    assert (out.float() - ref).abs().max().item() < _TOL
