# Copyright (c) 2026.
"""UT: FA3 FULL-graph prefill contracts (mechanism validation).

Validates the two mechanisms the M2 design (prefill inside FULL aclgraph)
relies on, using the *installed* package name (flash_attn_npu_3, NOT the
legacy flash_attn_npu_3 the older diagnostic UTs probe):

  1. test_bake_upper_bound_max_seqlen_q: scheduler_metadata baked with an
     over-provisioned max_seqlen_q (the capture-bucket upper bound =
     num_tokens) stays CORRECT when the replayed batch has a different
     composition (per-seq q_len << baked bound, e.g. 1/3/5-token queries
     packed in the same 512-token bucket) — the kernel must derive real
     boundaries from cu_seqlens_q, not from max_seqlen_q.

  2. test_kvcache_npu_graph_capture_replay_prefill: torch.npu.NPUGraph
     records addresses; replaying a prefill-shaped flash_attn_with_kvcache
     after refreshing the shared int32 buffers (cache_seqlens /
     cu_seqlens_q / page_table, exactly what refresh_fa3_graph_params does)
     produces the same output as an eager call with the new data.  This is
     the driver-level capture contract for FA3 prefill.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from flash_attn_npu_3 import (
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)

BLOCK = 128
HEAD_DIM = 128
NQ, NKV = 16, 4
TOL = 2e-2


def _ref(q, k_cache, v_cache, table, cache_seqlens, cu_q, scale):
    """Paged attention reference, bottom-right causal, fp32."""
    outs = []
    for i in range(len(cache_seqlens)):
        ki = int(cache_seqlens[i])
        q0, q1 = cu_q[i], cu_q[i + 1]
        qi = q1 - q0
        blocks = [int(b) for b in table[i][: (ki + BLOCK - 1) // BLOCK]]
        ks = torch.cat([k_cache[b] for b in blocks], dim=0)[:ki].repeat_interleave(4, dim=1)
        vs = torch.cat([v_cache[b] for b in blocks], dim=0)[:ki].repeat_interleave(4, dim=1)
        sc = torch.einsum("qhd,khd->hqk", q[q0:q1], ks) * scale
        off = ki - qi
        qi_idx = torch.arange(qi, device=q.device).unsqueeze(1)
        ki_idx = torch.arange(ki, device=q.device).unsqueeze(0)
        keep = ki_idx <= qi_idx + off
        sc = sc.masked_fill(~keep.unsqueeze(0), float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", torch.softmax(sc, -1), vs))
    return torch.cat(outs, dim=0)


def _make_case(device, seed=0, batch=4, table_width=24):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    k_cache = torch.randn(64, BLOCK, NKV, HEAD_DIM, dtype=torch.float32, device=device) * 0.05
    v_cache = torch.randn(64, BLOCK, NKV, HEAD_DIM, dtype=torch.float32, device=device) * 0.05
    perm = torch.randperm(64, generator=gen)[: batch * table_width]
    it = iter(perm.tolist())
    return k_cache, v_cache, it


def _fill_chunk(k_cache, v_cache, table, history, q_len, device, gen):
    for i, (h, ql) in enumerate(zip(history, q_len)):
        for t in range(ql):
            pos = h + t
            if int(table[i, pos // BLOCK]) < 0:
                table[i, pos // BLOCK] = torch.randint(0, 64, (1,), generator=gen).item()
            b = int(table[i, pos // BLOCK])
            k_cache[b, pos % BLOCK] = torch.randn(NKV, HEAD_DIM, device=device) * 0.05
            v_cache[b, pos % BLOCK] = torch.randn(NKV, HEAD_DIM, device=device) * 0.05


@pytest.fixture(autouse=True)
def _np_device():
    torch.npu.set_device(0)


def test_bake_upper_bound_max_seqlen_q():
    """Bucket bound bake: metadata max_seqlen_q = 512 (bucket), real q_lens 1..5."""
    device = "npu"
    scale = HEAD_DIM ** -0.5
    q_lens = [1, 3, 5, 2]
    history = [130, 65, 300, 1]
    cache_seqlens = [h + q for h, q in zip(history, q_lens)]
    num_tokens = sum(q_lens)  # 11
    k_cache, v_cache, _ = _make_case(device, seed=3, batch=4)
    gen = torch.Generator().manual_seed(9)
    table = torch.full((4, 24), -1, dtype=torch.int32)
    it = iter(torch.randperm(64, generator=gen).tolist())
    for i, L in enumerate(cache_seqlens):
        for j in range((L + BLOCK - 1) // BLOCK):
            table[i, j] = next(it)
    table = table.to(device)
    _fill_chunk(k_cache, v_cache, table, history, q_lens, device, gen)

    q = torch.randn(num_tokens, NQ, HEAD_DIM, dtype=torch.float32, device=device) * 0.05
    cu = [0]
    for ql in q_lens:
        cu.append(cu[-1] + ql)
    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
    cs_t = torch.tensor(cache_seqlens, dtype=torch.int32, device=device)

    baked_bound = 512  # capture-bucket upper bound >> real max q_len (5)
    capacity = table.shape[1] * BLOCK
    meta = get_scheduler_metadata(
        batch_size=4, max_seqlen_q=baked_bound, max_seqlen_k=capacity,
        num_heads_q=NQ, num_heads_kv=NKV, headdim=HEAD_DIM,
        cache_seqlens=cs_t, qkv_dtype=torch.bfloat16, cu_seqlens_q=cu_t,
        page_size=BLOCK, causal=True,
    )
    out = flash_attn_with_kvcache(
        q.to(torch.bfloat16), k_cache.to(torch.bfloat16), v_cache.to(torch.bfloat16),
        cache_seqlens=cs_t, page_table=table, cu_seqlens_q=cu_t,
        max_seqlen_q=baked_bound, softmax_scale=scale, causal=True,
        scheduler_metadata=meta,
    )
    ref = _ref(q, k_cache, v_cache, table, cache_seqlens, cu, scale)
    diff = (out.float() - ref).abs().max().item()
    assert diff < TOL, f"bucket-bound bake diff {diff}"


def test_kvcache_npu_graph_capture_replay_prefill():
    """NPUGraph address capture: prefill FA3 replay with refreshed buffers
    matches an eager call with the same (new) batch data."""
    device = "npu"
    scale = HEAD_DIM ** -0.5
    batch = 3
    table_width = 24
    capacity = table_width * BLOCK

    # ---- capture batch: q_lens 4/2/1, histories vary ----
    q_lens_a, hist_a = [4, 2, 1], [128, 200, 60]
    cs_a = [h + q for h, q in zip(hist_a, q_lens_a)]
    k_cache = torch.randn(64, BLOCK, NKV, HEAD_DIM, dtype=torch.float32, device=device) * 0.05
    v_cache = torch.randn(64, BLOCK, NKV, HEAD_DIM, dtype=torch.float32, device=device) * 0.05
    gen = torch.Generator().manual_seed(21)
    table_a = torch.full((batch, table_width), 0, dtype=torch.int32)
    it = iter(torch.randperm(64, generator=gen).tolist())
    for i, L in enumerate(cs_a):
        for j in range((L + BLOCK - 1) // BLOCK):
            table_a[i, j] = next(it)
    table_a = table_a.to(device)
    _fill_chunk(k_cache, v_cache, table_a, hist_a, q_lens_a, device, gen)
    total_a = sum(q_lens_a)
    q = torch.randn(total_a, NQ, HEAD_DIM, dtype=torch.float32, device=device) * 0.05
    cu_a = [0]
    for ql in q_lens_a:
        cu_a.append(cu_a[-1] + ql)

    # fixed-size shared buffers (what _get_fa3_graph_params allocates)
    max_tokens = 8  # bucket size >= total_a (padded)
    cs_buf = torch.zeros(max_tokens, dtype=torch.int32, device=device)
    cu_buf = torch.zeros(max_tokens + 1, dtype=torch.int32, device=device)
    bt_buf = torch.zeros(max_tokens, table_width, dtype=torch.int32, device=device)

    cs_buf[:batch] = torch.tensor(cs_a, dtype=torch.int32, device=device)
    cu_buf[: batch + 1] = torch.tensor(cu_a, dtype=torch.int32, device=device)
    bt_buf[:batch] = table_a

    baked_max_q = max_tokens  # bucket upper bound
    meta = get_scheduler_metadata(
        batch_size=max_tokens, max_seqlen_q=baked_max_q, max_seqlen_k=capacity,
        num_heads_q=NQ, num_heads_kv=NKV, headdim=HEAD_DIM,
        cache_seqlens=cs_buf, qkv_dtype=torch.bfloat16, cu_seqlens_q=cu_buf,
        page_size=BLOCK, causal=True,
    )

    # capture: FA3 inside torch.npu.graph (driver level, no task group)
    q_static = torch.zeros(max_tokens, NQ, HEAD_DIM, dtype=torch.bfloat16, device=device)
    q_static[:total_a] = q.to(torch.bfloat16)
    g = torch.npu.NPUGraph()
    s = torch.npu.Stream()
    s.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(s):
        with torch.npu.graph(g):
            out_static = flash_attn_with_kvcache(
                q_static, k_cache.to(torch.bfloat16), v_cache.to(torch.bfloat16),
                cache_seqlens=cs_buf, page_table=bt_buf, cu_seqlens_q=cu_buf,
                max_seqlen_q=baked_max_q, softmax_scale=scale, causal=True,
                scheduler_metadata=meta,
            )
    torch.npu.current_stream().wait_stream(s)

    # ---- replay batch: different composition (q_lens 1/1/2), new data ----
    q_lens_b, hist_b = [1, 1, 2], [333, 90, 130]
    cs_b = [h + q for h, q in zip(hist_b, q_lens_b)]
    table_b = torch.full((batch, table_width), 0, dtype=torch.int32)
    it = iter(torch.randperm(64, generator=gen).tolist())
    for i, L in enumerate(cs_b):
        for j in range((L + BLOCK - 1) // BLOCK):
            table_b[i, j] = next(it)
    table_b = table_b.to(device)
    _fill_chunk(k_cache, v_cache, table_b, hist_b, q_lens_b, device, gen)
    total_b = sum(q_lens_b)
    q_b = torch.randn(total_b, NQ, HEAD_DIM, dtype=torch.float32, device=device) * 0.05
    cu_b = [0]
    for ql in q_lens_b:
        cu_b.append(cu_b[-1] + ql)

    # refresh shared buffers on the current stream (refresh_fa3_graph_params)
    cs_buf[:batch] = torch.tensor(cs_b, dtype=torch.int32, device=device)
    cu_buf[: batch + 1] = torch.tensor(cu_b, dtype=torch.int32, device=device)
    bt_buf[:batch] = table_b
    bt_buf[batch:].zero_()
    cs_buf[batch:].fill_(1)
    last = cu_b[-1]
    for i in range(batch + 1, max_tokens + 1):
        last += 1
        cu_buf[i] = last
    q_static[:total_b] = q_b.to(torch.bfloat16)
    q_static[total_b:].zero_()

    g.replay()
    replayed = out_static[:total_b].clone()

    # eager reference with the SAME refreshed buffers
    eager = flash_attn_with_kvcache(
        q_static, k_cache.to(torch.bfloat16), v_cache.to(torch.bfloat16),
        cache_seqlens=cs_buf, page_table=bt_buf, cu_seqlens_q=cu_buf,
        max_seqlen_q=baked_max_q, softmax_scale=scale, causal=True,
        scheduler_metadata=meta,
    )[:total_b]

    # plain fp32 reference of the REAL batch (padding rows excluded)
    ref = _ref(q_b, k_cache, v_cache, table_b, cs_b, cu_b, scale)

    d1 = (replayed.float() - eager.float()).abs().max().item()
    d2 = (replayed.float() - ref).abs().max().item()
    assert d1 < TOL, f"replay vs eager diff {d1}"
    assert d2 < TOL, f"replay vs reference diff {d2}"
