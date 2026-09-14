# SPDX-License-Identifier: Apache-2.0
"""Independent CPU-float64 reference for native A5 GQA, including real strides.

Run only on a coordinated, idle A5 device with the selected CANN package.
The reference never calls another attention kernel or the PR15336 backend.
"""

import pytest
import torch
import torch_npu  # noqa: F401
from cann_ops_transformer.ops import flash_attn, flash_attn_metadata


def golden(q, key, value, table, cu, used, lengths, causal):
    """Explicit logical block gathering and independently scaled softmax."""
    q, key, value = (x.detach().cpu().to(torch.float64) for x in (q, key, value))
    table, cu, used, lengths = (x.detach().cpu().tolist() for x in (table, cu, used, lengths))
    result = torch.zeros_like(q)
    lse = torch.zeros(q.shape[1], q.shape[0], dtype=torch.float64)
    live = torch.zeros(q.shape[0], dtype=torch.bool)
    page = key.shape[2]
    for batch, (count, kvlen) in enumerate(zip(used, lengths)):
        if count == 0 or kvlen == 0:
            continue
        indices = [(table[batch][i // page], i % page) for i in range(kvlen)]
        k = torch.stack([key[block, :, token] for block, token in indices]).repeat_interleave(
            q.shape[1] // key.shape[1], 1
        )
        v = torch.stack([value[block, :, token] for block, token in indices]).repeat_interleave(
            q.shape[1] // key.shape[1], 1
        )
        queries = q[cu[batch] : cu[batch] + count]
        scores = torch.einsum("thd,shd->hts", queries, k) / 8
        if causal:
            absolute_query = kvlen - count + torch.arange(count)
            scores.masked_fill_(torch.arange(kvlen)[None, None, :] > absolute_query[None, :, None], -torch.inf)
        result[cu[batch] : cu[batch] + count] = torch.einsum("hts,shd->thd", scores.softmax(-1), v)
        lse[:, cu[batch] : cu[batch] + count] = scores.logsumexp(-1)
        live[cu[batch] : cu[batch] + count] = True
    return result, lse, live


def make_inputs(dtype, page, causal):
    torch.manual_seed(16464)
    device = "npu:0"
    pages, heads, dim = 4, 2, 64
    payload = heads * page * dim
    # K and V deliberately have different page pitches and nonzero offsets.
    kraw = torch.full((pages * (3 * payload + 128) + 256,), -91, dtype=dtype, device=device)
    vraw = torch.full((pages * (4 * payload + 256) + 512,), -73, dtype=dtype, device=device)
    k = kraw.as_strided((pages, heads, page, dim), (3 * payload + 128, page * dim, dim, 1), 64)
    v = vraw.as_strided((pages, heads, page, dim), (4 * payload + 256, page * dim, dim, 1), 128)
    k.copy_(torch.randn(k.shape, dtype=dtype).to(device))
    v.copy_(torch.randn(v.shape, dtype=dtype).to(device))
    q = torch.randn(20, 8, 64, dtype=dtype).to(device)
    ints = lambda data: torch.tensor(data, dtype=torch.int32, device=device)
    cu, used, lens = ints([0, 7, 14, 18, 20]), ints([7, 4, 0, 0]), ints([page + 5, 11, 0, 0])
    table = ints([[2, 0], [1, 3], [0, 0], [0, 0]])
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device=device), diagonal=1) if causal else None
    out, lse = torch.empty_like(q), torch.empty(8, 20, dtype=torch.float32, device=device)

    def run():
        schedule = flash_attn_metadata(
            8, 2, 64, batch_size=4, cu_seqlens_q=cu, seqused_q=used,
            seqused_kv=lens, max_seqlen_q=7, max_seqlen_kv=page * 2,
            mask_mode=3 if causal else 0,
            layout_q="TND", layout_kv="PA_BNBD", layout_out="TND",
        )
        attention_out, softmax_lse = flash_attn(
            q, k, v, block_table=table, cu_seqlens_q=cu, seqused_q=used,
            seqused_kv=lens, attn_mask=mask, metadata=schedule,
            softmax_scale=0.125, mask_mode=3 if causal else 0,
            max_seqlen_q=7, max_seqlen_kv=page * 2,
            layout_q="TND", layout_kv="PA_BNBD", layout_out="TND",
            return_softmax_lse=True,
        )
        out.copy_(attention_out)
        lse.copy_(softmax_lse)

    return q, k, v, table, cu, used, lens, out, lse, run


def assert_golden(inputs, causal):
    q, k, v, table, cu, used, lens, out, lse, _ = inputs
    torch.npu.synchronize()
    expected, expected_lse, live = golden(q, k, v, table, cu, used, lens, causal)
    atol, rtol = (0.008, 0.015) if q.dtype == torch.bfloat16 else (0.002, 0.005)
    observed = out.detach().cpu().to(torch.float64)
    assert torch.isfinite(observed[live]).all()
    torch.testing.assert_close(observed[live], expected[live], atol=atol, rtol=rtol)
    torch.testing.assert_close(lse.cpu().to(torch.float64)[:, live], expected_lse[:, live], atol=0.015, rtol=0.005)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("page", [16, 128, 768])
@pytest.mark.parametrize("causal", [False, True])
def test_native_padded_page_reader_matches_independent_float64(dtype, page, causal):
    inputs = make_inputs(dtype, page, causal)
    inputs[-1]()
    assert_golden(inputs, causal)
    # In-place accepted-length changes and a request/block reorder must force
    # the metadata producer to describe the current device epoch.
    q, _, _, table, _, used, lens, *_ = inputs
    q.mul_(0.75)
    table.copy_(table[[1, 0, 2, 3]])
    lens.copy_(torch.tensor([11, page + 3, 0, 0], dtype=torch.int32, device=q.device))
    used.copy_(torch.tensor([3, 7, 0, 0], dtype=torch.int32, device=q.device))
    inputs[-1]()
    assert_golden(inputs, causal)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("page", [16, 768])
def test_native_writer_updates_only_owned_slots_and_heads(dtype, page):
    payload = 4 * page * 64
    pitch, offset = 3 * payload + 128, 64
    raw = torch.full((4 * pitch + 512,), -91, dtype=dtype, device="npu:0")
    cache = raw.as_strided((4, 4, page, 64), (pitch, page * 64, 64, 1), offset)
    kc, vc = cache[:, :2], cache[:, 2:]
    torch.manual_seed(42)
    # Match DFlash split QKV: dense last axis, strided current token rows.
    current = torch.randn(7, 8 * 64, dtype=dtype, device="npu:0")
    k, v = current[:, :128].view(7, 2, 64), current[:, 128:256].view(7, 2, 64)
    slot_values = [2 * page + 2, -1, 0, page + 7, -1, 3 * page + 9, page - 1]
    slots = torch.tensor(slot_values, dtype=torch.int64, device="npu:0")
    expected = raw.cpu()
    kh, vh = k.cpu(), v.cpu()
    for row, slot in enumerate(slot_values):
        if slot < 0:
            continue
        for head in range(2):
            for base_head, source in ((head, kh), (head + 2, vh)):
                start = offset + slot // page * pitch + (base_head * page + slot % page) * 64
                expected[start : start + 64] = source[row, head]
    torch_npu.npu_scatter_pa_kv_cache(
        k, v, kc.permute(0, 2, 1, 3), vc.permute(0, 2, 1, 3), slots, cache_mode="Norm"
    )
    torch.npu.synchronize()
    torch.testing.assert_close(raw.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("page", [16, 768])
def test_native_graph_recomputes_metadata_after_length_and_block_changes(page):
    inputs = make_inputs(torch.bfloat16, page, False)
    for _ in range(3):
        inputs[-1]()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        inputs[-1]()
    graph.replay()
    assert_golden(inputs, False)
    q, k, v, table, _, used, lens, *_ = inputs
    q.mul_(0.5)
    k.add_(0.1)
    v.mul_(0.75)
    table.copy_(table[[1, 0, 2, 3]])
    lens.copy_(torch.tensor([page + 1, 8, 0, 0], dtype=torch.int32, device=q.device))
    used.copy_(torch.tensor([5, 3, 0, 0], dtype=torch.int32, device=q.device))
    graph.replay()
    assert_golden(inputs, False)
    # Empty graph epochs must clear scheduling state before later work.
    lens.zero_()
    used.zero_()
    graph.replay()
    torch.npu.synchronize()
    q.mul_(0.8)
    lens.copy_(torch.tensor([0, page + 2, 9, 0], dtype=torch.int32, device=q.device))
    used.copy_(torch.tensor([0, 4, 3, 0], dtype=torch.int32, device=q.device))
    table.copy_(table[[0, 2, 1, 3]])
    graph.replay()
    assert_golden(inputs, False)
