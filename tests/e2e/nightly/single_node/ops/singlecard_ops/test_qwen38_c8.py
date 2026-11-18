# SPDX-License-Identifier: Apache-2.0
"""Synthetic QSA C8 tests; run only on a reserved Ascend 910B/A3 device."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401

enable_custom_op()

from vllm_ascend.models.qwen4_exp.c8_attention import qsa_c8_attention  # noqa: E402
from vllm_ascend.models.qwen4_exp.lightning_indexer import qsa_select_paged_tokens_quantized  # noqa: E402
from vllm_ascend.models.qwen4_exp.quantization import (  # noqa: E402
    qsa_c8_cache_views,
    quantize_qsa_rows,
    quantize_qsa_values,
)
from vllm_ascend.ops.triton.qwen4_exp.c8 import pack_qsa_c8_rows, store_qsa_c8_rows  # noqa: E402


@torch.inference_mode()
@pytest.mark.parametrize("indexer", [False, True])
def test_c8_store_rounding_strides_and_invalid_slots(indexer):
    torch.manual_seed(1)
    raw = torch.full((4, 2, 128, 1, 132), 19, dtype=torch.int8, device="npu")
    cache = raw[:, 1]
    rows = torch.randn(6, 1, 128, dtype=torch.bfloat16, device="npu")
    slots = torch.tensor([0, 127, 128, 511, -1, 512], dtype=torch.int64, device="npu")
    expected_codes, expected_scales = quantize_qsa_rows(rows)
    store_qsa_c8_rows(cache, slots, rows, indexer=indexer)
    codes, scales = qsa_c8_cache_views(cache, indexer=indexer)
    for row, slot in enumerate([0, 127, 128, 511]):
        torch.testing.assert_close(codes[slot // 128, slot % 128].cpu(), expected_codes[row].cpu(), rtol=0, atol=0)
        torch.testing.assert_close(scales[slot // 128, slot % 128].cpu(), expected_scales[row].to(scales.dtype).cpu())
    assert (raw[:, 0].cpu() == 19).all()
    assert (cache[2].cpu() == 19).all()


def _dense_reference(query, key_cache, value_cache, table, lengths, v_scale):
    codes_q, scales_q = quantize_qsa_rows(query.cpu())
    q = codes_q.float() * scales_q.unsqueeze(-1)
    kc, ks = qsa_c8_cache_views(key_cache.cpu())
    vc, vs = qsa_c8_cache_views(value_cache.cpu())
    expected = torch.zeros_like(q)
    repeats = query.shape[1] // kc.shape[2]
    for row, length in enumerate(lengths):
        token = torch.arange(length)
        page = table[row, token // 128].long().cpu()
        keys = (kc[page, token % 128].float() * ks[page, token % 128].unsqueeze(-1)).repeat_interleave(repeats, 1)
        values = (vc[page, token % 128].float() * v_scale.cpu()).repeat_interleave(repeats, 1)
        score = torch.einsum("hd,shd->hs", q[row], keys) / query.shape[-1] ** 0.5
        exp = (score - score.amax(-1, keepdim=True)).exp()
        expected[row] = torch.einsum("hs,shd->hd", (exp * 127).round() / 127, values) / exp.sum(-1, keepdim=True)
    return expected


@torch.inference_mode()
@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("row_count", [2, 129])
@pytest.mark.parametrize("kv_heads,query_heads", [(2, 4), (1, 3)])
def test_c8_attention_eager_and_changed_input_graph_replay(dim, row_count, kv_heads, query_heads):
    torch.manual_seed(2)
    query = torch.randn(row_count, query_heads, dim, dtype=torch.bfloat16, device="npu")
    key_cache = torch.zeros(4, 128, kv_heads, dim + 4, dtype=torch.int8, device="npu")
    value_cache = torch.zeros_like(key_cache)
    slots = torch.arange(512, dtype=torch.int64, device="npu")
    v_scale = torch.rand(kv_heads, dim, device="npu") * 0.01 + 0.01
    p_scale = torch.tensor([127.0], device="npu")
    store_qsa_c8_rows(key_cache, slots, torch.randn(512, kv_heads, dim, dtype=torch.bfloat16, device="npu"))
    store_qsa_c8_rows(
        value_cache, slots, torch.randn(512, kv_heads, dim, dtype=torch.bfloat16, device="npu"), v_scale=v_scale
    )
    row_indices = torch.arange(row_count, dtype=torch.int32, device="npu")
    table = torch.tensor([[3, 1], [2, 0]], dtype=torch.int32, device="npu")[row_indices.long() % 2]
    requests = row_indices
    initial_lengths = [131 + row % 7 for row in range(row_count)]
    lengths = torch.tensor(initial_lengths, dtype=torch.int32, device="npu")
    positions = lengths - 1
    compact = torch.full((row_count, 515), -1, dtype=torch.int32, device="npu")
    compact[:, :34] = torch.arange(34, dtype=torch.int32, device="npu")
    output = torch.empty_like(query)

    def run():
        return qsa_c8_attention(
            query, key_cache, value_cache, compact, table, requests, positions, lengths, output, v_scale, p_scale
        )

    expected = _dense_reference(query, key_cache, value_cache, table, initial_lengths, v_scale)
    torch.testing.assert_close(run().float().cpu(), expected, rtol=2e-2, atol=2e-2)
    # Warm up compilation and ACLNN tiling before entering graph capture.
    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    # Static shapes and addresses, changed query, visibility and physical pages.
    query.mul_(0.75)
    table.copy_(torch.tensor([[0, 2], [1, 3]], dtype=torch.int32, device="npu")[row_indices.long() % 2])
    replay_lengths = [129 + row % 5 for row in range(row_count)]
    lengths.copy_(torch.tensor(replay_lengths, dtype=torch.int32, device="npu"))
    positions.copy_(lengths - 1)
    graph.replay()
    torch.npu.synchronize()
    expected = _dense_reference(query, key_cache, value_cache, table, replay_lengths, v_scale)
    torch.testing.assert_close(output.float().cpu(), expected, rtol=2e-2, atol=2e-2)


@torch.inference_mode()
def test_c8_indexer_matches_all_visible_group_set():
    # With <512 visible groups, all complete groups must be selected exactly
    # once regardless of floating-point score rounding or tie order.
    query = torch.randn(2, 4, 128, dtype=torch.bfloat16, device="npu")
    cache = torch.zeros(4, 32, 1, 132, dtype=torch.int8, device="npu")
    store_qsa_c8_rows(
        cache,
        torch.arange(128, dtype=torch.int64, device="npu"),
        torch.randn(128, 1, 128, dtype=torch.bfloat16, device="npu"),
        indexer=True,
    )
    selected = qsa_select_paged_tokens_quantized(
        query,
        cache,
        torch.tensor([[3, 1], [2, 0]], dtype=torch.int32, device="npu"),
        torch.tensor([0, 1], dtype=torch.int32, device="npu"),
        torch.tensor([130, 136], dtype=torch.int32, device="npu"),
        torch.tensor([131, 137], dtype=torch.int32, device="npu"),
        torch.tensor([0, 1, 2], dtype=torch.int32, device="npu"),
        2048,
        4,
    ).cpu()
    for row, groups in enumerate((32, 34)):
        torch.testing.assert_close(selected[row, :groups].sort().values, torch.arange(groups, dtype=torch.int32))
        assert (selected[row, groups:512] == -1).all()


@torch.inference_mode()
def test_c8_static_v_store_channel_scales_and_saturation():
    cache = torch.full((3, 128, 2, 260), 37, dtype=torch.int8, device="npu")
    rows = torch.randn(5, 2, 256, dtype=torch.bfloat16, device="npu") * 100
    scales = torch.rand(2, 256, device="npu") * 0.1 + 0.01
    slots = torch.tensor([0, 127, 128, -1, 384], dtype=torch.int64, device="npu")
    store_qsa_c8_rows(cache, slots, rows, v_scale=scales)
    codes, _ = qsa_c8_cache_views(cache)
    expected = quantize_qsa_values(rows, scales)
    for row, slot in enumerate((0, 127, 128)):
        torch.testing.assert_close(codes[slot // 128, slot % 128].cpu(), expected[row].cpu(), atol=0, rtol=0)
    assert (cache[2].cpu() == 37).all()


@torch.inference_mode()
@pytest.mark.parametrize("dim,heads", [(128, 2), (256, 1)])
@pytest.mark.parametrize("rows", [1, 3, 128])
def test_c8_pack_matches_integer_gather_with_strided_pages(dim, heads, rows):
    """Check exact codes/scales, including residual tokens and invalid mappings."""
    torch.manual_seed(8)
    raw = torch.randint(-127, 128, (8, 2, 128, heads, dim + 4), dtype=torch.int8)
    cache = raw[:, 1]
    codes, scales = qsa_c8_cache_views(cache)
    scales.copy_(torch.rand(scales.shape) + 0.01)
    groups = torch.randint(0, 96, (heads, rows, 37), dtype=torch.int32)
    groups[..., 1] = -1
    groups[..., 2] = 1000
    counts = torch.randint(0, 38, (heads, rows), dtype=torch.int32)
    counts[:, 0] = 37
    visible = torch.full((rows,), 259, dtype=torch.int32)
    table = torch.tensor([[7, 2, 5]], dtype=torch.int32).repeat(rows, 1)
    if rows > 1:
        table[1, 0] = -1
        table[1, 1] = 8
        counts[:, -1] = 0
    token = torch.arange(256)
    group_slot = token // 4
    selected = groups.permute(1, 2, 0)[:, group_slot.clamp_max(36)].long()
    logical = selected * 4 + (token % 4)[None, :, None]
    columns = (logical // 128).clamp(0, 2)
    row_index = torch.arange(rows)[:, None, None]
    page = table[row_index, columns].long()
    valid = (group_slot[None, :, None] < counts.T[:, None, :]) & (group_slot[None, :, None] < 37)
    valid &= (selected >= 0) & (logical >= 0) & (logical < visible[:, None, None])
    valid &= (logical // 128 < 3) & (page >= 0) & (page < 8)
    safe_page, offset = page.clamp(0, 7), logical.clamp_min(0) % 128
    head = torch.arange(heads)[None, None, :]
    expected_codes = torch.where(valid[..., None], codes[safe_page, offset, head], 0)
    expected_scales = torch.where(valid, scales[safe_page, offset, head], 0)
    expected_scales = expected_scales.reshape(rows, 2, 128, heads).permute(0, 1, 3, 2).reshape(rows * 2, heads, 128)
    # Preserve an owner gap between pages instead of making the cache dense.
    device_raw = raw.to("npu")
    args = [x.to("npu") for x in (groups, counts, visible, table)]
    for copy_scales in (False, True):
        output = torch.empty(rows, 256, heads, dim, dtype=torch.int8, device="npu")
        output_scales = torch.empty(rows * 2, heads, 128, device="npu") if copy_scales else None
        pack_qsa_c8_rows(device_raw[:, 1], *args, output, output_scales)
        torch.testing.assert_close(output.cpu(), expected_codes, rtol=0, atol=0)
        if copy_scales:
            torch.testing.assert_close(output_scales.cpu(), expected_scales, rtol=0, atol=0)
