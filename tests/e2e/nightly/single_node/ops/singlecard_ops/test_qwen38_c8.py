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
    qsa_c8_indexer_cache_views,
    quantize_qsa_rows,
    quantize_qsa_values,
)
from vllm_ascend.ops.triton.qwen4_exp.c8 import quantize_qsa_query, store_qsa_c8_rows  # noqa: E402
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _initialize_triton():
    init_device_properties_triton()


@torch.inference_mode()
@pytest.mark.parametrize("indexer", [False, True])
def test_c8_store_rounding_strides_and_invalid_slots(indexer):
    torch.manual_seed(1)
    raw = torch.full((4, 2, 128, 1, 132 if indexer else 128), 19, dtype=torch.int8, device="npu")
    cache = raw[:, 1]
    rows = torch.randn(6, 1, 128, dtype=torch.bfloat16, device="npu")
    slots = torch.tensor([0, 127, 128, 511, -1, 512], dtype=torch.int64, device="npu")
    expected_codes, expected_scales = quantize_qsa_rows(rows)
    if indexer:
        store_qsa_c8_rows(cache, slots, rows, indexer=True)
        codes, scales = qsa_c8_indexer_cache_views(cache)
    else:
        key_scales = torch.full((4, 1, 128), 0.25, device="npu")
        store_qsa_c8_rows(cache, slots, rows, key_scales=key_scales)
        codes, scales = cache, key_scales.transpose(1, 2)
        assert (scales[2].cpu() == 0.25).all()
    for row, slot in enumerate([0, 127, 128, 511]):
        torch.testing.assert_close(codes[slot // 128, slot % 128].cpu(), expected_codes[row].cpu(), rtol=0, atol=0)
        torch.testing.assert_close(scales[slot // 128, slot % 128].cpu(), expected_scales[row].to(scales.dtype).cpu())
    assert (raw[:, 0].cpu() == 19).all()
    assert (cache[2].cpu() == 19).all()


def _dense_reference(query, key_cache, value_cache, key_scales, table, lengths, v_scale):
    codes_q, scales_q = quantize_qsa_rows(query.cpu())
    q = codes_q.float() * scales_q.unsqueeze(-1)
    kc, vc = key_cache.cpu(), value_cache.cpu()
    ks = key_scales.cpu().transpose(1, 2)
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
    # Preserve the page stride imposed by the shared K/V allocation.
    backing = torch.zeros(4, 2, 128, kv_heads, dim, dtype=torch.int8, device="npu")
    key_cache, value_cache = backing[:, 0], backing[:, 1]
    key_scales = torch.empty(4, kv_heads, 128, dtype=torch.float32, device="npu")
    slots = torch.arange(512, dtype=torch.int64, device="npu")
    v_scale = torch.rand(kv_heads, dim, device="npu") * 0.01 + 0.01
    p_scale = torch.tensor([127.0], device="npu")
    store_qsa_c8_rows(
        key_cache,
        slots,
        torch.randn(512, kv_heads, dim, dtype=torch.bfloat16, device="npu"),
        key_scales=key_scales,
    )
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
            query,
            key_cache,
            value_cache,
            compact,
            table,
            requests,
            positions,
            lengths,
            output,
            v_scale,
            p_scale,
            key_scales=key_scales,
        )

    expected = _dense_reference(query, key_cache, value_cache, key_scales, table, initial_lengths, v_scale)
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
    expected = _dense_reference(query, key_cache, value_cache, key_scales, table, replay_lengths, v_scale)
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
    cache = torch.full((3, 128, 2, 256), 37, dtype=torch.int8, device="npu")
    rows = torch.randn(5, 2, 256, dtype=torch.bfloat16, device="npu") * 100
    scales = torch.rand(2, 256, device="npu") * 0.1 + 0.01
    slots = torch.tensor([0, 127, 128, -1, 384], dtype=torch.int64, device="npu")
    store_qsa_c8_rows(cache, slots, rows, v_scale=scales)
    expected = quantize_qsa_values(rows, scales)
    for row, slot in enumerate((0, 127, 128)):
        torch.testing.assert_close(cache[slot // 128, slot % 128].cpu(), expected[row].cpu(), atol=0, rtol=0)
    assert (cache[2].cpu() == 37).all()


@pytest.mark.parametrize("heads, dim", [(8, 128), (3, 256)])
@pytest.mark.parametrize("tokens", [3, 96, 4096])
@torch.inference_mode()
def test_c8_query_fusion_matches_reference(heads, dim, tokens):
    generator = torch.Generator().manual_seed(84359)
    query = torch.randn(tokens, dim, heads, generator=generator).transpose(1, 2)
    query[0].zero_()
    query[1].mul_(1e-9)
    query[2, :, :6] = torch.tensor([127.0, -127.0, 0.5, 1.5, -0.5, -1.5])
    query = query.to(device="npu", dtype=torch.bfloat16)
    assert not query.is_contiguous()
    torch.testing.assert_close(quantize_qsa_query(query), quantize_qsa_rows(query), rtol=0, atol=0)


@pytest.mark.parametrize("heads, dim", [(8, 128), (3, 256)])
@torch.inference_mode()
def test_c8_query_fusion_graph_replay_matches_reference(heads, dim):
    torch.manual_seed(17)
    query = torch.randn(96, heads, dim, device="npu", dtype=torch.bfloat16)
    quantize_qsa_query(query)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = quantize_qsa_query(query)

    query.mul_(0.25)
    query[0].zero_()
    query[1, :, :6] = torch.tensor([127.0, -127.0, 0.5, 1.5, -0.5, -1.5], device="npu", dtype=torch.bfloat16)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(captured, quantize_qsa_rows(query), rtol=0, atol=0)
