# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the bundled A3 FlashMLA reads Ling's shared component-major pages."""

import pytest
import torch
import torch_npu  # noqa: F401  # Register the NPU device backend.

import vllm_ascend.ops  # noqa: F401  # Register ops before importing device modules.
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.flash_mla_a3 import flash_mla_decode, prepare_decode
from vllm_ascend.ops.flash_mla_a3.graph_decode import (
    flash_mla_decode_graph,
    graph_decode_ops_available,
    prepare_decode_graph,
)

BLOCK_SIZE = 128
LATENT_DIM = 512
ROPE_DIM = 64
SOFTMAX_SCALE = 192**-0.5


def _paged_reference(query, cache, table, lengths):
    outputs = []
    for request, (query_chunk, length) in enumerate(zip(query, lengths)):
        if length == 0:
            outputs.append(torch.zeros(query_chunk.size(1), query_chunk.size(0), LATENT_DIM, dtype=torch.float64))
            continue
        page_count = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
        pages = cache[table[request, :page_count].long()]
        keys = pages.reshape(-1, LATENT_DIM + ROPE_DIM)[:length].double()
        scores = torch.einsum("thd,kd->htk", query_chunk.double(), keys) * SOFTMAX_SCALE
        last_key = length - query_chunk.size(0) + torch.arange(query_chunk.size(0))
        visible = torch.arange(length)[None, :] <= last_key[:, None]
        probs = scores.masked_fill(~visible[None, :, :], -torch.inf).softmax(dim=-1).nan_to_num()
        outputs.append(torch.einsum("htk,kd->htd", probs, keys[:, :LATENT_DIM]))
    return torch.cat(outputs, dim=1).float()


def _component_major_cache(cache):
    pages = cache.size(0)
    latent_size = BLOCK_SIZE * LATENT_DIM
    rope_size = BLOCK_SIZE * ROPE_DIM
    stride = latent_size + rope_size + 64
    host_backing = torch.full((pages, stride), float("nan"), dtype=torch.bfloat16)
    host_backing[:, :latent_size].copy_(cache[..., :LATENT_DIM].reshape(pages, latent_size))
    host_backing[:, latent_size : latent_size + rope_size].copy_(cache[..., LATENT_DIM:].reshape(pages, rope_size))
    backing = host_backing.npu()
    latent = backing.as_strided((pages, BLOCK_SIZE, 1, LATENT_DIM), (stride, LATENT_DIM, LATENT_DIM, 1))
    rope = backing.as_strided((pages, BLOCK_SIZE, 1, ROPE_DIM), (stride, ROPE_DIM, ROPE_DIM, 1), latent_size)
    return (latent, rope), (stride, stride)


@pytest.mark.parametrize("query_length", [1, 4])
@pytest.mark.parametrize("query_heads", [4, 8])
@pytest.mark.parametrize("layout_out", ["NTD", "TND"])
@pytest.mark.parametrize(
    ("lengths", "page_ids"),
    [
        ((33, 95), [[4], [2]]),
        ((257, 129), [[4, 1, 3], [2, 5, -1]]),
        ((0, 1), [[-1], [2]]),
        ((0, 1, 386), [[-1, -1, -1, -1], [2, -1, -1, -1], [4, 1, 3, 5]]),
    ],
    ids=["single-page", "multi-page", "single-page-masked", "multi-page-masked"],
)
def test_bailing_flashmla_reads_shared_strided_kv(query_length, query_heads, layout_out, lengths, page_ids):
    torch.npu.set_device(0)
    if not get_current_hardware_profile().supports(HardwareCapability.MLA_FLASH_A3_SPLIT_CACHE):
        pytest.skip("A3 FlashMLA is unavailable on this device")

    rng = torch.Generator().manual_seed(923)
    pages = 7
    cache = torch.randn(pages, BLOCK_SIZE, LATENT_DIM + ROPE_DIM, generator=rng).bfloat16()
    query = torch.randn(len(lengths), query_length, query_heads, LATENT_DIM + ROPE_DIM, generator=rng).bfloat16()
    table = torch.tensor(page_ids, dtype=torch.int32)

    cache_pair, strides = _component_major_cache(cache)
    assert all(not component.is_contiguous() for component in cache_pair)

    query_pair = tuple(part.contiguous().npu() for part in query.flatten(0, 1).split((LATENT_DIM, ROPE_DIM), dim=-1))
    plan = prepare_decode(
        table.npu(),
        lengths,
        query_length,
        pages,
        softmax_scale=SOFTMAX_SCALE,
        layout_out=layout_out,
        split_cache=True,
        cache_page_strides=strides,
        num_query_heads=query_heads,
    )
    result = flash_mla_decode(query_pair, cache_pair, plan, layout_out=layout_out)
    expected = _paged_reference(query, cache, table, lengths)
    if layout_out == "TND":
        expected = expected.transpose(0, 1)
    assert result.is_contiguous()
    torch.testing.assert_close(result.cpu().float(), expected, rtol=0.02, atol=0.004)


@pytest.mark.parametrize("query_length", [1, 4])
@pytest.mark.parametrize("query_heads", [4, 8])
def test_bailing_flashmla_graph_reads_updated_device_metadata(query_length, query_heads):
    torch.npu.set_device(0)
    if (
        not get_current_hardware_profile().supports(HardwareCapability.MLA_FLASH_A3_SPLIT_CACHE)
        or not graph_decode_ops_available()
    ):
        pytest.skip("A3 FlashMLA graph decode is unavailable on this device")

    rng = torch.Generator().manual_seed(517)
    pages, batches, columns = 7, 2, 4
    channels = LATENT_DIM + ROPE_DIM
    cache = torch.randn(pages, BLOCK_SIZE, channels, generator=rng).bfloat16()
    cache_pair, strides = _component_major_cache(cache)
    q_shape = (batches, query_length, query_heads, channels)
    query = torch.zeros(q_shape, dtype=torch.bfloat16)
    query_pair = tuple(part.contiguous().npu() for part in query.flatten(0, 1).split((LATENT_DIM, ROPE_DIM), dim=-1))
    # Logical rows are narrower than their physical stride, as in a view of
    # a standing graph input allocation. Changes must reach the captured op.
    table = torch.full((batches, columns + 2), -1, dtype=torch.int32, device="npu")[:, :columns]
    lengths = torch.zeros(batches, dtype=torch.int32, device="npu")
    plan = prepare_decode_graph(
        table,
        lengths,
        query_length,
        pages,
        num_query_heads=query_heads,
        cache_page_strides=strides,
        softmax_scale=SOFTMAX_SCALE,
    )
    assert plan.block_table is table and plan.cache_seqlens is lengths

    flash_mla_decode_graph(query_pair, cache_pair, plan)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = flash_mla_decode_graph(query_pair, cache_pair, plan)
    address = output.data_ptr()

    for current_lengths, page_ids in (
        ((386, 143), [[4, 1, 3, 5], [2, 5, -1, -1]]),
        ((0, 1), [[-1, -1, -1, -1], [6, -1, -1, -1]]),
        ((129, 385), [[6, 2, -1, -1], [1, 5, 4, 0]]),
        ((128, 0), [[3, -1, -1, -1], [-1, -1, -1, -1]]),
    ):
        query.normal_(generator=rng)
        for target, source in zip(query_pair, query.flatten(0, 1).split((LATENT_DIM, ROPE_DIM), dim=-1)):
            target.copy_(source)
        host_table = torch.tensor(page_ids, dtype=torch.int32)
        table.copy_(host_table)
        lengths.copy_(torch.tensor(current_lengths, dtype=torch.int32))
        graph.replay()
        result = output.cpu().float()
        assert output.data_ptr() == address
        expected = _paged_reference(query, cache, host_table, current_lengths)
        masked_rows = torch.arange(query_length)[None, :] + torch.tensor(current_lengths)[:, None] < query_length
        assert torch.count_nonzero(result[:, masked_rows.flatten()]) == 0
        torch.testing.assert_close(result, expected, rtol=0.02, atol=0.004)
