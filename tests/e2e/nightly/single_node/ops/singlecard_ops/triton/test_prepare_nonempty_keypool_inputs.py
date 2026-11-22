# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy tests for nonempty KPool inputs and their CANN integration.

Run under mssanitizer memcheck as well to detect writes past CANN's TND output.
"""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.kv_cache_interface import SparseCacheRole

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.glm5next.cache_views import view_hisparse_kpool_cache
from vllm_ascend.ops.cann_keypool import pool_key_indexer_nonempty
from vllm_ascend.ops.triton.glm5_next_cann_keypool import pack_keypool_indices
from vllm_ascend.ops.triton.prepare_nonempty_keypool_inputs import prepare_nonempty_keypool_inputs
from vllm_ascend.utils import bootstrap_custom_op_env


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize("tokens,requests,columns", [(1, 1, 1), (5, 2, 128), (65, 64, 1025), (257, 4, 65)])
@pytest.mark.parametrize("strided", [False, True])
def test_prepare_nonempty_keypool_inputs_matches_reference(tokens, requests, columns, strided):
    torch.manual_seed(1024)
    device = "npu:0"
    step = 2 if strided else 1
    query = torch.randn(tokens, 32, 128 * step, device=device, dtype=torch.bfloat16)[..., ::step]
    weights = torch.randn(tokens, 32 * step, device=device, dtype=torch.bfloat16)[:, ::step]
    ends = torch.arange(requests * step, device=device, dtype=torch.int64)[::step] * tokens // (requests * step)
    lengths = torch.arange(requests * step, device=device, dtype=torch.int64)[::step]
    tails = lengths % 4
    table = torch.arange(requests * columns * step, device=device, dtype=torch.int32).view(requests, -1)[:, ::step]
    inputs = (query, weights, ends, lengths, tails, table)
    originals = tuple(t.clone() for t in inputs)
    expected = (
        torch.cat((query, torch.zeros_like(query[:1]))),
        torch.cat((weights, torch.zeros_like(weights[:1]))),
        torch.cat((ends, ends.new_full((1,), tokens + 1))),
        torch.cat((lengths.clamp_min(1), lengths.new_ones((1,)))),
        torch.cat((tails, tails.new_zeros((1,)))),
        torch.cat((table, torch.zeros_like(table[:1]))),
    )
    actual = prepare_nonempty_keypool_inputs(*inputs)
    for result, reference in zip(actual, expected):
        assert result.is_contiguous()
        torch.testing.assert_close(result.cpu(), reference.cpu(), rtol=0, atol=0)
    for value, original in zip(inputs, originals):
        torch.testing.assert_close(value.cpu(), original.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="Requires Ascend A5 FP8")
def test_compact_hisparse_kpool_subpages_on_graph_replay():
    device = "npu:0"
    blocks, manager_tokens, pool = 3, 1536, 4
    manager_rows = manager_tokens // pool
    spec = AscendMLAAttentionSpec(
        block_size=manager_tokens,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        model_version="glm5_next",
        cache_role=SparseCacheRole.INDEXER,
        tokens_per_state=pool,
        state_content_bytes=132,
        scale_dim=1,
        scale_dtype=torch.float32,
    )
    raw = torch.zeros((blocks, 1, manager_rows, 132), dtype=torch.uint8, device=device)
    keys, scales = view_hisparse_kpool_cache(raw.view(spec.dtype), spec)
    subpages, kernel_rows = 12, 32
    assert keys.shape == (blocks * subpages, kernel_rows, 1, 128)
    assert scales.shape == (blocks * subpages, kernel_rows, 1, 1)
    assert keys.stride(0) == scales.stride(0) * scales.element_size() == 4224
    assert scales.data_ptr() - keys.data_ptr() == kernel_rows * spec.head_size

    # The writer consumes scalar pooled slots. This one is the first pooled row
    # of the second kernel page of manager block 1.
    slot = manager_rows + kernel_rows
    page, row = divmod(slot, keys.size(1))
    keys.view(torch.uint8)[page, row].fill_(33)
    scales[page, row].fill_(2)
    offset = (manager_rows + kernel_rows) * spec.state_content_size_bytes
    assert torch.all(raw.flatten()[offset : offset + 128] == 33)
    assert raw.flatten()[offset + 4096 : offset + 4100].view(torch.float32).item() == 2

    query = torch.zeros((2, 1, 128), dtype=torch.bfloat16, device=device)
    weights = torch.ones((2, 1), dtype=query.dtype, device=device)
    ends = torch.tensor([1, 2], dtype=torch.int64, device=device)
    lengths = torch.tensor([385, 32], dtype=torch.int64, device=device)
    tails = torch.zeros_like(lengths)
    table = torch.tensor([[2, 1], [1, 0]], dtype=torch.int32, device=device)

    def pad():
        return prepare_nonempty_keypool_inputs(query, weights, ends, lengths, tails, table, subpages)[-1]

    pad()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        expanded = pad()
    replacement = torch.tensor([[1, 2], [2, 0]], dtype=torch.int32)
    table.copy_(replacement)
    graph.replay()
    expected = torch.cat(
        (
            torch.stack((torch.arange(12, 36), torch.cat((torch.arange(24, 36), torch.arange(12))))),
            torch.zeros(1, 24, dtype=torch.int32),
        )
    ).to(torch.int32)
    torch.testing.assert_close(expanded.cpu(), expected)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize("graph_mode", [False, True])
def test_empty_pool_shards_and_padding(graph_mode):
    bootstrap_custom_op_env(include_vendor_lib=True)
    import cann_ops_transformer  # noqa: F401

    device = "npu:0"
    query = torch.zeros(5, 32, 128, dtype=torch.bfloat16, device=device)
    weights = torch.ones(5, 32, dtype=torch.bfloat16, device=device)
    cache = torch.zeros(16, 32, 1, 128, dtype=torch.bfloat16, device=device)
    table = torch.zeros(2, 128, dtype=torch.int32, device=device)
    ends = torch.tensor([3, 3], dtype=torch.int64, device=device)
    lengths = torch.zeros(2, dtype=torch.int64, device=device)
    tails = torch.tensor([3, 0], dtype=torch.int64, device=device)
    positions = torch.tensor([0, 1, 2, 0, 0], dtype=torch.int64, device=device)
    metadata = SimpleNamespace(positions=positions, cum_query_lens=ends)

    def run():
        indices, _ = pool_key_indexer_nonempty(
            torch.ops.cann_ops_transformer.pool_key_indexer,
            query,
            cache,
            weights,
            tails,
            actual_seq_q=ends,
            actual_seq_k=lengths,
            block_table=table,
            topk=2048,
            pool_size=4,
        )
        return pack_keypool_indices(indices, metadata, 2048, 4)

    run()
    torch.npu.synchronize()
    graph = None
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = run()
    scenarios = [
        ([3, 3], [0, 0], [3, 0], [0, 1, 2, 0, 0]),
        ([0, 3], [0, 0], [0, 3], [0, 1, 2, 0, 0]),
        ([0, 0], [0, 0], [0, 0], [0, 0, 0, 0, 0]),
        ([2, 4], [1, 2], [1, 1], [3, 4, 7, 8, 0]),
    ]
    for q, k, t, p in scenarios:
        for target, value in ((ends, q), (lengths, k), (tails, t), (positions, p)):
            target.copy_(torch.tensor(value, dtype=target.dtype, device=device))
        if graph is None:
            output = run()
        else:
            graph.replay()
        actual = output.cpu()
        expected = torch.full((5, 1, 2051), -1, dtype=torch.int32)
        for row in range(q[-1]):
            expected[row, 0, : p[row] + 1] = torch.arange(p[row] + 1, dtype=torch.int32)
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
def test_nonempty_topk_matches_unadapted_cann():
    bootstrap_custom_op_env(include_vendor_lib=True)
    import cann_ops_transformer  # noqa: F401

    torch.manual_seed(1024)
    device = "npu:0"
    query = torch.randn(3, 32, 128, dtype=torch.bfloat16, device=device)
    weights = torch.ones(3, 32, dtype=torch.bfloat16, device=device)
    cache = torch.randn(32, 32, 1, 128, dtype=torch.bfloat16, device=device)
    ends = torch.tensor([3], dtype=torch.int64, device=device)
    lengths = torch.tensor([700], dtype=torch.int64, device=device)
    tails = torch.zeros(1, dtype=torch.int64, device=device)
    table = torch.arange(32, dtype=torch.int32, device=device).view(1, 32)
    kwargs = dict(
        actual_seq_q=ends,
        actual_seq_k=lengths,
        block_table=table,
        topk=2048,
        pool_size=4,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
    )
    op = torch.ops.cann_ops_transformer.pool_key_indexer
    expected, _ = op(query, cache, weights, tails, **kwargs)
    actual, _ = pool_key_indexer_nonempty(op, query, cache, weights, tails, **kwargs)
    torch.testing.assert_close(actual.cpu(), expected.cpu())
