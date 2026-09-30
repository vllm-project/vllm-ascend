# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

pytestmark = pytest.mark.skipif(
    "910" not in torch.npu.get_device_name(0),
    reason="Regression for the A2/A3 unquantized SFA kernel",
)


def _make_inputs(dtype, rope_dim, query_lengths, kv_lengths, selected_counts, kv_capacity=512):
    torch.manual_seed(930)
    heads, head_dim, block_size, sparse_capacity = 64, 512, 128, 2048
    batch_size = len(query_lengths)
    tokens = sum(query_lengths)
    query = torch.randn(tokens, heads, head_dim, dtype=dtype)
    key = torch.randn(batch_size, kv_capacity, head_dim, dtype=dtype)
    query_rope = torch.randn(tokens, heads, rope_dim, dtype=dtype) if rope_dim else None
    key_rope = torch.randn(batch_size, kv_capacity, rope_dim, dtype=dtype) if rope_dim else None

    # Logical pages are deliberately out of physical order. The reference
    # gathers logical tokens, independently of the device's paging code.
    blocks_per_request = kv_capacity // block_size
    page_order = torch.randperm(batch_size * blocks_per_request)
    block_table = page_order.reshape(batch_size, blocks_per_request).int()
    pages = torch.empty(batch_size * blocks_per_request, block_size, 1, head_dim, dtype=dtype)
    rope_pages = (
        torch.empty(batch_size * blocks_per_request, block_size, 1, rope_dim, dtype=dtype) if rope_dim else None
    )
    for batch in range(batch_size):
        for block in range(blocks_per_request):
            start = block * block_size
            pages[block_table[batch, block], :, 0] = key[batch, start : start + block_size]
            if rope_dim:
                rope_pages[block_table[batch, block], :, 0] = key_rope[batch, start : start + block_size]

    indices = torch.full((tokens, 1, sparse_capacity), -1, dtype=torch.int32)
    expected_output = torch.zeros(tokens, heads, head_dim, dtype=torch.float64)
    expected_lse = torch.full((tokens, heads), -torch.inf, dtype=torch.float64)
    token = 0
    for batch, query_length in enumerate(query_lengths):
        for _ in range(query_length):
            count = selected_counts[token]
            assert 0 <= count <= kv_lengths[batch]
            selected = torch.randperm(kv_lengths[batch])[:count].sort().values
            indices[token, 0, :count] = selected.int()
            if count:
                logits = query[token].double() @ key[batch, selected].double().T
                if rope_dim:
                    logits += query_rope[token].double() @ key_rope[batch, selected].double().T
                logits /= 24
                expected_output[token] = logits.softmax(-1) @ key[batch, selected].double()
                expected_lse[token] = logits.logsumexp(-1)
            token += 1

    device_inputs = {
        "query": query.npu(),
        "key": pages.npu(),
        "sparse_indices": indices.npu(),
        "block_table": block_table.npu(),
        "actual_seq_lengths_query": torch.tensor(query_lengths, dtype=torch.int32).cumsum(0).int().npu(),
        "actual_seq_lengths_kv": torch.tensor(kv_lengths, dtype=torch.int32).npu(),
        "query_rope": query_rope.npu() if rope_dim else None,
        "key_rope": rope_pages.npu() if rope_dim else None,
    }
    device_inputs["value"] = device_inputs["key"]
    return device_inputs, expected_output, expected_lse


def _run(inputs):
    return torch.ops._C_ascend.npu_sparse_flash_attention(
        **inputs,
        scale_value=1 / 24,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )


def _check(result, expected_output, expected_lse):
    assert result[1].dtype == result[2].dtype == torch.float32
    output, maximum, total = (tensor.cpu().double() for tensor in result)
    assert output.shape == expected_output.shape
    assert maximum.shape == total.shape == (1, *expected_lse.shape)
    lse = (maximum + total.log()).squeeze(0)
    empty = torch.isneginf(expected_lse[:, 0])
    # The existing BF16 AMLA accumulator starts with a 2**-80 bias.
    # Keep the established attention/LSE math outside this performance patch.
    torch.testing.assert_close(output[empty], expected_output[empty], atol=1e-20, rtol=0)
    torch.testing.assert_close(output[~empty], expected_output[~empty], atol=0.03, rtol=0.01)
    torch.testing.assert_close(lse[~empty], expected_lse[~empty], atol=0.005, rtol=0.001)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("selected_count", [1, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512])
def test_sparse_flash_attention_kv_padding(dtype, rope_dim, selected_count):
    assert enable_custom_op()
    inputs, output, lse = _make_inputs(dtype, rope_dim, [1], [512], [selected_count])
    _check(_run(inputs), output, lse)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rope_dim", [0, 64])
def test_sparse_flash_attention_kv_padding_pipeline(dtype, rope_dim):
    assert enable_custom_op()
    inputs, output, lse = _make_inputs(
        dtype,
        rope_dim,
        [7, 28],
        [4096, 4096],
        [0, 1, 513, 0, 1025, 2048, 0] * 5,
        kv_capacity=4096,
    )
    _check(_run(inputs), output, lse)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rope_dim", [0, 64])
def test_sparse_flash_attention_kv_padding_graph(dtype, rope_dim):
    assert enable_custom_op()
    inputs, _, _ = _make_inputs(dtype, rope_dim, [1], [512], [33])
    for _ in range(3):
        _run(inputs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = _run(inputs)
    for count in (1, 257, 512, 0, 33):
        updated, output, lse = _make_inputs(dtype, rope_dim, [1], [512], [count])
        inputs["sparse_indices"].copy_(updated["sparse_indices"])
        graph.replay()
        torch.npu.synchronize()
        _check(captured, output, lse)
        for actual, expected in zip(captured, _run(inputs), strict=True):
            assert torch.equal(actual.cpu().view(torch.uint8), expected.cpu().view(torch.uint8))
