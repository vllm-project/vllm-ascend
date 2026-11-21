# SPDX-License-Identifier: Apache-2.0
"""Check query ownership, causal rows and graph padding with fewer active cores."""

import math

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

pytestmark = pytest.mark.skipif("910" not in torch.npu.get_device_name(0), reason="Requires an A2/A3 NPU")

QUERY_LENGTHS = [
    (1,),
    (2,),
    (3,),
    (4,),
    (9,),
    (10,),
    (11,),
    (12,),
    (13,),
    (16,),
    (20,),
    (21,),
    (32,),
    (2, 1),
    (0, 3, 1),
]
HEAD_DIM = 512
KV_LENGTH = 4096


def make_inputs(dtype, lengths, capacity, page_size, heads=64, rope_dim=0, block_size=1, mode=3, kv_length=KV_LENGTH):
    torch.manual_seed(1001)
    total = sum(lengths)
    batches = len(lengths)
    physical_capacity = (KV_LENGTH + page_size - 1) // page_size * page_size
    query = torch.randn(total, heads, HEAD_DIM, dtype=dtype)
    query_rope = torch.randn(total, heads, rope_dim, dtype=dtype) if rope_dim else None
    logical = torch.randn(batches, physical_capacity, 1, HEAD_DIM, dtype=dtype)
    logical_rope = torch.randn(batches, physical_capacity, 1, rope_dim, dtype=dtype) if rope_dim else None
    pages = torch.empty(batches * physical_capacity // page_size, page_size, 1, HEAD_DIM, dtype=dtype)
    permutation = torch.randperm(pages.shape[0])
    table = permutation.reshape(batches, -1).int()
    pages[permutation] = logical.reshape_as(pages)
    rope_pages = torch.empty(*pages.shape[:-1], rope_dim, dtype=dtype) if rope_dim else None
    if rope_dim:
        rope_pages[permutation] = logical_rope.reshape_as(rope_pages)
    indices = torch.full((total, 1, capacity), -1, dtype=torch.int32)
    row = 0
    for length in lengths:
        for local_row in range(length):
            visible = kv_length - length + local_row + 1 if mode == 3 else kv_length
            # Fully valid blocks avoid mixing the existing partial-block issue
            # into this head-ownership regression test.
            available = max(0, visible // block_size)
            count = min(capacity, available)
            selected = torch.randperm(available)[:count]
            indices[row, 0, :count] = selected.int()
            row += 1
    scale = 1 / math.sqrt(HEAD_DIM + rope_dim)
    cpu = dict(
        query=query,
        logical=logical,
        query_rope=query_rope,
        logical_rope=logical_rope,
        indices=indices,
        lengths=lengths,
        scale=scale,
        block_size=block_size,
        mode=mode,
        kv_length=kv_length,
    )
    inputs = dict(
        query=query.npu(),
        key=pages.npu(),
        sparse_indices=indices.npu(),
        query_rope=query_rope.npu() if rope_dim else None,
        key_rope=rope_pages.npu() if rope_dim else None,
        block_table=table.npu(),
        actual_seq_lengths_query=torch.tensor(lengths, dtype=torch.int32).cumsum(0).int().npu(),
        actual_seq_lengths_kv=torch.full((batches,), kv_length, dtype=torch.int32).npu(),
        scale_value=scale,
        sparse_block_size=block_size,
        sparse_mode=mode,
        layout_query="TND",
        layout_kv="PA_BSND",
        attention_mode=2,
        return_softmax_lse=True,
    )
    inputs["value"] = inputs["key"]
    return inputs, cpu


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads", [1, 7, 16, 31, 63, 128])
@pytest.mark.parametrize("lengths", [(1,), (3,), (20,), (21,)])
@pytest.mark.parametrize("mode", [0, 3])
@torch.inference_mode()
def test_active_core_head_boundaries(dtype, heads, lengths, mode):
    assert enable_custom_op()
    inputs, cpu = make_inputs(dtype, lengths, 17, 128, heads=heads, mode=mode)
    check_outputs(torch.ops._C_ascend.npu_sparse_flash_attention(**inputs), cpu)


def check_outputs(outputs, cpu):
    output, maximum, total = [tensor.cpu().double() for tensor in outputs]
    assert output.shape == cpu["query"].shape
    expected_shape = (1, *cpu["query"].shape[:2])
    assert maximum.shape == total.shape == expected_shape
    lse = (maximum + total.log()).squeeze(0)
    row = 0
    for batch, length in enumerate(cpu["lengths"]):
        for local_row in range(length):
            visible = cpu["kv_length"] - length + local_row + 1 if cpu["mode"] == 3 else cpu["kv_length"]
            blocks = cpu["indices"][row, 0]
            blocks = blocks[blocks >= 0].long()
            selected = (blocks[:, None] * cpu["block_size"] + torch.arange(cpu["block_size"])).flatten()
            selected = selected[selected < visible]
            key = cpu["logical"][batch, selected, 0].double()
            if key.shape[0] == 0:
                # Empty-row LSE semantics have a separate regression fix.
                # Core scheduling must preserve the zero attention output.
                assert torch.count_nonzero(output[row]) == 0
                row += 1
                continue
            scores = cpu["query"][row].double() @ key.T
            if cpu["query_rope"] is not None:
                scores += cpu["query_rope"][row].double() @ cpu["logical_rope"][batch, selected, 0].double().T
            scores *= cpu["scale"]
            torch.testing.assert_close(output[row], scores.softmax(-1) @ key, atol=0.03, rtol=0.01)
            torch.testing.assert_close(lse[row], scores.logsumexp(-1), atol=0.005, rtol=0.001)
            row += 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lengths", QUERY_LENGTHS)
@pytest.mark.parametrize("page_size", [48, 128])
@pytest.mark.parametrize("capacity", [17, 2048])
@pytest.mark.parametrize("mode", [0, 3])
@torch.inference_mode()
def test_active_core_query_heads(dtype, lengths, page_size, capacity, mode):
    assert enable_custom_op()
    inputs, cpu = make_inputs(dtype, lengths, capacity, page_size, mode=mode)
    check_outputs(torch.ops._C_ascend.npu_sparse_flash_attention(**inputs), cpu)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", [0, 3])
@pytest.mark.parametrize("heads,rope_dim,block_size", [(32, 0, 1), (64, 64, 1), (64, 0, 2)])
@torch.inference_mode()
def test_active_core_fallbacks(dtype, mode, heads, rope_dim, block_size):
    assert enable_custom_op()
    inputs, cpu = make_inputs(dtype, (2, 1), 17, 128, heads, rope_dim, block_size, mode)
    check_outputs(torch.ops._C_ascend.npu_sparse_flash_attention(**inputs), cpu)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_active_core_graph_metadata(dtype):
    assert enable_custom_op()
    inputs, cpu = make_inputs(dtype, (4, 4), 17, 128)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = torch.ops._C_ascend.npu_sparse_flash_attention(**inputs)
    for lengths in [(4, 4), (0, 8), (2, 6)]:
        fresh, expected = make_inputs(dtype, lengths, 17, 128)
        for name in [
            "query",
            "key",
            "sparse_indices",
            "block_table",
            "actual_seq_lengths_query",
            "actual_seq_lengths_kv",
        ]:
            inputs[name].copy_(fresh[name])
        graph.replay()
        torch.npu.synchronize()
        check_outputs(outputs, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_active_core_zero_kv_heads(dtype):
    assert enable_custom_op()
    inputs, cpu = make_inputs(dtype, (1,), 17, 128)
    inputs["actual_seq_lengths_kv"].zero_()
    outputs = torch.ops._C_ascend.npu_sparse_flash_attention(**inputs)
    output, maximum, total = [tensor.cpu() for tensor in outputs]
    assert torch.count_nonzero(output) == 0
    assert torch.count_nonzero(maximum) == 0
    assert torch.count_nonzero(total) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", [0, 3])
@pytest.mark.parametrize("length", [1, 3])
@pytest.mark.parametrize("kv_length", [1, 4, 17, 127, 129, 2048])
@torch.inference_mode()
def test_active_core_short_kv_with_padded_topk(dtype, mode, length, kv_length):
    assert enable_custom_op()
    inputs, cpu = make_inputs(dtype, (length,), 2048, 128, mode=mode, kv_length=kv_length)
    check_outputs(torch.ops._C_ascend.npu_sparse_flash_attention(**inputs), cpu)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_active_core_graph_padded_queries(dtype):
    assert enable_custom_op()
    inputs, cpu = make_inputs(dtype, (4, 4), 17, 128)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = torch.ops._C_ascend.npu_sparse_flash_attention(**inputs)
    for lengths in [(4, 4), (0, 3), (0, 0), (2, 6)]:
        total_queries = sum(lengths)
        # TND does not initialize rows outside actual query lengths. Sentinels
        # make ownership observable without depending on allocator contents.
        sentinel = 0.125
        for tensor in outputs:
            tensor.fill_(sentinel)
        prefix = torch.tensor(lengths, dtype=torch.int32).cumsum(0).int()
        inputs["actual_seq_lengths_query"].copy_(prefix.npu())
        if total_queries:
            fresh, expected = make_inputs(dtype, lengths, 17, 128)
            for name in ["query", "sparse_indices"]:
                inputs[name][:total_queries].copy_(fresh[name])
            for name in ["key", "block_table", "actual_seq_lengths_kv"]:
                inputs[name].copy_(fresh[name])
        graph.replay()
        torch.npu.synchronize()
        if total_queries:
            output, maximum, sums = outputs
            check_outputs((output[:total_queries], maximum[:, :total_queries], sums[:, :total_queries]), expected)
        for index, tensor in enumerate(outputs):
            inactive = tensor[total_queries:] if index == 0 else tensor[:, total_queries:]
            assert torch.all(inactive.cpu() == sentinel)
