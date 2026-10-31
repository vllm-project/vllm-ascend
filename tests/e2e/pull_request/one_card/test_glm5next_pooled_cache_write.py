# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton import glm5_next_cann_keypool as adapters


@pytest.mark.parametrize("cp_start", [None, 0, 4, 8])
@torch.inference_mode()
def test_cann_facing_metadata_and_tail_indices(cp_start):
    device = "npu"
    common = SimpleNamespace(
        num_reqs=3,
        num_actual_tokens=6,
        query_start_loc=torch.tensor([0, 3, 6, -1], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([65539, 65542, 0], dtype=torch.int32, device=device),
        positions=torch.tensor([65536, 65537, 65538, 65539, 65540, 65541], device=device),
        slot_mapping=torch.tensor([-1, 0, 1, 135, 136, 137], device=device),
        block_table_tensor=torch.tensor([[1, 9], [0, 9], [-1, 9]], dtype=torch.int32, device=device)[:, :1],
    )
    fields = {
        "positions": (8, torch.int64),
        "slot_mapping": (8, torch.int64),
        "seq_lens": (3, torch.int64),
        "raw_seq_lens": (3, torch.int32),
        "query_start_loc": (4, torch.int32),
        "cum_query_lens": (3, torch.int64),
        "start_pos": (3, torch.int32),
        "pool_tail": (3, torch.int64),
        "start_offsets": (3, torch.int32),
        "pooled_slot_mapping": (9, torch.int64),
    }
    buffers = SimpleNamespace(
        **{name: torch.full((size,), 77, dtype=dtype, device=device) for name, (size, dtype) in fields.items()}
    )
    query = (
        None
        if cp_start is None
        else SimpleNamespace(
            positions=buffers.positions[cp_start : cp_start + 4],
            **{
                name: torch.full((3,), 77, dtype=torch.int64, device=device)
                for name in ("cum_query_lens", "seq_lens", "pool_tail")
            },
        )
    )
    outputs = adapters.build_keypool_metadata(
        common, buffers, 8, 6, 128, 4, query_metadata=query, local_start=cp_start or 0
    )
    if query is not None:
        expected_query = {
            0: ([3, 4, 4], [16384, 16385, 0], [3, 0, 0]),
            4: ([0, 2, 2], [0, 16385, 0], [0, 2, 0]),
            8: ([0, 0, 0], [0, 0, 0], [0, 0, 0]),
        }[cp_start]
        for field, values in zip(("cum_query_lens", "seq_lens", "pool_tail"), expected_query):
            torch.testing.assert_close(
                getattr(query, field).cpu(), torch.tensor(values, dtype=torch.int64), rtol=0, atol=0
            )
    expected = (
        [65536, 65537, 65538, 65539, 65540, 65541, 0, 0],
        [-1, -1, -1, 33, -1, -1, -1, -1],
        [16384, 16385, 0],
        [65539, 65542, 0],
        [0, 3, 6, 6],
        [3, 6, 6],
        [65536, 65539, 0],
        [3, 2, 0],
    )
    for output, values in zip(outputs, expected):
        torch.testing.assert_close(output.cpu(), torch.tensor(values, dtype=output.dtype), rtol=0, atol=0)
    assert buffers.start_offsets.cpu().tolist() == [0, 3, 0]
    assert buffers.pooled_slot_mapping.cpu().tolist() == [-1, 33, -1, -1, -1, -1, 77, 77, 77]

    tail_buffers = {
        name: torch.full((3, 9), 77, dtype=torch.int64, device=device) for name in ("writes", "state", "retained")
    }
    tail_buffers["history"] = torch.empty(3, 4, dtype=torch.int64, device=device)
    ordinary = adapters.build_tail_inputs(common, 4, 4, 80, 0, tail_buffers)
    assert ordinary.history_indices.cpu().tolist() == [[80, 81, 82, 83], [0, 1, 2, 3], [0, 1, 2, 3]]
    assert ordinary.write_indices.cpu().tolist() == [[80, 81, 82, -1], [0, 1, -1, -1], [-1, -1, -1, -1]]
    assert ordinary.state_indices.cpu().tolist() == [[4, 5, 6, 7], [20, 21, 22, 23], [12, 13, 14, 15]]
    assert ordinary.retained_indices is None
    common.slot_mapping.copy_(torch.tensor([7, 8, 9, 0, 1, 14], device=device))
    speculative = adapters.build_tail_inputs(common, 4, 7, 80, 3, tail_buffers)
    assert speculative.history_indices.cpu().tolist() == [[82, 83, 84, 85], [2, 3, 4, 5], [0, 1, 2, 3]]
    assert speculative.retained_indices.cpu().tolist() == [[0, 1, 2], [3, 4, 5], [3, 4, 5]]
    assert speculative.write_indices.cpu().tolist() == [[82, 83, 84], [5, 6, 0], [-1, -1, -1]]
    common.slot_mapping[1] = -1
    speculative = adapters.build_tail_inputs(common, 4, 9, 80, 6, tail_buffers)
    assert speculative.write_indices.cpu().tolist() == [
        [-1, -1, -1, 87, -1, 80],
        [-1, -1, -1, 1, 2, 3],
        [-1, -1, -1, -1, -1, -1],
    ]
    suffix = torch.tensor([[87, -1, 80], [1, 2, 3], [-1, -1, -1]], device=device)
    for width in (1, 3, 9):
        inputs = adapters.build_tail_inputs(common, 4, 9, 80, width, tail_buffers)
        columns = min(width, 3)
        torch.testing.assert_close(inputs.write_indices[:, -columns:], suffix[:, -columns:], rtol=0, atol=0)
        if width > 3:
            assert (inputs.write_indices[:, :-3] == -1).all()


@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_cache_writes_with_changing_row_width_and_stride(use_graph):
    dim, device = 128, "npu"
    tail = torch.empty((2, 7, 2 * dim), device=device)
    cache = torch.empty((2, 32, 1, dim), dtype=torch.bfloat16, device=device)
    key = torch.arange(6 * dim, dtype=torch.float32, device=device).view(6, dim)
    gate = key + 1000
    pooled = torch.ones((1, dim), dtype=torch.bfloat16, device=device)
    metadata = SimpleNamespace(pooled_slot_mapping=torch.tensor([0], device=device))
    expected = torch.full_like(tail, 17)
    expected[0, :2] = torch.cat((key[:2], gate[:2]), dim=1)
    expected[1, :2] = torch.cat((key[2:4], gate[2:4]), dim=1)
    for width in (1, 3, 6, 1):
        storage = torch.empty((6 // width, width + 3), dtype=torch.int64, device=device)
        writes = storage[:, :width]
        writes.copy_(torch.tensor([0, 1, 7, 8, -1, 14], device=device).view(-1, width))
        inputs = SimpleNamespace(write_indices=writes, state_indices=None, page_rows=7)
        tail.fill_(17)
        cache.fill_(17)

        def run(inputs=inputs):
            adapters.write_keypool_caches(
                tail, cache, key, gate, pooled, metadata, SimpleNamespace(cann_inputs=inputs), None
            )

        run()
        if use_graph:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                run()
            graph.replay()
        torch.testing.assert_close(tail, expected, rtol=0, atol=0)
        torch.testing.assert_close(cache[0, 0, 0], pooled[0], rtol=0, atol=0)
        assert (cache[0, 1:] == 17).all() and (cache[1] == 17).all()


@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("speculative", [False, True])
@torch.inference_mode()
def test_padded_cache_writes_and_business_shape_index_packing(use_graph, speculative):
    dim, topk, device = 128, 2048, "npu"
    tail_storage = torch.full((2 * 80 * 2 * dim + 2 * dim,), 17.0, device=device)
    tail = tail_storage.as_strided((2, 7, 2 * dim), (80 * 2 * dim, 2 * dim, 1), dim)
    cache_storage = torch.full((3 * 48 * dim + 2 * dim,), 17, dtype=torch.bfloat16, device=device)
    cache = cache_storage.as_strided((3, 32, 1, dim), (48 * dim, dim, dim, 1), dim)
    key = torch.arange(5 * dim, dtype=torch.float32, device=device).view(5, dim)
    gate = key + 1000
    state = torch.cat((key, gate), dim=1)
    pooled = (
        torch.arange(4 * (dim + 64), dtype=torch.float32, device=device).view(4, dim + 64).to(torch.bfloat16)[:, :dim]
    )
    writes = torch.tensor([[6, 86, 160, 7, -1]], device=device)
    metadata = SimpleNamespace(
        pooled_slot_mapping=torch.tensor([0, 33, 96, -1], device=device),
        positions=torch.tensor([65536, 65539, 65540], device=device),
        cum_query_lens=torch.tensor([1, 3], device=device),
    )
    tail_metadata = SimpleNamespace(
        cann_inputs=SimpleNamespace(
            write_indices=writes, state_indices=torch.arange(5, device=device).view(1, 5), page_rows=80
        )
    )
    indices = torch.arange(2051, dtype=torch.int32, device=device).expand(5, -1).contiguous()
    output = torch.full((7, 2080), 17, dtype=torch.int32, device=device)

    def run():
        adapters.write_keypool_caches(
            tail,
            cache,
            key if speculative else None,
            gate if speculative else None,
            pooled,
            metadata,
            tail_metadata,
            state,
        )
        return adapters.pack_keypool_indices(indices, metadata, topk, 4, output)

    for _ in range(3):
        run()
    graph = torch.npu.NPUGraph() if use_graph else None
    if graph is not None:
        with torch.npu.graph(graph):
            result = run()
    for advance in (0, 1):
        writes.copy_(torch.tensor([[6 - advance, 86 - advance, 160, 7, -1]], device=device))
        metadata.positions.copy_(torch.tensor([65536 + advance, 65539, 65540 + advance], device=device))
        tail_storage.fill_(17)
        cache_storage.fill_(17)
        output.fill_(17)
        if graph is not None:
            graph.replay()
        else:
            result = run()
        expected_tail = torch.full_like(tail_storage, 17)
        expected_tail[dim + (6 - advance) * 2 * dim : dim + (7 - advance) * 2 * dim] = torch.cat((key[0], gate[0]))
        expected_tail[dim + (86 - advance) * 2 * dim : dim + (87 - advance) * 2 * dim] = torch.cat((key[1], gate[1]))
        expected_cache = torch.full_like(cache_storage, 17)
        expected_cache[dim : 2 * dim] = pooled[0]
        expected_cache[50 * dim : 51 * dim] = pooled[1]
        expected_output = torch.full_like(output, 17)
        expected_output[:5].fill_(-1)
        expected_output[:3, :topk] = torch.arange(topk, dtype=torch.int32, device=device)
        expected_output[:3, topk] = torch.tensor([65536, -1, 65540], dtype=torch.int32, device=device)
        if advance:
            expected_output[:3, topk + 1] = torch.tensor([65537, -1, 65541], dtype=torch.int32, device=device)
        torch.testing.assert_close(tail_storage, expected_tail, rtol=0, atol=0)
        torch.testing.assert_close(cache_storage, expected_cache, rtol=0, atol=0)
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
        assert result.shape == (5, 1, 2051)
        assert result.data_ptr() == output.data_ptr()
        key.add_(1)
        gate.add_(2)
        state.copy_(torch.cat((key, gate), dim=1))
        pooled.add_(3)


@torch.inference_mode()
def test_business_shape_index_packing_in_place():
    indices = torch.arange(2051, dtype=torch.int32, device="npu").expand(5, -1).contiguous()
    metadata = SimpleNamespace(
        positions=torch.tensor([65536, 65539, 65540], device="npu"),
        cum_query_lens=torch.tensor([1, 3], device="npu"),
    )
    result = adapters.pack_keypool_indices(indices, metadata, 2048, 4)
    expected = torch.full_like(indices, -1)
    expected[:3, :2048] = torch.arange(2048, dtype=torch.int32, device="npu")
    expected[:3, 2048] = torch.tensor([65536, -1, 65540], dtype=torch.int32, device="npu")
    torch.testing.assert_close(indices, expected, rtol=0, atol=0)
    assert result.data_ptr() == indices.data_ptr()
