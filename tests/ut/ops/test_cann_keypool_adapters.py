# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ops.triton import glm5_next_cann_keypool as adapters


@pytest.fixture(autouse=True)
def mock_triton_launch_config(monkeypatch):
    triton = MagicMock()
    triton.cdiv = MagicMock(wraps=lambda x, y: (x + y - 1) // y)
    triton.next_power_of_2 = MagicMock(wraps=lambda x: 1 << (max(x, 1) - 1).bit_length())
    monkeypatch.setattr(adapters, "triton", triton)


@pytest.mark.parametrize("retained_width,cp_start", [(0, None), (3, 2)])
def test_metadata_launches_use_preallocated_request_and_tail_buffers(monkeypatch, retained_width, cp_start):
    launch = MagicMock()
    monkeypatch.setattr(adapters, "_build_keypool_metadata_kernel", launch)
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=3,
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        seq_lens=torch.tensor([65538, 65539], dtype=torch.int32),
        positions=torch.empty(3, dtype=torch.int64),
        slot_mapping=torch.empty(3, dtype=torch.int64),
        block_table_tensor=torch.empty(2, 1, dtype=torch.int32),
    )
    fields = (
        "positions",
        "slot_mapping",
        "seq_lens",
        "raw_seq_lens",
        "query_start_loc",
        "cum_query_lens",
        "start_pos",
        "pool_tail",
    )
    buffers = SimpleNamespace(
        **{name: torch.empty(5, dtype=torch.int64) for name in (*fields, "start_offsets", "pooled_slot_mapping")}
    )
    query = (
        None
        if cp_start is None
        else SimpleNamespace(
            positions=buffers.positions[cp_start:4],
            **{name: torch.empty(2, dtype=torch.int64) for name in ("cum_query_lens", "seq_lens", "pool_tail")},
        )
    )
    result = adapters.build_keypool_metadata(
        common, buffers, 4, 3, 128, 4, query_metadata=query, local_start=cp_start or 0
    )
    for view, name, size in zip(result, fields, (4, 4, 2, 2, 3, 2, 2, 2)):
        assert view.data_ptr() == getattr(buffers, name).data_ptr()
        assert view.shape == (size,)
    assert launch.__getitem__.return_value.call_count == 1
    args = launch.__getitem__.return_value.call_args.args
    assert args[0] is common.query_start_loc
    assert args[1] is common.seq_lens
    assert args[2] is common.positions
    assert args[3] is common.slot_mapping
    if query is None:
        assert args[-6:] == (None, None, None, 0, 0, False)
    else:
        assert args[-6] is query.cum_query_lens
        assert args[-5] is query.seq_lens
        assert args[-4] is query.pool_tail
        assert args[-3:] == (cp_start, 4, True)

    tail_buffers = {name: torch.empty(2, 7, dtype=torch.int64) for name in ("history", "writes", "state", "retained")}
    tail_buffers["history"] = torch.empty(2, 4, dtype=torch.int64)
    tail = adapters.build_tail_inputs(common, 4, 7, 80, retained_width, tail_buffers)
    assert launch.__getitem__.return_value.call_count == 2
    assert tail.page_rows == 80
    assert tail.history_indices.shape == (2, 4)
    assert tail.history_indices.data_ptr() == tail_buffers["history"].data_ptr()
    width = retained_width or 4
    for view, name in ((tail.write_indices, "writes"), (tail.state_indices, "state")):
        assert view.shape == (2, width)
        assert view.data_ptr() == tail_buffers[name].data_ptr()
        assert view.is_contiguous()
    if retained_width:
        assert tail.retained_indices.shape == (2, width)
        assert tail.retained_indices.data_ptr() == tail_buffers["retained"].data_ptr()
    else:
        assert tail.retained_indices is None
    args = launch.__getitem__.return_value.call_args.args
    assert args[0] is common.query_start_loc
    assert args[2] is common.positions
    assert args[26] == common.positions.stride(0)
    assert args[4] is common.block_table_tensor
    assert args[15] is tail.history_indices
    assert args[16] is tail.write_indices
    assert args[17] is tail.state_indices
    assert args[18] is tail.retained_indices
    assert args[30:35] == (4, 7, 80, width, True)


@pytest.mark.parametrize("use_output_buffer", [False, True])
def test_business_shape_index_packing_reuses_output_without_running_kernel(monkeypatch, use_output_buffer):
    launch = MagicMock()
    monkeypatch.setattr(adapters, "_pack_indices_kernel", launch)
    adapters.triton.next_power_of_2.return_value = 4096
    indices = torch.empty(3, 2051, dtype=torch.int32)
    storage = torch.empty(5, 2080, dtype=torch.int32) if use_output_buffer else None
    metadata = SimpleNamespace(
        positions=torch.empty(2, dtype=torch.int64), cum_query_lens=torch.empty(2, dtype=torch.int64)
    )
    result = adapters.pack_keypool_indices(indices, metadata, 2048, 4, storage)
    output = storage if use_output_buffer else indices
    assert result.shape == (3, 1, 2051)
    assert result.data_ptr() == output.data_ptr()
    assert result.stride(0) == output.stride(0)
    args = launch.__getitem__.return_value.call_args.args
    assert args[0] is indices
    assert args[1] is metadata.positions
    assert args[2] is metadata.cum_query_lens
    assert args[3].data_ptr() == output.data_ptr()
    assert args[4:11] == (2051, output.stride(0), 2048, 4, 2, output.shape[1], 4096)


def test_cache_writer_passes_padded_page_and_block_layout_without_running_kernel(monkeypatch):
    launch = MagicMock()
    monkeypatch.setattr(adapters, "_write_keypool_caches_kernel", launch)
    dim = 128
    tail = torch.empty(3, 80, 2 * dim)[:, :7]
    cache = torch.empty(3, 48, 1, dim)[:, :32]
    pooled = torch.empty(4, dim + 64)[:, :dim]
    writes = torch.tensor([[1, 81, 241]], dtype=torch.int64)
    state_rows = torch.empty_like(writes)
    slots = torch.tensor([0, 33, 96, -1], dtype=torch.int64)
    metadata = SimpleNamespace(pooled_slot_mapping=slots)
    tail_metadata = SimpleNamespace(
        cann_inputs=SimpleNamespace(write_indices=writes, state_indices=state_rows, page_rows=80)
    )
    state = torch.empty(5, 4, 2 * dim)
    adapters.write_keypool_caches(tail, cache, None, None, pooled, metadata, tail_metadata, state)
    args = launch.__getitem__.return_value.call_args.args
    assert launch.__getitem__.call_args.args == ((7,),)
    assert args[0] is tail
    assert args[1] is cache
    tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
    assert any(arg is slots for arg in tensors)
    assert any(arg is writes for arg in tensors)
    assert any(arg is state_rows for arg in tensors)
    scalars = [arg for arg in args if isinstance(arg, int)]
    assert scalars[:7] == [3, dim, 80, 3, 7, 3, 32]
    assert 48 * dim in scalars
    assert dim + 64 in scalars
