# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.attention.context_parallel.sfa_dcp_utils import (
    _index_select_kv_blocks,
    as_byte_view,
    build_sfa_dcp_compact_kv_metadata,
)


def test_c4_scale_pages_are_selected_and_transported_as_bytes(monkeypatch):
    raw = torch.arange(512, dtype=torch.int32).to(torch.uint8).reshape(4, 32, 1, 2, 2)
    scales = raw.view(torch.float8_e8m0fnu)
    select = torch.index_select

    def select_bytes(cache, dim, ids):
        assert cache.dtype == torch.uint8
        return select(cache, dim, ids)

    monkeypatch.setattr(torch, "index_select", select_bytes)
    pages = _index_select_kv_blocks(scales, torch.tensor([3, 0, 3]))
    wire = as_byte_view(pages)
    assert pages.dtype == scales.dtype and wire.dtype == torch.uint8
    torch.testing.assert_close(wire, raw[[3, 0, 3]])


@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("interleave", [128, 256, 768])
@pytest.mark.parametrize("first_block", [0, 5])
def test_compact_history_bounds_stale_tails_without_changing_valid_addresses(world, interleave, first_block):
    # Missing IDs include a negative placeholder, an insertion point between
    # allocated blocks and a stale ID beyond the largest allocated block.
    global_table = torch.tensor([[first_block, 9, -1, 0, 6, 999]], dtype=torch.int32)
    local_table = global_table.expand(2, -1).clone()
    original = local_table.clone()
    order = [0, 1] if world == 2 else [0, 4, 1, 5, 2, 6, 3, 7]
    ids, table = build_sfa_dcp_compact_kv_metadata(
        local_table,
        torch.arange(local_table.shape[1] * world, dtype=torch.int32),
        torch.tensor(order, dtype=torch.int32),
        128,
        interleave,
        global_dcp_block_table=global_table,
        global_dcp_num_blocks=torch.tensor([2], dtype=torch.int32),
    )
    assert ids.tolist() == [first_block, 9]
    assert table.dtype == torch.int32
    assert table.shape == (2, 6 * world)
    assert torch.all((table >= 0) & (table < world * len(ids)))
    torch.testing.assert_close(local_table, original)

    # Decode actual gather addresses against an independent token-owner oracle.
    # Reading every column also exercises the bounds of padded columns.
    payload = torch.arange(world * len(ids), dtype=torch.int64)
    selected = payload[table.long()]
    for global_column in range(table.shape[1]):
        position = global_column * 128
        owner = (position // interleave) % world
        local_position = position // (world * interleave) * interleave + position % interleave
        block_id = local_table[0, local_position // 128].item()
        expected = order[owner] * len(ids) + ids.tolist().index(block_id) if block_id in ids.tolist() else 0
        assert selected[:, global_column].tolist() == [expected, expected]


@pytest.mark.parametrize("rows", [0, 1, 2])
@pytest.mark.parametrize("world", [2, 8])
def test_compact_history_with_no_allocated_blocks_produces_padding_only(rows, world):
    global_table = torch.tensor([[999, -1], [888, 777]], dtype=torch.int32)
    ids, table = build_sfa_dcp_compact_kv_metadata(
        global_table[:rows],
        torch.arange(2 * world, dtype=torch.int32),
        torch.arange(world, dtype=torch.int32),
        128,
        128,
        global_dcp_block_table=global_table,
        global_dcp_num_blocks=torch.zeros(2, dtype=torch.int32),
    )
    assert ids.numel() == 0
    assert table.shape == (rows, 2 * world)
    assert table.dtype == torch.int32
    assert torch.count_nonzero(table) == 0


@pytest.mark.parametrize("world", [2, 8])
def test_empty_receiver_keeps_globally_allocated_history(world):
    ids, table = build_sfa_dcp_compact_kv_metadata(
        torch.empty((0, 2), dtype=torch.int32),
        torch.arange(2 * world, dtype=torch.int32),
        torch.arange(world, dtype=torch.int32),
        128,
        128,
        global_dcp_block_table=torch.tensor([[7, 999]], dtype=torch.int32),
        global_dcp_num_blocks=torch.tensor([1], dtype=torch.int32),
    )
    assert ids.tolist() == [7]
    assert table.shape == (0, 2 * world)
