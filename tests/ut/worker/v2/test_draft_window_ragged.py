# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_dense import (
    DenseDraftReadBatch,
    prepare_dense_draft_read,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_read import DirectPrefixSource


@pytest.mark.parametrize("causal", [False, True])
def test_ragged_queries_keep_full_row_and_use_device_exact_at_boundary(causal):
    table = torch.arange(32, dtype=torch.int32).reshape(2, 16)
    source = DirectPrefixSource(table, (1, 0), (16, 16), 128)
    before = table.clone()
    offsets = torch.tensor([0, 3, 8], dtype=torch.int32)
    exact = torch.tensor([1151, 1408], dtype=torch.int32)
    held = []
    read = prepare_dense_draft_read(
        source,
        exact_device=exact,
        row_indices=torch.tensor([1, 0]),
        upper_cpu=torch.tensor([1152, 1411], dtype=torch.int32),
        window_size=1024,
        anchor_span=128,
        query_width=5,
        causal=causal,
        retain=lambda *items: held.extend(items),
        full_rows_cpu=torch.tensor([False, True]),
        query_start_loc_cpu=offsets,
        query_start_loc_device=offsets.clone(),
    )
    assert read.starts.tolist() == [0, 0]
    assert read.seq_lens.tolist() == [1151, 1408]
    assert read.capacity == 1440  # Full-row U controls allocation only.
    assert read.num_query_tokens == 8
    assert isinstance(read, DenseDraftReadBatch)
    assert [part.query_indices.tolist() for part in read.parts] == [[0, 1, 2], [3, 4, 5, 6, 7]]
    assert [part.read.capacity for part in read.parts] == [1152, 1440]
    for row, (state_row, end, width) in enumerate(((1, 1151, 3), (0, 1408, 5))):
        expected = [int(before[state_row, pos // 128]) * 128 + pos % 128 for pos in range(end)]
        private = read.parts[row].read
        assert private.slots[0, :end].tolist() == expected
        assert private.slots[0, end:].tolist() == [expected[-1]] * (private.capacity - end)
        for local in range(width):
            length = end - width + local + 1 if causal else end
            assert not private.mask[0, 0, local, :length].any()
            assert private.mask[0, 0, local, length:].all()
    assert torch.equal(table, before)
    offsets.zero_()
    exact.fill_(1)
    assert read.seq_lens.tolist() == [1151, 1408]
    assert read.parts[1].query_indices.tolist() == [3, 4, 5, 6, 7]
    assert any(item is read for item in held)


def test_ragged_eagle_native_padding_preserves_frozen_anchor_and_full_continuation():
    table = torch.arange(64, dtype=torch.int32).reshape(2, 32)
    source = DirectPrefixSource(table, (0, 1), (32, 32), 128)
    offsets = torch.tensor([0, 4, 261], dtype=torch.int32)
    read = prepare_dense_draft_read(
        source,
        exact_device=torch.tensor([1795, 2177], dtype=torch.int32),
        query_ends=torch.tensor([1796, 2177], dtype=torch.int32),
        row_indices=torch.tensor([0, 1]),
        upper_cpu=torch.tensor([1800, 2180], dtype=torch.int32),
        window_size=1035,
        anchor_span=128,
        query_width=257,
        causal=True,
        frozen_starts=torch.tensor([640, 1024], dtype=torch.int32),
        full_rows_cpu=torch.tensor([False, True]),
        query_start_loc_cpu=offsets,
        query_start_loc_device=offsets.clone(),
        retain=lambda *items: None,
    )
    assert read.starts.tolist() == [640, 0]
    assert read.seq_lens.tolist() == [1155, 2177]
    assert read.num_query_tokens == 261
    assert isinstance(read, DenseDraftReadBatch)
    assert (~read.parts[0].read.mask[0, 0]).sum(-1).tolist() == [1153, 1154, 1155, 1155]
    assert read.parts[0].query_indices.tolist() == [0, 1, 2, 3]
    assert read.parts[1].query_indices.tolist() == list(range(4, 261))
    assert [part.read.capacity for part in read.parts] == [1152 + 32, 2208]


def test_uniform_full_read_rows_keep_original_query_layout():
    source = DirectPrefixSource(torch.arange(24, dtype=torch.int32).reshape(2, 12), (0, 1), (12, 12), 128)
    read = prepare_dense_draft_read(
        source,
        exact_device=torch.tensor([1151, 1287], dtype=torch.int32),
        row_indices=torch.tensor([0, 1]),
        upper_cpu=torch.tensor([1152, 1290], dtype=torch.int32),
        window_size=1024,
        anchor_span=128,
        query_width=8,
        causal=False,
        full_rows_cpu=torch.tensor([False, True]),
        retain=lambda *items: None,
    )
    assert read.starts.tolist() == [0, 0]
    assert read.capacity == 1312
    assert read.num_query_tokens == 16
    assert isinstance(read, DenseDraftReadBatch)
    assert [part.query_indices.tolist() for part in read.parts] == [list(range(8)), list(range(8, 16))]
    assert all(part.read.query_indices is None and part.read.output_indices is None for part in read.parts)
