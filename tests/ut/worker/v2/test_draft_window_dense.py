# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_dense import gather_dense_draft_kv, prepare_dense_draft_read
from vllm_ascend.worker.v2.spec_decode.draft_window_read import DirectPrefixSource


def test_frozen_eagle_anchor_does_not_follow_later_exact_lengths():
    source = DirectPrefixSource(torch.tensor([[3, 1, 2, 4]], dtype=torch.int32), (0,), (4,), 4)
    frozen = torch.tensor([4], dtype=torch.int32)  # E0=10,K3,W8,B4 -> H13,S4.
    for end in (11, 12, 13):
        read = prepare_dense_draft_read(
            source,
            exact_device=torch.tensor([end], dtype=torch.int32),
            row_indices=torch.tensor([0]),
            upper_cpu=torch.tensor([14]),
            window_size=8,
            anchor_span=4,
            query_width=1,
            causal=True,
            retain=lambda *objects: None,
            frozen_starts=frozen,
        )
        assert read.starts.tolist() == [4]
        assert read.seq_lens.tolist() == [end - 4]
        expected = [int(source.table[0, pos // 4]) * 4 + pos % 4 for pos in range(4, end)]
        assert read.slots[0, : end - 4].tolist() == expected
        assert not read.mask[0, 0, 0, : end - 4].any()


def test_rejected_first_pass_keeps_native_query_alignment_and_excludes_unwritten_tail():
    end, native_end, block = 1159, 1160, 128
    source = DirectPrefixSource(torch.arange(12, dtype=torch.int32).reshape(1, 12), (0,), (12,), block)
    held = []
    read = prepare_dense_draft_read(
        source,
        exact_device=torch.tensor([end], dtype=torch.int32),
        query_ends=torch.tensor([native_end], dtype=torch.int32),
        row_indices=torch.tensor([0]),
        upper_cpu=torch.tensor([native_end + 1]),
        window_size=1035,
        anchor_span=block,
        query_width=4,
        causal=True,
        frozen_starts=torch.tensor([0], dtype=torch.int32),
        retain=lambda *items: held.extend(items),
    )
    assert read.seq_lens.tolist() == [end]
    assert read.slots[0, :end].tolist() == list(range(end))
    assert read.slots[0, end:].tolist() == [end - 1] * (read.capacity - end)
    assert (~read.mask[0, 0]).sum(-1).tolist() == [1157, 1158, 1159, 1159]
    cache = torch.full((12, block, 1, 16), torch.nan, dtype=torch.bfloat16)
    cache.view(-1, 1, 16)[:end] = 1
    keys, values = gather_dense_draft_kv(read, cache, cache)
    assert torch.isfinite(keys).all() and torch.isfinite(values).all()


@pytest.mark.parametrize("page_layers", [0, 3])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("query_width", [8, 9])
@pytest.mark.parametrize("output_dtype", [None, torch.float32])
def test_device_exact_read_set_reorder_rejection_and_nan_tail(causal, query_width, page_layers, output_dtype):
    block = 128
    table = torch.tensor(
        [[3, 1, 8, 0, 11, 4, 2, 10, 9, 5, 6, 7], [16, 12, 20, 15, 13, 22, 14, 21, 23, 18, 17, 19]], dtype=torch.int32
    )
    ends = torch.tensor([1151, 1152], dtype=torch.int32)
    upper = torch.tensor([1152, 1152], dtype=torch.int32)
    held = []
    source = DirectPrefixSource(table, (1, 0), (12, 12), block)
    read = prepare_dense_draft_read(
        source,
        exact_device=ends,
        row_indices=torch.tensor([1, 0]),
        upper_cpu=upper,
        window_size=1024,
        anchor_span=block,
        query_width=query_width,
        causal=causal,
        retain=lambda *objects: held.extend(objects),
    )
    assert read.starts.tolist() == [0, 128]  # U1152 must not move row0's start.
    assert read.seq_lens.tolist() == [1151, 1024]
    cache = torch.full((24, block, 2, 16), torch.nan, dtype=torch.bfloat16)
    backing = None
    if page_layers:
        backing = torch.full((24, page_layers, 2, block, 2, 16), torch.nan, dtype=torch.bfloat16)
        cache = backing[:, 2, 1]  # Nonzero storage offset and gaps between pages.
        assert not cache.is_contiguous()
    reference = []
    for row, (physical_row, end) in enumerate(zip((1, 0), ends.tolist())):
        start = max(end - 1024, 0) // block * block
        expected = []
        for logical in range(end):
            block_id = int(table[physical_row, logical // block])
            cache[block_id, logical % block] = logical + 10000 * row
            if logical >= start:
                expected.append(block_id * block + logical % block)
        actual = read.slots[row].tolist()
        assert actual[: len(expected)] == expected
        assert actual[len(expected) :] == [expected[-1]] * (read.capacity - len(expected))
        assert set(actual) == set(expected)
        reference.append(cache.flatten(0, 1)[expected].clone())
    backing_before = backing.clone() if backing is not None else None
    keys, values = gather_dense_draft_kv(read, cache, cache, output_dtype=output_dtype)
    assert keys.dtype == values.dtype == (output_dtype or cache.dtype)
    assert torch.isfinite(keys).all() and torch.isfinite(values).all()
    for row, expected_values in enumerate(reference):
        torch.testing.assert_close(keys[row, : len(expected_values)], expected_values.to(keys.dtype), rtol=0, atol=0)
        visible = len(expected_values)
        for q in range(query_width):
            allowed = min(visible, visible - query_width + q + 1) if causal else visible
            assert not read.mask[row, 0, q, :allowed].any()
            assert read.mask[row, 0, q, allowed:].all()
    if backing is not None:
        torch.testing.assert_close(backing, backing_before, atol=0, rtol=0, equal_nan=True)
    ends.fill_(1)
    table.fill_(0)
    assert read.seq_lens.tolist() == [1151, 1024]
    assert any(value is keys for value in held)


def test_apply_keeps_full_write_and_absolute_query_operands():
    source = DirectPrefixSource(torch.tensor([[2, 1]], dtype=torch.int32), (0,), (2,), 128)
    read = prepare_dense_draft_read(
        source,
        exact_device=torch.tensor([129], dtype=torch.int32),
        row_indices=torch.tensor([0]),
        upper_cpu=torch.tensor([130]),
        window_size=128,
        anchor_span=128,
        query_width=9,
        causal=False,
        retain=lambda *objects: None,
    )
    full = source.table
    slots, positions = object(), object()
    common = SimpleNamespace(num_reqs=2, block_table_tensor=full, slot_mapping=slots, positions=positions)
    read.apply(common)
    assert common.block_table_tensor is full and common.slot_mapping is slots and common.positions is positions
    assert common.seq_lens.tolist() == [129, 0]
    assert common.seq_lens_cpu is None and common._seq_lens_cpu is None
    assert read.capacity_cpu(common).tolist() == [160, 160]
    with pytest.raises(ValueError, match="another common"):
        read.capacity_cpu(SimpleNamespace(num_reqs=2))


def test_allocation_upper_cannot_authorize_uninitialized_columns():
    source = DirectPrefixSource(torch.tensor([[1]], dtype=torch.int32), (0,), (1,), 128)
    with pytest.raises(ValueError, match="Allocation upper"):
        prepare_dense_draft_read(
            source,
            exact_device=torch.tensor([127]),
            row_indices=torch.tensor([0]),
            upper_cpu=torch.tensor([129]),
            window_size=128,
            anchor_span=128,
            query_width=8,
            causal=False,
            retain=lambda *objects: None,
        )


def test_cpu_capacity_fields_ignore_default_tensor_device():
    source = DirectPrefixSource(torch.tensor([[2, 1]], dtype=torch.int32), (0,), (2,), 128)
    exact, upper, rows = torch.tensor([129]), torch.tensor([130]), torch.tensor([0])
    with torch.device("meta"):
        read = prepare_dense_draft_read(
            source,
            exact_device=exact,
            row_indices=rows,
            upper_cpu=upper,
            window_size=128,
            anchor_span=128,
            query_width=8,
            causal=False,
            retain=lambda *objects: None,
        )
        common = SimpleNamespace(num_reqs=1)
        read.apply(common)
        assert common.seq_lens_cpu_upper_bound.device.type == "cpu"
        assert read.capacity_cpu(common).device.type == "cpu"


def test_short_context_large_window_does_not_gather_window_sized_padding():
    source = DirectPrefixSource(torch.tensor([[1]], dtype=torch.int32), (0,), (1,), 128)
    read = prepare_dense_draft_read(
        source,
        exact_device=torch.tensor([31], dtype=torch.int32),
        row_indices=torch.tensor([0]),
        upper_cpu=torch.tensor([34]),
        window_size=4096,
        anchor_span=128,
        query_width=8,
        causal=False,
        retain=lambda *objects: None,
    )
    assert read.capacity == 64
    assert read.starts.tolist() == [0] and read.seq_lens.tolist() == [31]
    assert read.slots[0, :31].tolist() == list(range(128, 159))
    assert read.slots[0, 31:].tolist() == [158] * 33
    assert not read.mask[..., :31].any() and read.mask[..., 31:].all()
