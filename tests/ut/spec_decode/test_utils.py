# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V1 speculative decoding sliding-window utilities."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.worker.v1.spec_decode.utils import SlidingWindowAdapter


def test_sliding_window_updates_host_length_mirrors():
    adapter = SlidingWindowAdapter(
        window_size=128,
        block_size=64,
        max_num_reqs=1,
        future_offset=0,
        device=torch.device("cpu"),
    )
    metadata = SimpleNamespace(
        block_table_tensor=torch.arange(10, dtype=torch.int32).view(1, 10),
        seq_lens=torch.tensor([300], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([300], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([300], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([300], dtype=torch.int32),
    )

    adapter.apply(metadata)

    expected = torch.tensor([172], dtype=torch.int32)
    torch.testing.assert_close(metadata.seq_lens, expected)
    torch.testing.assert_close(metadata.seq_lens_cpu, expected)
    torch.testing.assert_close(metadata._seq_lens_cpu, expected)
    torch.testing.assert_close(metadata.seq_lens_cpu_upper_bound, expected)


@pytest.mark.parametrize(
    "window,block,future,seq_len,start_block,expected_len",
    [
        pytest.param(8, 4, 0, 7, 0, 7, id="below-window"),
        pytest.param(8, 4, 0, 8, 0, 8, id="exact-window"),
        pytest.param(8, 4, 0, 11, 0, 11, id="partial-oldest-block"),
        pytest.param(8, 4, 0, 12, 1, 8, id="next-complete-block"),
        pytest.param(10, 4, 0, 14, 1, 10, id="unaligned-window"),
        pytest.param(8, 4, 3, 8, 0, 8, id="future-before-block-boundary"),
        pytest.param(8, 4, 3, 9, 1, 5, id="future-crosses-block-boundary"),
        pytest.param(10, 4, 3, 19, 3, 7, id="unaligned-window-with-future"),
    ],
)
def test_sliding_window_block_boundaries(window, block, future, seq_len, start_block, expected_len):
    adapter = SlidingWindowAdapter(window, block, 1, future, torch.device("cpu"))
    full_table = torch.arange(10, 22, dtype=torch.int32).view(1, -1)
    original_lengths = torch.tensor([seq_len], dtype=torch.int32)
    slots = torch.tensor([101, 102, 103], dtype=torch.int64)
    metadata = SimpleNamespace(
        block_table_tensor=full_table,
        seq_lens=original_lengths,
        seq_lens_cpu=original_lengths,
        _seq_lens_cpu=original_lengths,
        seq_lens_cpu_upper_bound=original_lengths,
        slot_mapping=slots,
    )

    adapter.apply(metadata)

    expected_table = torch.zeros_like(metadata.block_table_tensor)
    used_blocks = (expected_len + block - 1) // block
    expected_table[0, :used_blocks] = full_table[0, start_block : start_block + used_blocks]
    torch.testing.assert_close(metadata.block_table_tensor, expected_table)
    for name in ("seq_lens", "seq_lens_cpu", "_seq_lens_cpu", "seq_lens_cpu_upper_bound"):
        torch.testing.assert_close(getattr(metadata, name), torch.tensor([expected_len], dtype=torch.int32))
    assert adapter.full_block_table is full_table
    torch.testing.assert_close(full_table, torch.arange(10, 22, dtype=torch.int32).view(1, -1))
    torch.testing.assert_close(original_lengths, torch.tensor([seq_len], dtype=torch.int32))
    assert metadata.slot_mapping is slots
    torch.testing.assert_close(slots, torch.tensor([101, 102, 103], dtype=torch.int64))


def test_sliding_window_preserves_distinct_cpu_upper_bound_and_missing_mirrors():
    adapter = SlidingWindowAdapter(8, 4, 2, 0, torch.device("cpu"))
    metadata = SimpleNamespace(
        block_table_tensor=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32),
        seq_lens=torch.tensor([12, 7], dtype=torch.int32),
        seq_lens_cpu=None,
        _seq_lens_cpu=None,
        seq_lens_cpu_upper_bound=torch.tensor([15, 9], dtype=torch.int32),
    )

    adapter.apply(metadata)

    assert metadata.seq_lens_cpu is None
    assert metadata._seq_lens_cpu is None
    torch.testing.assert_close(metadata.seq_lens, torch.tensor([8, 7], dtype=torch.int32))
    torch.testing.assert_close(metadata.seq_lens_cpu_upper_bound, torch.tensor([11, 9], dtype=torch.int32))
    torch.testing.assert_close(metadata.block_table_tensor, torch.tensor([[11, 12, 0], [20, 21, 0]], dtype=torch.int32))


def test_sliding_window_reuse_clears_padding_and_respects_short_full_table():
    adapter = SlidingWindowAdapter(8, 4, 2, 0, torch.device("cpu"))
    storage = adapter._block_table_clone.data_ptr()
    first = SimpleNamespace(
        block_table_tensor=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32),
        seq_lens=torch.tensor([11, 11], dtype=torch.int32),
    )
    adapter.apply(first)
    torch.testing.assert_close(first.block_table_tensor, torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32))
    second = SimpleNamespace(
        block_table_tensor=torch.tensor([[90]], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
    )

    full_table = second.block_table_tensor
    adapter.apply(second)

    assert second.block_table_tensor.data_ptr() == storage
    assert second.block_table_tensor.shape == (1, 3)
    torch.testing.assert_close(second.block_table_tensor, torch.tensor([[90, 0, 0]], dtype=torch.int32))
    assert adapter.full_block_table is full_table
