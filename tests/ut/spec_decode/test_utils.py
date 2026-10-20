# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V1 speculative decoding sliding-window utilities."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.worker.v1.spec_decode.utils import (
    SlidingWindowAdapter,
    copy_slot_mapping_to_padded_buffer,
)


def test_copy_slot_mapping_pads_reusable_buffer() -> None:
    buffer = torch.full((8,), 99, dtype=torch.int32)
    slot_mapping = torch.tensor([3, 7, 11, 12], dtype=torch.int64)

    result = copy_slot_mapping_to_padded_buffer(
        buffer,
        slot_mapping,
        copy_tokens=3,
        padding_value=-1,
    )

    assert result is buffer
    assert result.tolist() == [3, 7, 11, -1, -1, -1, -1, -1]
    assert result.dtype == torch.int32


def test_copy_slot_mapping_handles_overlapping_source_view() -> None:
    buffer = torch.tensor([3, 7, 11, 12, 99, 99], dtype=torch.int32)

    result = copy_slot_mapping_to_padded_buffer(
        buffer,
        buffer[:4],
        padding_value=-1,
    )

    assert result is buffer
    assert result.tolist() == [3, 7, 11, 12, -1, -1]


def test_copy_slot_mapping_rejects_invalid_lengths() -> None:
    buffer = torch.zeros(4, dtype=torch.int32)
    slot_mapping = torch.arange(3, dtype=torch.int32)

    with pytest.raises(ValueError, match="copy_tokens=5"):
        copy_slot_mapping_to_padded_buffer(
            buffer,
            slot_mapping,
            copy_tokens=5,
            padding_value=-1,
        )


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
