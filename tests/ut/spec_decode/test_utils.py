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


def test_sliding_window_align_must_be_a_multiple_of_block_size():
    with pytest.raises(ValueError, match="positive multiple"):
        SlidingWindowAdapter(
            window_size=128,
            block_size=64,
            max_num_reqs=1,
            future_offset=0,
            device=torch.device("cpu"),
            align=96,
        )
