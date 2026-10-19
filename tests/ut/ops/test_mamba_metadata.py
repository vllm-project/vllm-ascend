# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ops.triton.mamba import metadata as mamba_metadata
from vllm_ascend.ops.triton.mamba.metadata import aligned_mamba_block_table


@pytest.mark.parametrize("num_requests", [0, 1, 7])
@pytest.mark.parametrize("num_speculative_blocks", [0, 1, 3])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_aligned_mamba_block_table(num_requests, num_speculative_blocks, strided, dtype):
    table = torch.arange(num_requests * 32, dtype=dtype).reshape(num_requests, 32)
    lengths = torch.tensor([0, 1, 16, 17, 32, 33, 48], dtype=dtype)[:num_requests]
    if strided:
        table = table[:, ::2]
        lengths = torch.stack((lengths, lengths), dim=1)[:, 0]
    expected = (
        torch.stack(
            [
                table[row, max((length - 1) // 16, 0) : max((length - 1) // 16, 0) + num_speculative_blocks + 1]
                for row, length in enumerate(lengths.tolist())
            ]
        )
        if num_requests
        else table.new_empty((0, num_speculative_blocks + 1))
    )

    actual = aligned_mamba_block_table(table, lengths, 16, num_speculative_blocks)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == table.dtype


@pytest.mark.parametrize(
    "table,lengths,block_size,num_speculative_blocks",
    [
        (torch.empty(2, 4), torch.empty(1), 16, 0),
        (torch.empty(4), torch.empty(4), 16, 0),
        (torch.empty(2, 4), torch.empty(2, 1), 16, 0),
        (torch.empty(2, 4), torch.empty(2), 0, 0),
        (torch.empty(2, 4), torch.empty(2), 16, -1),
        (torch.empty(2, 1), torch.empty(2), 16, 1),
    ],
)
def test_aligned_mamba_block_table_rejects_invalid_metadata(table, lengths, block_size, num_speculative_blocks):
    with pytest.raises(ValueError):
        aligned_mamba_block_table(table, lengths, block_size, num_speculative_blocks)


def test_aligned_mamba_block_table_rejects_cross_device_metadata():
    with pytest.raises(ValueError, match="same device"):
        aligned_mamba_block_table(torch.empty(2, 4), torch.empty(2, device="meta"), 16, 0)


@pytest.mark.parametrize("num_requests", [0, 1, 7])
def test_aligned_mamba_block_table_uses_single_device_launch(monkeypatch, num_requests):
    launch = MagicMock()
    kernel = MagicMock()
    kernel.__getitem__.return_value = launch
    monkeypatch.setattr(mamba_metadata, "_aligned_mamba_block_table_kernel", kernel)
    table = torch.empty((num_requests, 32), dtype=torch.int32, device="meta")[:, ::2]
    lengths = torch.empty((num_requests, 2), dtype=torch.int32, device="meta")[:, 0]

    actual = aligned_mamba_block_table(table, lengths, 16, 3)

    assert actual.shape == (num_requests, 4)
    if num_requests:
        kernel.__getitem__.assert_called_once_with((num_requests,))
        launch.assert_called_once()
        assert launch.call_args.args[3:6] == (32, 2, 2)
    else:
        kernel.__getitem__.assert_not_called()
