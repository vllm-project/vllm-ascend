# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator


def communicator(rank=0, world_size=2):
    # Test the NPU method without repeating process-group initialization tests.
    comm = object.__new__(NPUCommunicator)
    comm.rank_in_group = rank
    comm.world_size = world_size
    comm.device_group = object()
    return comm


@pytest.mark.parametrize(
    ("sizes", "rank", "as_list"),
    [
        (None, 0, False),  # Implicit equal sizes.
        ([3, 3], 1, False),  # Explicit equal sizes.
        ([3, 2], 1, True),  # Unequal sizes and multiple dtypes.
        ([0, 2], 0, False),  # Empty local shard must still participate.
        ([0, 2], 1, False),  # Empty peer shard.
        ([0, 0], 0, True),  # All-empty group skips all collectives.
        ([2], 0, False),  # Single-rank group.
    ],
)
def test_all_gatherv_preserves_rank_order(sizes, rank, as_list):
    rows = sizes if sizes is not None else [3, 3]
    comm = communicator(rank, len(rows))
    expected = torch.arange(sum(rows) * 2, dtype=torch.int32).reshape(2, -1).T
    peers = expected.split(rows)
    inputs = [peers[rank], peers[rank].to(torch.int64) + 100] if as_list else peers[rank]
    calls = 0

    def all_gather(outputs, input_, group):
        nonlocal calls
        reference = expected if calls == 0 else expected.to(torch.int64) + 100
        assert group is comm.device_group
        assert input_.is_contiguous()
        torch.testing.assert_close(input_, reference.split(rows)[rank])
        assert [len(out) for out in outputs] == rows
        for output, peer in zip(outputs, reference.split(rows)):
            output.copy_(peer)
        calls += 1

    with patch.object(dist, "all_gather", side_effect=all_gather):
        result = comm.all_gatherv(inputs, sizes=sizes)

    if as_list:
        assert isinstance(result, list)
        torch.testing.assert_close(result[0], expected)
        torch.testing.assert_close(result[1], expected.to(torch.int64) + 100)
    else:
        torch.testing.assert_close(result, expected)
    assert calls == (0 if sum(rows) == 0 else (2 if as_list else 1))


@pytest.mark.parametrize(
    ("sizes", "message"),
    [([], "group size"), ([2, -1], "non-negative"), ([2, 3], "local rows")],
)
def test_all_gatherv_rejects_invalid_sizes_before_collective(sizes, message):
    with patch.object(dist, "all_gather") as gather:
        with pytest.raises(ValueError, match=message):
            communicator(rank=1).all_gatherv(torch.ones(2, 4), sizes=sizes)
        gather.assert_not_called()


def test_all_gatherv_rejects_unsupported_dim():
    with patch.object(dist, "all_gather") as gather:
        with pytest.raises(NotImplementedError, match="dim 0"):
            communicator().all_gatherv(torch.ones(2, 4), dim=1)
        gather.assert_not_called()


def test_all_gatherv_empty_tensor_list():
    with patch.object(dist, "all_gather") as gather:
        assert communicator().all_gatherv([], sizes=[2, 3]) == []
        gather.assert_not_called()
