# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase

from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator
from vllm_ascend.models.common.ops import sequence_parallel as sp


@pytest.mark.parametrize("custom_collective", [False, True])
@pytest.mark.parametrize("num_tokens,tp_size", [(8, 4), (10, 4), (1, 4), (3, 1), (0, 4)])
def test_reduce_scatter_padding(monkeypatch, custom_collective, num_tokens, tp_size):
    x = torch.arange(num_tokens * 8, dtype=torch.float32).reshape(num_tokens, 8)
    padding = (-num_tokens) % tp_size
    expected = torch.nn.functional.pad(x, (0, 0, 0, padding))
    inputs = []

    def reduce_scatter(value):
        inputs.append(value)
        torch.testing.assert_close(value, expected)
        if padding == 0:
            # The aligned path must preserve the input buffer, avoiding Cat.
            assert value is x
        return value[: value.shape[0] // tp_size]

    custom = Mock(side_effect=reduce_scatter) if custom_collective else Mock(return_value=None)
    fallback = Mock(side_effect=lambda value, dim: reduce_scatter(value))
    communicator = SimpleNamespace(custom_reduce_scatter=custom)
    monkeypatch.setattr(sp, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(sp, "get_tp_group", lambda: SimpleNamespace(device_communicator=communicator))
    monkeypatch.setattr(sp, "tensor_model_parallel_reduce_scatter", fallback)

    # This CPU-only test exercises the custom-op implementation directly;
    # the public wrapper intentionally registers only the PrivateUse1 kernel.
    result = sp._ascend_sp_reduce_scatter_impl(x)

    assert len(inputs) == 1
    torch.testing.assert_close(result, expected[: expected.shape[0] // tp_size])
    custom.assert_called_once()
    if custom_collective:
        fallback.assert_not_called()
    else:
        fallback.assert_called_once_with(inputs[0], 0)


def test_chunked_sequence_collectives_preserve_rank_and_row_order(monkeypatch):
    comm = object.__new__(NPUCommunicator)
    comm.world_size = 2
    comm.max_sequence_collective_peer_bytes = 16
    input_rows = torch.arange(15, dtype=torch.float32).view(5, 3)
    peer_rows = input_rows + 100
    full_rows = torch.cat((input_rows, peer_rows), dim=0)

    def gather(self, chunk, dim):
        assert dim == 0 and chunk.numel() * chunk.element_size() <= comm.max_sequence_collective_peer_bytes
        return torch.cat((chunk, chunk + 100))

    def reduce(self, chunks, dim):
        rank_chunks = chunks.view(self.world_size, -1)
        assert dim == 0 and rank_chunks[0].numel() * chunks.element_size() <= comm.max_sequence_collective_peer_bytes
        return rank_chunks[1]

    monkeypatch.setattr(DeviceCommunicatorBase, "all_gather", gather)
    monkeypatch.setattr(DeviceCommunicatorBase, "reduce_scatter", reduce)
    torch.testing.assert_close(comm.all_gather(input_rows, dim=0), full_rows)
    torch.testing.assert_close(comm.reduce_scatter(full_rows, dim=0), peer_rows)
