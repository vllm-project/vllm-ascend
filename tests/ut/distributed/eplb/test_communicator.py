# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import MagicMock, call

import pytest
import torch
from vllm.distributed.eplb.eplb_communicator import (
    EplbCommunicator,
    TorchDistGlooStagedEplbCommunicator,
)

from vllm_ascend.distributed.eplb.communicator import AscendGlooEplbCommunicator


@pytest.fixture
def communicator(monkeypatch):
    monkeypatch.setattr(EplbCommunicator, "_log_initialized", lambda self: None)
    return AscendGlooEplbCommunicator(cpu_group=MagicMock())


def test_communicator_reuses_upstream_gloo_staging(communicator):
    assert isinstance(communicator, TorchDistGlooStagedEplbCommunicator)
    assert communicator.needs_profile_buffer_reservation is False


def test_execute_uses_group_local_peer_ranks(communicator, monkeypatch):
    send_tensor = torch.arange(2)
    recv_tensor = torch.zeros(2)
    communicator.add_send([send_tensor], dst_rank=1, expert_id=3)
    communicator.add_recv([recv_tensor], src_rank=0, expert_id=3)
    staging_tensors = [torch.empty_like(send_tensor), torch.empty_like(recv_tensor)]
    monkeypatch.setattr(
        communicator,
        "_acquire_staging_buffer",
        MagicMock(side_effect=staging_tensors),
    )
    p2p_op = MagicMock(side_effect=lambda *args, **kwargs: (args, kwargs))
    monkeypatch.setattr("vllm_ascend.distributed.eplb.communicator.P2POp", p2p_op)
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.communicator.batch_isend_irecv",
        lambda _ops: [],
    )
    current_stream = MagicMock()
    monkeypatch.setattr(torch.accelerator, "current_stream", lambda: current_stream)

    communicator.execute()

    assert p2p_op.call_args_list == [
        call(torch.distributed.isend, staging_tensors[0], group=communicator._cpu_group, group_peer=1),
        call(torch.distributed.irecv, staging_tensors[1], group=communicator._cpu_group, group_peer=0),
    ]


def test_pinned_staging_buffers_are_reused_between_transfers(communicator, monkeypatch):
    allocated = [object(), object()]
    empty_like = MagicMock(side_effect=allocated)
    monkeypatch.setattr(torch, "empty_like", empty_like)
    tensor = torch.zeros((2, 3), dtype=torch.float32)

    first_transfer: dict[tuple[torch.dtype, tuple[int, ...]], int] = {}
    assert communicator._acquire_staging_buffer(tensor, first_transfer) is allocated[0]
    assert communicator._acquire_staging_buffer(tensor, first_transfer) is allocated[1]

    second_transfer: dict[tuple[torch.dtype, tuple[int, ...]], int] = {}
    assert communicator._acquire_staging_buffer(tensor, second_transfer) is allocated[0]
    assert empty_like.call_count == 2
    empty_like.assert_called_with(tensor, device="cpu", pin_memory=True)


def test_execute_uses_current_device_stream_when_stream_is_unset(communicator, monkeypatch):
    tensor = torch.zeros(1)
    communicator._ops.append(("send", tensor, 0))
    monkeypatch.setattr(communicator, "_acquire_staging_buffer", lambda *_args: tensor)
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.communicator.P2POp",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.communicator.batch_isend_irecv",
        lambda _ops: [],
    )
    current_stream = MagicMock()
    monkeypatch.setattr(torch.accelerator, "current_stream", lambda: current_stream)

    communicator.execute()

    current_stream.synchronize.assert_called_once_with()
