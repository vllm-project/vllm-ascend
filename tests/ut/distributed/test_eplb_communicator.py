# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from unittest.mock import MagicMock, call

import pytest
import torch
from vllm.distributed.eplb import eplb_communicator as upstream_communicator
from vllm.distributed.eplb.eplb_communicator import (
    EplbCommunicator,
    TorchDistNcclEplbCommunicator,
)

from vllm_ascend.distributed.eplb_communicator import HcclEplbCommunicator


@pytest.fixture
def communicator(monkeypatch):
    monkeypatch.setattr(EplbCommunicator, "_log_initialized", lambda self: None)
    return HcclEplbCommunicator(MagicMock())


def test_communicator_reuses_upstream_torch_distributed_transport(communicator):
    assert isinstance(communicator, TorchDistNcclEplbCommunicator)
    assert communicator.needs_profile_buffer_reservation is False


def test_send_and_recv_use_persistent_expert_tensors_directly(communicator, monkeypatch):
    communicator._ep_group.size.return_value = 2
    get_global_rank = MagicMock(side_effect=[3, 2])
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb_communicator.dist.get_global_rank",
        get_global_rank,
    )
    monkeypatch.setattr(
        upstream_communicator,
        "P2POp",
        lambda op, tensor, rank, group: (op, tensor, rank, group),
    )
    send_tensor = torch.arange(2)
    recv_tensor = torch.zeros(2)

    communicator.add_send([send_tensor], dst_rank=1, expert_id=3)
    communicator.add_recv([recv_tensor], src_rank=0, expert_id=3)

    assert communicator._p2p_ops[0][1] is send_tensor
    assert communicator._p2p_ops[1][1] is recv_tensor
    assert communicator._p2p_ops[0][2] == 3
    assert communicator._p2p_ops[1][2] == 2
    assert all(op[1].storage_offset() == 0 for op in communicator._p2p_ops)
    assert get_global_rank.call_args_list == [
        call(communicator._ep_group, 1),
        call(communicator._ep_group, 0),
    ]


def test_peer_group_rank_must_be_in_range(communicator):
    communicator._ep_group.size.return_value = 2

    with pytest.raises(ValueError, match=r"group rank 2.*\[0, 2\)"):
        communicator.add_send([torch.zeros(1)], dst_rank=2, expert_id=3)


def test_set_stream_is_ready_for_async_transfer(communicator):
    stream = object()

    communicator.set_stream(stream)

    assert communicator._cuda_stream is stream


def test_execute_clears_queue_after_failure(communicator, monkeypatch):
    communicator._p2p_ops.append(object())
    monkeypatch.setattr(
        upstream_communicator.torch.cuda,
        "stream",
        lambda stream: nullcontext(),
    )
    monkeypatch.setattr(
        upstream_communicator,
        "batch_isend_irecv",
        MagicMock(side_effect=RuntimeError("transfer failed")),
    )

    with pytest.raises(RuntimeError, match="transfer failed"):
        communicator.execute()

    assert communicator._p2p_ops == []
