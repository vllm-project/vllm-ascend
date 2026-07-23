# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.eplb.eplb_communicator import EplbCommunicator

from vllm_ascend.distributed import eplb_communicator as communicator_module
from vllm_ascend.distributed.eplb_communicator import HcclEplbCommunicator


@pytest.fixture
def communicator(monkeypatch):
    monkeypatch.setattr(EplbCommunicator, "_log_initialized", lambda self: None)
    return HcclEplbCommunicator(MagicMock())


def test_execute_is_noop_without_transfers(communicator, monkeypatch):
    batch = MagicMock()
    monkeypatch.setattr(communicator_module, "batch_isend_irecv", batch)

    communicator.execute()

    batch.assert_not_called()


def test_execute_waits_for_all_transfers_and_clears_queue(communicator, monkeypatch):
    monkeypatch.setattr(
        communicator_module,
        "P2POp",
        lambda op, tensor, rank, group: (op, tensor, rank, group),
    )
    requests = [MagicMock(), MagicMock()]
    batch = MagicMock(return_value=requests)
    monkeypatch.setattr(communicator_module, "batch_isend_irecv", batch)
    tensors = [torch.ones(1), torch.zeros(1)]

    communicator.add_send(tensors, dst_rank=1, expert_id=3)
    communicator.execute()

    batch.assert_called_once()
    for request in requests:
        request.wait.assert_called_once_with()
    assert communicator._p2p_ops == []


def test_send_stages_nonzero_storage_offset(communicator, monkeypatch):
    monkeypatch.setattr(
        communicator_module,
        "P2POp",
        lambda op, tensor, rank, group: (op, tensor, rank, group),
    )
    tensor = torch.arange(6).view(3, 2)[1]

    communicator.add_send([tensor], dst_rank=1, expert_id=3)

    send_tensor = communicator._p2p_ops[0][1]
    assert tensor.storage_offset() != 0
    assert send_tensor.storage_offset() == 0
    assert send_tensor.data_ptr() != tensor.data_ptr()
    torch.testing.assert_close(send_tensor, tensor)


def test_recv_stages_nonzero_storage_offset_and_copies_back(
    communicator,
    monkeypatch,
):
    monkeypatch.setattr(
        communicator_module,
        "P2POp",
        lambda op, tensor, rank, group: (op, tensor, rank, group),
    )
    request = MagicMock()
    monkeypatch.setattr(
        communicator_module,
        "batch_isend_irecv",
        MagicMock(return_value=[request]),
    )
    tensor = torch.zeros(3, 2)[1]

    communicator.add_recv([tensor], src_rank=1, expert_id=3)
    recv_tensor = communicator._p2p_ops[0][1]
    recv_tensor.fill_(7)
    communicator.execute()

    assert recv_tensor.storage_offset() == 0
    torch.testing.assert_close(tensor, torch.full_like(tensor, 7))
    request.wait.assert_called_once_with()
    assert communicator._p2p_ops == []
    assert communicator._recv_staging == []


def test_execute_clears_queue_after_failure(communicator, monkeypatch):
    communicator._p2p_ops.append(object())
    communicator._recv_staging.append((torch.empty(1), torch.empty(1)))
    monkeypatch.setattr(
        communicator_module,
        "batch_isend_irecv",
        MagicMock(side_effect=RuntimeError("transfer failed")),
    )

    with pytest.raises(RuntimeError, match="transfer failed"):
        communicator.execute()

    assert communicator._p2p_ops == []
    assert communicator._recv_staging == []
