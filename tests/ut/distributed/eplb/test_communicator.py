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


def test_send_and_recv_translate_group_local_peer_ranks(communicator, monkeypatch):
    communicator._cpu_group.size.return_value = 2
    get_global_rank = MagicMock(side_effect=[3, 2])
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.communicator.dist.get_global_rank",
        get_global_rank,
    )
    send_tensor = torch.arange(2)
    recv_tensor = torch.zeros(2)

    communicator.add_send([send_tensor], dst_rank=1, expert_id=3)
    communicator.add_recv([recv_tensor], src_rank=0, expert_id=3)

    assert communicator._ops == [
        ("send", send_tensor, 3),
        ("recv", recv_tensor, 2),
    ]
    assert get_global_rank.call_args_list == [
        call(communicator._cpu_group, 1),
        call(communicator._cpu_group, 0),
    ]


def test_peer_group_rank_must_be_in_range(communicator):
    communicator._cpu_group.size.return_value = 2

    with pytest.raises(ValueError, match=r"group rank 2.*\[0, 2\)"):
        communicator.add_send([torch.zeros(1)], dst_rank=2, expert_id=3)
