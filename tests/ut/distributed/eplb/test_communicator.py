# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import MagicMock, call

import pytest
import torch
import torch_npu
from vllm.distributed.eplb.eplb_communicator import (
    EplbCommunicator,
    TorchDistGlooStagedEplbCommunicator,
)

from vllm_ascend.distributed.eplb.eplb_communicator import AscendGlooEplbCommunicator
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ


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
    monkeypatch.setattr("vllm_ascend.distributed.eplb.eplb_communicator.P2POp", p2p_op)
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.eplb_communicator.batch_isend_irecv",
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
        "vllm_ascend.distributed.eplb.eplb_communicator.P2POp",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.eplb_communicator.batch_isend_irecv",
        lambda _ops: [],
    )
    current_stream = MagicMock()
    monkeypatch.setattr(torch.accelerator, "current_stream", lambda: current_stream)

    communicator.execute()

    current_stream.synchronize.assert_called_once_with()


@pytest.mark.parametrize(
    "device_type,npu_format",
    [("cpu", ACL_FORMAT_FRACTAL_ND), ("npu", ACL_FORMAT_FRACTAL_ND), ("npu", ACL_FORMAT_FRACTAL_NZ)],
)
def test_execute_converts_nz_receives_on_device(communicator, monkeypatch, device_type, npu_format):
    recv_tensor = MagicMock()
    recv_tensor.device.type = device_type
    staging_tensor = MagicMock()
    nz_tensor = MagicMock()
    communicator._ops.append(("recv", recv_tensor, 0))
    monkeypatch.setattr(communicator, "_acquire_staging_buffer", lambda *_args: staging_tensor)
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.eplb_communicator.P2POp",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.eplb_communicator.batch_isend_irecv",
        lambda _ops: [],
    )
    monkeypatch.setattr(torch.accelerator, "current_stream", MagicMock())
    get_format = MagicMock(return_value=npu_format)
    format_cast = MagicMock(return_value=nz_tensor)
    monkeypatch.setattr(torch_npu, "get_npu_format", get_format, raising=False)
    monkeypatch.setattr(torch_npu, "npu_format_cast", format_cast)

    communicator.execute()

    if device_type == "npu" and npu_format == ACL_FORMAT_FRACTAL_NZ:
        staging_tensor.to.assert_called_once_with(recv_tensor.device, non_blocking=True)
        format_cast.assert_called_once_with(staging_tensor.to.return_value, ACL_FORMAT_FRACTAL_NZ)
        recv_tensor.copy_.assert_called_once_with(nz_tensor, non_blocking=True)
    else:
        staging_tensor.to.assert_not_called()
        format_cast.assert_not_called()
        recv_tensor.copy_.assert_called_once_with(staging_tensor, non_blocking=True)
    if device_type == "cpu":
        get_format.assert_not_called()
