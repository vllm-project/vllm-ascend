# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np
import pytest
import torch

import vllm_ascend.distributed.eplb.expert_copy as expert_copy
import vllm_ascend.distributed.eplb.explicit_transfer as explicit_transfer


class _FakeNpuTensor:
    def __init__(
        self,
        *,
        npu_format=50,
        storage_offset=0,
        shape=(2, 3),
        dtype=torch.float16,
    ):
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.device = torch.device("npu")
        self.nbytes = torch.empty(shape, dtype=dtype).nbytes
        self.npu_format = npu_format
        self._storage_offset = storage_offset
        self.copy_ = MagicMock(return_value=self)

    def storage_offset(self):
        return self._storage_offset


class _FakeCommunicator:
    receiver_initiated = False

    def __init__(self):
        self.sends = []
        self.recvs = []
        self.executed = False

    def set_transfer_context(self, old, layer_idx):
        self.context = old, layer_idx

    def add_send(self, tensors, rank, expert):
        self.sends.append((tensors, rank, expert))

    def add_recv(self, tensors, rank, expert):
        self.recvs.append((tensors, rank, expert))

    def execute(self):
        self.executed = True


@pytest.fixture
def npu_copy_mocks(monkeypatch):
    get_npu_format = MagicMock(side_effect=lambda tensor: tensor.npu_format)
    copy_memory = MagicMock(side_effect=lambda dst, src, non_blocking=False: dst)
    monkeypatch.setattr(
        expert_copy.torch_npu,
        "Format",
        SimpleNamespace(ND=2),
        raising=False,
    )
    monkeypatch.setattr(
        expert_copy.torch_npu,
        "get_npu_format",
        get_npu_format,
        raising=False,
    )
    monkeypatch.setattr(
        expert_copy.torch_npu,
        "copy_memory_",
        copy_memory,
        raising=False,
    )
    return get_npu_format, copy_memory


def test_copy_expert_tensor_uses_standard_copy_on_cpu():
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dst = torch.zeros_like(src)

    result = expert_copy.copy_expert_tensor_(dst, src, non_blocking=True)

    assert result is dst
    torch.testing.assert_close(dst, src)


def test_copy_expert_tensor_uses_copy_memory_for_internal_format(npu_copy_mocks):
    _, copy_memory = npu_copy_mocks
    src = _FakeNpuTensor(npu_format=50)
    dst = _FakeNpuTensor(npu_format=50)

    result = expert_copy.copy_expert_tensor_(dst, src, non_blocking=True)

    assert result is dst
    copy_memory.assert_called_once_with(dst, src, non_blocking=True)
    dst.copy_.assert_not_called()


def test_copy_expert_tensor_uses_standard_copy_for_nd(npu_copy_mocks):
    _, copy_memory = npu_copy_mocks
    src = _FakeNpuTensor(npu_format=2)
    dst = _FakeNpuTensor(npu_format=2)

    result = expert_copy.copy_expert_tensor_(dst, src, non_blocking=True)

    assert result is dst
    dst.copy_.assert_called_once_with(src, non_blocking=True)
    copy_memory.assert_not_called()


def test_copy_expert_tensor_rejects_mismatched_npu_formats(npu_copy_mocks):
    _, copy_memory = npu_copy_mocks
    src = _FakeNpuTensor(npu_format=50)
    dst = _FakeNpuTensor(npu_format=29)

    with pytest.raises(ValueError, match="matching NPU formats"):
        expert_copy.copy_expert_tensor_(dst, src)

    dst.copy_.assert_not_called()
    copy_memory.assert_not_called()


@pytest.mark.parametrize(
    ("dst_offset", "src_offset"),
    [
        pytest.param(1, 0, id="destination-offset"),
        pytest.param(0, 1, id="source-offset"),
    ],
)
def test_copy_expert_tensor_rejects_internal_format_offsets(
    npu_copy_mocks,
    dst_offset,
    src_offset,
):
    _, copy_memory = npu_copy_mocks
    src = _FakeNpuTensor(npu_format=50, storage_offset=src_offset)
    dst = _FakeNpuTensor(npu_format=50, storage_offset=dst_offset)

    with pytest.raises(ValueError, match="offset-0 tensors"):
        expert_copy.copy_expert_tensor_(dst, src)

    dst.copy_.assert_not_called()
    copy_memory.assert_not_called()


def test_move_from_buffer_commits_primary_and_duplicate_experts(monkeypatch):
    copy_tensor = MagicMock()
    monkeypatch.setattr(expert_copy, "copy_expert_tensor_", copy_tensor)
    expert_weights = [
        ["weight-0", "weight-1", "weight-2", "weight-3"],
        ["scale-0", "scale-1", "scale-2", "scale-3"],
    ]
    expert_buffers = [
        ["buffer-weight-0", "buffer-weight-1", "buffer-weight-2", "buffer-weight-3"],
        ["buffer-scale-0", "buffer-scale-1", "buffer-scale-2", "buffer-scale-3"],
    ]
    metadata = SimpleNamespace(
        is_unchanged=np.array([True, False, False, False], dtype=np.bool_),
        is_received_locally=np.array([False, True, False, False], dtype=np.bool_),
        recv_primary_mask=np.array([False, False, True, False], dtype=np.bool_),
        recv_count=1,
        recv_expert_ids=np.array([42, -1, -1, -1], dtype=np.int64),
        recv_dst_rows=np.array([2, -1, -1, -1], dtype=np.int32),
    )

    expert_copy.move_from_buffer(
        expert_weights=expert_weights,
        expert_weights_buffers=expert_buffers,
        transfer_metadata=metadata,
        new_indices=np.array([0, 7, 42, 42], dtype=np.int32),
        ep_rank=0,
    )

    assert copy_tensor.call_args_list == [
        call("weight-1", "buffer-weight-1", non_blocking=True),
        call("scale-1", "buffer-scale-1", non_blocking=True),
        call("weight-2", "buffer-weight-2", non_blocking=True),
        call("scale-2", "buffer-scale-2", non_blocking=True),
        call("weight-3", "weight-2", non_blocking=True),
        call("scale-3", "scale-2", non_blocking=True),
    ]


def test_explicit_local_staging_uses_expert_copy(monkeypatch):
    copy_tensor = MagicMock()
    monkeypatch.setattr(
        explicit_transfer,
        "copy_expert_tensor_",
        copy_tensor,
    )
    communicator = _FakeCommunicator()
    weights = [torch.tensor([[10.0], [11.0]])]
    buffers = [torch.zeros_like(weights[0])]

    metadata = explicit_transfer.stage_explicit_layer_transfer(
        torch.tensor([0, 1]),
        torch.tensor([1, 0]),
        np.array([[0, 0]]),
        np.array([[1, 0]]),
        weights,
        buffers,
        SimpleNamespace(size=lambda: 1, rank=lambda: 0),
        communicator,
    )

    assert copy_tensor.call_count == 2
    first, second = copy_tensor.call_args_list
    assert first.args[0].data_ptr() == buffers[0][0].data_ptr()
    assert first.args[1].data_ptr() == weights[0][1].data_ptr()
    assert first.kwargs == {"non_blocking": True}
    assert second.args[0].data_ptr() == buffers[0][1].data_ptr()
    assert second.args[1].data_ptr() == weights[0][0].data_ptr()
    assert second.kwargs == {"non_blocking": True}
    np.testing.assert_array_equal(metadata.is_received_locally, [True, True])
    np.testing.assert_array_equal(metadata.is_unchanged, [False, False])
    assert metadata.recv_count == 0
    assert communicator.executed
    assert not communicator.sends
    assert not communicator.recvs
