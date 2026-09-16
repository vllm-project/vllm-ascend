import weakref
from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase

from vllm_ascend.distributed.device_communicators.npu_communicator import (
    _AIV_OUT_OF_PLACE_COPY_LIMIT_BYTES,
    NPUCommunicator,
)


def _communicator(*, use_aiv: bool, reset_instances: bool = True) -> NPUCommunicator:
    if reset_instances:
        NPUCommunicator._instances = weakref.WeakSet()
    communicator = NPUCommunicator.__new__(NPUCommunicator)
    communicator._use_aiv = use_aiv
    communicator.device_group = MagicMock()
    communicator.device = 0
    communicator._pending_aiv_outputs = []
    communicator._last_aiv_work = None
    communicator._instances.add(communicator)
    return communicator


def test_small_aiv_all_reduce_uses_independent_output():
    communicator = _communicator(use_aiv=True)
    input_ = torch.ones(8, dtype=torch.float32)
    work = MagicMock()

    def reduce_in_place(output, *, group, async_op):
        assert group is communicator.device_group
        assert async_op
        output.mul_(2)
        return work

    with patch(
        "vllm_ascend.distributed.device_communicators.npu_communicator.dist.all_reduce",
        side_effect=reduce_in_place,
    ):
        output = communicator.all_reduce(input_)

    assert output.data_ptr() != input_.data_ptr()
    torch.testing.assert_close(input_, torch.ones_like(input_))
    torch.testing.assert_close(output, torch.full_like(output, 2))
    assert len(communicator._pending_aiv_outputs) == 1
    assert communicator._pending_aiv_outputs[0] is output
    assert communicator._last_aiv_work is work
    work.wait.assert_called_once_with()


def test_release_aiv_outputs_waits_for_all_ranks_before_reclaim():
    communicator = _communicator(use_aiv=True)
    output = torch.ones(8, dtype=torch.float32)
    work = MagicMock()
    communicator._pending_aiv_outputs.append(output)
    communicator._last_aiv_work = work

    def assert_output_is_live(*, group):
        assert group is communicator.device_group
        assert len(communicator._pending_aiv_outputs) == 1
        assert communicator._pending_aiv_outputs[0] is output

    with (
        patch("torch.npu.synchronize") as synchronize,
        patch(
            "vllm_ascend.distributed.device_communicators.npu_communicator.dist.barrier",
            side_effect=assert_output_is_live,
        ) as barrier,
    ):
        communicator.release_aiv_outputs()

    work.wait.assert_called_once_with()
    synchronize.assert_called_once_with(0)
    barrier.assert_called_once_with(group=communicator.device_group)
    assert communicator._pending_aiv_outputs == []
    assert communicator._last_aiv_work is None


def test_release_aiv_outputs_without_pending_work_is_noop():
    communicator = _communicator(use_aiv=True)

    with (
        patch("torch.npu.synchronize") as synchronize,
        patch("vllm_ascend.distributed.device_communicators.npu_communicator.dist.barrier") as barrier,
    ):
        communicator.release_aiv_outputs()

    synchronize.assert_not_called()
    barrier.assert_not_called()


def test_release_aiv_outputs_drains_other_communicator_instance():
    caller = _communicator(use_aiv=True)
    owner = _communicator(use_aiv=True, reset_instances=False)
    owner.device = 1
    output = torch.ones(8, dtype=torch.float32)
    work = MagicMock()
    owner._pending_aiv_outputs.append(output)
    owner._last_aiv_work = work

    with (
        patch("torch.npu.synchronize") as synchronize,
        patch("vllm_ascend.distributed.device_communicators.npu_communicator.dist.barrier") as barrier,
    ):
        caller.release_aiv_outputs()

    work.wait.assert_called_once_with()
    synchronize.assert_called_once_with(1)
    barrier.assert_called_once_with(group=caller.device_group)
    assert owner._pending_aiv_outputs == []
    assert owner._last_aiv_work is None


def test_large_aiv_all_reduce_keeps_default_path():
    communicator = _communicator(use_aiv=True)
    numel = _AIV_OUT_OF_PLACE_COPY_LIMIT_BYTES // torch.float32.itemsize + 1
    input_ = torch.ones(numel, dtype=torch.float32)
    reduced = torch.full_like(input_, 2)

    with patch.object(DeviceCommunicatorBase, "all_reduce", return_value=reduced) as default_all_reduce:
        output = communicator.all_reduce(input_)

    assert output is reduced
    default_all_reduce.assert_called_once_with(input_)


def test_non_aiv_all_reduce_keeps_default_path():
    communicator = _communicator(use_aiv=False)
    input_ = torch.ones(8, dtype=torch.float32)
    reduced = torch.full_like(input_, 2)

    with patch.object(DeviceCommunicatorBase, "all_reduce", return_value=reduced) as default_all_reduce:
        output = communicator.all_reduce(input_)

    assert output is reduced
    default_all_reduce.assert_called_once_with(input_)
