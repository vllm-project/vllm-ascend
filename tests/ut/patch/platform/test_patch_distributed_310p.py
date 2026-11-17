# SPDX-License-Identifier: Apache-2.0
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def communication():
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_distributed.py"
    spec = spec_from_file_location("test_310p_communication", path)
    module = module_from_spec(spec)
    with (
        patch(
            "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
            return_value=SimpleNamespace(supports=lambda cap: False),
        ),
        patch.object(torch.distributed, "broadcast", autospec=True) as original,
        patch.object(torch.distributed.distributed_c10d, "broadcast"),
        patch.object(torch.distributed, "all_reduce", autospec=True) as original_reduce,
        patch.object(torch.distributed.distributed_c10d, "all_reduce"),
    ):
        spec.loader.exec_module(module)
        module.communication_adaptation_310p()
        yield SimpleNamespace(
            broadcast=torch.distributed.broadcast,
            original_broadcast=original,
            all_reduce=torch.distributed.all_reduce,
            original_all_reduce=original_reduce,
        )


@pytest.fixture
def broadcast(communication):
    return communication.broadcast, communication.original_broadcast


@pytest.mark.parametrize("sources", [{"src": 2}, {"group_src": 1}])
@pytest.mark.parametrize("async_op", [False, True])
def test_cpu_broadcast_preserves_root_and_async_handle(broadcast, sources, async_op):
    fn, original = broadcast
    tensor = torch.ones(1)
    group = object()
    original.return_value = MagicMock() if async_op else None
    assert fn(tensor, group=group, async_op=async_op, **sources) is original.return_value
    original.assert_called_once_with(
        tensor, src=sources.get("src"), group=group, async_op=async_op, group_src=sources.get("group_src")
    )


@pytest.mark.parametrize("sources", [{}, {"src": 2, "group_src": 1}])
def test_cpu_broadcast_preserves_upstream_validation_errors(broadcast, sources):
    fn, original = broadcast
    tensor = torch.ones(1)
    group = object()
    error = ValueError("invalid broadcast root")
    original.side_effect = error
    with pytest.raises(ValueError) as caught:
        fn(tensor, group=group, **sources)
    assert caught.value is error
    original.assert_called_once_with(
        tensor, src=sources.get("src"), group=group, async_op=False, group_src=sources.get("group_src")
    )


@pytest.mark.parametrize("sources", [{}, {"src": 2, "group_src": 0}])
def test_device_broadcast_rejects_missing_or_ambiguous_root(broadcast, sources):
    fn, _ = broadcast
    tensor = MagicMock()
    tensor.device.type = "npu"
    with pytest.raises(ValueError, match="exactly one"):
        fn(tensor, **sources)


@pytest.mark.parametrize("sources", [{"src": 2}, {"group_src": 0}])
def test_device_broadcast_maps_subgroup_root(broadcast, sources):
    fn, _ = broadcast
    tensor = MagicMock()
    tensor.device.type = "npu"
    group = object()
    root_buffer = MagicMock()
    other_buffer = MagicMock()
    with (
        patch.object(
            torch.distributed.distributed_c10d, "get_group_rank", side_effect=lambda g, rank: [2, 3].index(rank)
        ),
        patch.object(torch.distributed.distributed_c10d, "get_global_rank", side_effect=lambda g, rank: [2, 3][rank]),
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch.object(torch, "empty_like", side_effect=[root_buffer, other_buffer]),
        patch.object(torch.distributed, "all_gather") as gather,
    ):
        fn(tensor, group=group, **sources)
    gather.assert_called_once_with([root_buffer, other_buffer], tensor, group=group)
    tensor.__setitem__.assert_called_once_with(Ellipsis, root_buffer)


def test_non_group_member_returns_without_collectives(broadcast):
    fn, _ = broadcast
    tensor = MagicMock()
    tensor.device.type = "npu"
    non_member = torch.distributed.distributed_c10d.GroupMember.NON_GROUP_MEMBER
    with patch.object(torch.distributed, "all_gather") as gather:
        assert fn(tensor, src=0, group=non_member) is None
    gather.assert_not_called()


def test_world_group_is_used_when_group_is_omitted(broadcast):
    fn, _ = broadcast
    tensor = MagicMock()
    tensor.device.type = "npu"
    c10d = torch.distributed.distributed_c10d
    with (
        patch.object(c10d, "get_group_rank", side_effect=lambda group, rank: rank) as group_rank,
        patch.object(torch.distributed, "get_rank", return_value=0),
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch.object(torch, "empty_like", return_value=MagicMock()),
        patch.object(torch.distributed, "all_gather"),
    ):
        fn(tensor, src=1)
    group_rank.assert_called_once_with(c10d.group.WORLD, 1)


@pytest.mark.parametrize("op", [torch.distributed.ReduceOp.SUM, torch.distributed.ReduceOp.MAX])
@pytest.mark.parametrize("async_op", [False, True])
def test_cpu_int64_all_reduce_preserves_inplace_result_and_handle(communication, op, async_op):
    tensor = torch.tensor([1], dtype=torch.int64)
    group = object()
    handle = MagicMock() if async_op else None

    def native_reduce(value, reduction, process_group, asynchronous):
        value.fill_(3)
        return handle

    communication.original_all_reduce.side_effect = native_reduce
    result = communication.all_reduce(tensor, op=op, group=group, async_op=async_op)
    assert tensor.tolist() == [3]
    assert result is handle
    communication.original_all_reduce.assert_called_once_with(tensor, op, group, async_op)
