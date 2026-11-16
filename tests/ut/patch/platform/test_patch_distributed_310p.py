# SPDX-License-Identifier: Apache-2.0
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def broadcast():
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_distributed.py"
    spec = spec_from_file_location("test_310p_communication", path)
    module = module_from_spec(spec)
    with (
        patch(
            "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
            return_value=SimpleNamespace(supports=lambda cap: False),
        ),
        patch.object(torch.distributed, "broadcast") as original,
        patch.object(torch.distributed.distributed_c10d, "broadcast"),
        patch.object(torch.distributed, "all_reduce"),
        patch.object(torch.distributed.distributed_c10d, "all_reduce"),
    ):
        spec.loader.exec_module(module)
        module.communication_adaptation_310p()
        yield torch.distributed.broadcast, original


def test_cpu_broadcast_delegates_rank_validation_and_async_handle(broadcast):
    fn, original = broadcast
    tensor = torch.ones(1)
    group = object()
    assert fn(tensor, group=group, group_src=1, async_op=True) is original.return_value
    original.assert_called_once_with(tensor, src=None, group=group, async_op=True, group_src=1)


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
