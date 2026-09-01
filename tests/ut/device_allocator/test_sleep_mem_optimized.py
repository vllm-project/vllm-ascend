#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.device_allocator.sleep_mem_optimized import (
    AclGraphSleepWakeupManager,
    HcclSleepWakeupManager,
    SleepWakeupManager,
)
from vllm_ascend.utils import (
    SLEEP_LIFECYCLE_ANCHOR_BUFFER_SIZE,
    SLEEP_LIFECYCLE_ANCHOR_GROUP_NAME,
    get_hccl_config_for_pg_options,
)


@dataclass
class DummyGraphParams:
    events: dict[int, list]
    workspaces: dict[int, object]
    extra_handles: dict[int, list]
    metadata: dict[int, tuple]


def test_acl_graph_reset_graph_params_clears_list_values_only():
    workspace = object()
    params = DummyGraphParams(
        events={1: ["event"]},
        workspaces={1: workspace},
        extra_handles={1: ["handle"]},
        metadata={1: ("keep",)},
    )

    AclGraphSleepWakeupManager.reset_graph_params(params)

    assert params.events == {1: []}
    assert params.extra_handles == {1: []}
    assert params.workspaces == {1: workspace}
    assert params.metadata == {1: ("keep",)}


def test_acl_graph_wakeup_waits_for_kv_cache_tag():
    model_runner = MagicMock()
    manager = AclGraphSleepWakeupManager(MagicMock(), lambda: model_runner)

    manager.wakeup(tags=["weights"])
    model_runner.capture_model.assert_not_called()

    manager.wakeup(tags=["kv_cache"])
    model_runner.capture_model.assert_called_once_with()


def test_sleep_wakeup_manager_skips_acl_sleep_when_aclgraph_disabled():
    model_runner = MagicMock()
    model_runner.use_aclgraph = False
    manager = SleepWakeupManager(MagicMock(), MagicMock(), lambda: model_runner)
    manager.acl_graph.sleep = MagicMock()
    manager.hccl.sleep = MagicMock()
    with patch(
        "vllm_ascend.device_allocator.sleep_mem_optimized.torch.npu.mem_get_info",
        side_effect=[(10, 20), (12, 20)],
    ):
        manager.sleep()

    manager.acl_graph.sleep.assert_not_called()
    manager.hccl.sleep.assert_called_once_with()


def test_sleep_wakeup_manager_cleans_acl_before_hccl_when_aclgraph_enabled():
    model_runner = MagicMock()
    model_runner.use_aclgraph = True
    manager = SleepWakeupManager(MagicMock(), MagicMock(), lambda: model_runner)
    calls = []
    manager.acl_graph.sleep = MagicMock(side_effect=lambda: calls.append("acl"))
    manager.hccl.sleep = MagicMock(side_effect=lambda: calls.append("hccl"))

    mem_info = [(10, 20), (12, 20), (12, 20), (13, 20)]
    with patch("vllm_ascend.device_allocator.sleep_mem_optimized.torch.npu.mem_get_info", side_effect=mem_info):
        manager.sleep()

    assert calls == ["acl", "hccl"]


def test_hccl_wakeup_restores_and_refreshes_moe_groups():
    manager = HcclSleepWakeupManager(MagicMock(), MagicMock())

    with (
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.set_current_vllm_config", return_value=nullcontext()),
        patch.object(manager, "restore_hccl", return_value=2) as mock_restore,
        patch.object(manager, "refresh_moe_hccl_groups") as mock_refresh,
    ):
        manager.wakeup()

    mock_restore.assert_called_once_with()
    mock_refresh.assert_called_once_with()


def test_hccl_lifecycle_anchor_uses_minimum_supported_buffer_size():
    assert get_hccl_config_for_pg_options(SLEEP_LIFECYCLE_ANCHOR_GROUP_NAME) == {
        "hccl_buffer_size": SLEEP_LIFECYCLE_ANCHOR_BUFFER_SIZE
    }
    assert SLEEP_LIFECYCLE_ANCHOR_BUFFER_SIZE == 1


def test_hccl_lifecycle_anchor_is_physically_initialized_once_and_reused():
    manager = HcclSleepWakeupManager(MagicMock(), MagicMock())
    device_group = object()
    anchor_group = SimpleNamespace(device_group=device_group)

    with (
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.torch.distributed.get_world_size",
            return_value=4,
        ),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.get_world_group",
            return_value=SimpleNamespace(local_rank=2),
        ),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.init_model_parallel_group",
            return_value=anchor_group,
        ) as mock_init_group,
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.torch.npu.current_device",
            return_value=2,
        ),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.torch.distributed.barrier") as mock_barrier,
    ):
        assert manager._ensure_lifecycle_anchor() is True
        assert manager._ensure_lifecycle_anchor() is True

    mock_init_group.assert_called_once_with(
        [[0, 1, 2, 3]],
        2,
        "hccl",
        use_device_communicator=False,
        group_name=SLEEP_LIFECYCLE_ANCHOR_GROUP_NAME,
    )
    assert mock_barrier.call_count == 2
    mock_barrier.assert_called_with(group=device_group, device_ids=[2])


def test_hccl_lifecycle_anchor_skipped_for_single_rank():
    manager = HcclSleepWakeupManager(MagicMock(), MagicMock())
    tp_group = MagicMock(group_name="tp")

    with (
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.torch.distributed.get_world_size",
            return_value=1,
        ),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.init_model_parallel_group",
        ) as mock_init_group,
        patch.object(manager, "iter_alive_group_coordinators", return_value=[tp_group]),
    ):
        assert manager._ensure_lifecycle_anchor() is False
        assert manager.destroy_hccl() == 0

    mock_init_group.assert_not_called()
    tp_group.destroy_hccl.assert_not_called()
    assert manager._lifecycle_anchor_group is None
    assert manager._skip_hccl_cleanup_for_cycle


def test_hccl_lifecycle_anchor_restores_existing_coordinator_for_next_cycle():
    manager = HcclSleepWakeupManager(MagicMock(), MagicMock())
    device_group = object()
    anchor_group = SimpleNamespace(device_group=None)

    def restore_hccl():
        anchor_group.device_group = device_group
        return True

    anchor_group.restore_hccl = MagicMock(side_effect=restore_hccl)
    manager._lifecycle_anchor_group = anchor_group

    with (
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.torch.distributed.get_world_size",
            return_value=4,
        ),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.torch.npu.current_device",
            return_value=1,
        ),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.torch.distributed.barrier") as mock_barrier,
    ):
        assert manager._ensure_lifecycle_anchor() is True

    anchor_group.restore_hccl.assert_called_once_with()
    mock_barrier.assert_called_once_with(group=device_group, device_ids=[1])


def test_hccl_lifecycle_anchor_released_after_recovery():
    manager = HcclSleepWakeupManager(MagicMock(), MagicMock())
    device_group = object()
    anchor_group = MagicMock(device_group=device_group)
    anchor_group.destroy_hccl.return_value = True
    manager._lifecycle_anchor_group = anchor_group

    with (
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.torch.npu.current_device",
            return_value=3,
        ),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.torch.distributed.barrier") as mock_barrier,
    ):
        assert manager.release_lifecycle_anchor() is True

    mock_barrier.assert_called_once_with(group=device_group, device_ids=[3])
    anchor_group.destroy_hccl.assert_called_once_with()


def test_hccl_lifecycle_anchor_initialization_failure_is_safe():
    manager = HcclSleepWakeupManager(MagicMock(), MagicMock())

    with (
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.torch.distributed.get_world_size",
            return_value=4,
        ),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.get_world_group",
            return_value=SimpleNamespace(local_rank=0),
        ),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.init_model_parallel_group",
            side_effect=RuntimeError("new_group failed"),
        ),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.logger.exception") as mock_log,
    ):
        assert manager._ensure_lifecycle_anchor() is False

    assert manager._lifecycle_anchor_group is None
    mock_log.assert_called_once()


def test_hccl_sleep_preserves_anchor_and_manages_business_groups():
    manager = HcclSleepWakeupManager(MagicMock(), MagicMock())
    anchor_group = MagicMock(group_name=SLEEP_LIFECYCLE_ANCHOR_GROUP_NAME)
    tp_group = MagicMock(group_name="tp")
    ep_group = MagicMock(group_name="ep")
    manager._lifecycle_anchor_group = anchor_group
    tp_group.destroy_hccl.return_value = True
    ep_group.destroy_hccl.return_value = True
    tp_group.restore_hccl.return_value = True
    ep_group.restore_hccl.return_value = True
    groups = [anchor_group, tp_group, ep_group]

    with (
        patch.object(manager, "_ensure_lifecycle_anchor", return_value=True),
        patch.object(manager, "iter_alive_group_coordinators", return_value=groups),
    ):
        assert manager.destroy_hccl() == 2
    with patch.object(manager, "iter_alive_group_coordinators", return_value=groups):
        assert manager.restore_hccl() == 2

    anchor_group.destroy_hccl.assert_not_called()
    anchor_group.restore_hccl.assert_not_called()
    tp_group.destroy_hccl.assert_called_once_with()
    ep_group.destroy_hccl.assert_called_once_with()
    tp_group.restore_hccl.assert_called_once_with()
    ep_group.restore_hccl.assert_called_once_with()


def test_hccl_sleep_skips_teardown_when_anchor_initialization_fails():
    manager = HcclSleepWakeupManager(MagicMock(), MagicMock())
    business_group = MagicMock(group_name="tp")

    with (
        patch.object(manager, "_ensure_lifecycle_anchor", return_value=False),
        patch.object(manager, "iter_alive_group_coordinators", return_value=[business_group]),
    ):
        assert manager.destroy_hccl() == 0
    with patch.object(manager, "iter_alive_group_coordinators", return_value=[business_group]):
        assert manager.restore_hccl() == 0

    business_group.destroy_hccl.assert_not_called()
    business_group.restore_hccl.assert_not_called()


def test_sleep_wakeup_releases_anchor_after_aclgraph_recapture():
    model_runner = MagicMock(use_aclgraph=True)
    manager = SleepWakeupManager(MagicMock(), MagicMock(), lambda: model_runner)
    calls: list[str] = []
    manager.hccl.wakeup = MagicMock(side_effect=lambda: calls.append("hccl"))
    manager.acl_graph.wakeup = MagicMock(side_effect=lambda tags: calls.append("acl"))
    manager.hccl.release_lifecycle_anchor = MagicMock(side_effect=lambda: calls.append("release"))

    manager.wakeup()

    assert calls == ["hccl", "acl", "release"]


def test_staged_wakeup_keeps_anchor_until_kv_cache_recapture():
    model_runner = MagicMock(use_aclgraph=True)
    manager = SleepWakeupManager(MagicMock(), MagicMock(), lambda: model_runner)
    manager.hccl.wakeup = MagicMock()
    manager.acl_graph.wakeup = MagicMock()
    manager.hccl.release_lifecycle_anchor = MagicMock()

    manager.wakeup(tags=["weights"])
    manager.hccl.release_lifecycle_anchor.assert_not_called()

    manager.wakeup(tags=["kv_cache"])
    manager.hccl.release_lifecycle_anchor.assert_called_once_with()
