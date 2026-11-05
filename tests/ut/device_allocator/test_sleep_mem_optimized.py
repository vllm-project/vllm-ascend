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
from ctypes import c_void_p
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.device_allocator.hccl_suspend import comm_to_int
from vllm_ascend.device_allocator.sleep_mem_optimized import (
    AclGraphSleepPolicy,
    AclGraphSleepWakeupManager,
    HcclSleepPolicy,
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
        patch.object(manager, "prepare_moe_hccl_teardown", return_value=True) as mock_prepare_moe,
        patch.object(manager, "iter_alive_group_coordinators", return_value=groups),
    ):
        assert manager.destroy_hccl() == 2
    with patch.object(manager, "iter_alive_group_coordinators", return_value=groups):
        assert manager.restore_hccl() == 2

    anchor_group.destroy_hccl.assert_not_called()
    anchor_group.restore_hccl.assert_not_called()
    mock_prepare_moe.assert_called_once_with()
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


def test_hccl_sleep_prepares_moe_state_before_business_group_teardown():
    manager = HcclSleepWakeupManager(MagicMock(), MagicMock())
    business_group = MagicMock(group_name="mc2")
    business_group.destroy_hccl.return_value = True
    calls: list[str] = []

    def record_prepare() -> bool:
        calls.append("prepare")
        return True

    def record_destroy() -> bool:
        calls.append("destroy")
        return True

    with (
        patch.object(manager, "_ensure_lifecycle_anchor", return_value=True),
        patch.object(manager, "prepare_moe_hccl_teardown", side_effect=record_prepare),
        patch.object(manager, "iter_alive_group_coordinators", return_value=[business_group]),
    ):
        business_group.destroy_hccl.side_effect = record_destroy
        assert manager.destroy_hccl() == 1

    assert calls == ["prepare", "destroy"]


def test_hccl_wakeup_refreshes_dispatcher_then_moe_runtime_state():
    manager = HcclSleepWakeupManager(MagicMock(), MagicMock())
    calls: list[str] = []

    def record_restore() -> int:
        calls.append("restore")
        return 2

    def record_dispatcher() -> None:
        calls.append("dispatcher")

    def record_runtime() -> None:
        calls.append("runtime")

    with (
        patch.object(manager, "restore_hccl", side_effect=record_restore),
        patch.object(manager, "refresh_moe_hccl_groups", side_effect=record_dispatcher),
        patch.object(manager, "refresh_moe_hccl_runtime_state", side_effect=record_runtime),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.set_current_vllm_config",
            return_value=nullcontext(),
        ),
    ):
        manager.wakeup()

    assert calls == ["restore", "dispatcher", "runtime"]


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


def test_wakeup_defers_acl_and_anchor_when_hccl_resume_is_incomplete():
    model_runner = MagicMock(use_aclgraph=True)
    manager = SleepWakeupManager(MagicMock(), MagicMock(), lambda: model_runner)
    manager.hccl.wakeup = MagicMock(return_value=False)
    manager.acl_graph.wakeup = MagicMock()
    manager.hccl.release_lifecycle_anchor = MagicMock()

    manager.wakeup()

    manager.acl_graph.wakeup.assert_not_called()
    manager.hccl.release_lifecycle_anchor.assert_not_called()


def test_acl_graph_inplace_skips_recapture_on_main_path():
    model_runner = MagicMock()
    manager = AclGraphSleepWakeupManager(MagicMock(), lambda: model_runner)
    manager.acl_policy = AclGraphSleepPolicy.INPLACE
    wrapper = MagicMock()
    wrapper.concrete_aclgraph_entries = {"keep": object()}
    with patch(
        "vllm_ascend.device_allocator.sleep_mem_optimized.acl_graph._acl_graph_wrappers",
        [wrapper],
    ):
        manager.sleep()
        manager.wakeup(tags=None)
    model_runner.capture_model.assert_not_called()


def test_hccl_lightweight_suspend_path_on_main():
    worker = MagicMock()
    worker._pp_send_work = []
    manager = HcclSleepWakeupManager(MagicMock(), worker)
    manager.hccl_policy = HcclSleepPolicy.LIGHTWEIGHT
    manager.allow_fallback = False
    with (
        patch.object(manager, "_has_multi_rank_group", return_value=False),
        patch.object(manager, "_suspend_all_comms", return_value=1),
        patch.object(manager, "_resume_all_comms", return_value=1),
        patch.object(manager, "refresh_moe_hccl_groups"),
        patch.object(manager, "refresh_moe_hccl_runtime_state"),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.set_current_vllm_config", return_value=nullcontext()),
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.npu.synchronize"),
    ):
        manager.sleep()
        manager.wakeup()


def test_acl_graph_inplace_fallback_when_entries_missing():
    model_runner = MagicMock()
    manager = AclGraphSleepWakeupManager(MagicMock(), lambda: model_runner)
    manager.acl_policy = AclGraphSleepPolicy.INPLACE
    manager.allow_fallback = True
    manager._used_inplace = True
    wrapper = MagicMock()
    wrapper.concrete_aclgraph_entries = {}
    with (
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.acl_graph._acl_graph_wrappers", [wrapper]),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.set_current_vllm_config", return_value=nullcontext()),
    ):
        manager.wakeup(tags=None)
    model_runner.capture_model.assert_called_once_with()


def test_comm_to_int_accepts_ctypes_and_nested_wrapper_handles():
    assert comm_to_int(c_void_p(17)) == 17
    assert comm_to_int(SimpleNamespace(value=c_void_p(23))) == 23
    assert comm_to_int(SimpleNamespace(handle=SimpleNamespace(ptr=29))) == 29
    assert comm_to_int(object()) == 0


def test_hccl_suspend_barriers_all_groups_before_first_suspend():
    manager = HcclSleepWakeupManager(MagicMock(), MagicMock())
    group_a = SimpleNamespace(device_group=SimpleNamespace(size=lambda: 2))
    group_b = SimpleNamespace(device_group=SimpleNamespace(size=lambda: 2))
    calls: list[str] = []

    def _record(label: str) -> None:
        calls.append(label)

    def _suspend(comm: object) -> bool:
        calls.append(f"suspend-{comm}")
        return True

    with (
        patch.object(manager, "iter_alive_group_coordinators", return_value=[group_a, group_b]),
        patch.object(manager, "_global_barrier", side_effect=lambda: _record("global")),
        patch.object(manager, "_safe_barrier", side_effect=lambda _: _record("barrier")),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.hccl_suspend_available", return_value=True),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.get_backend_hccl_comm",
            side_effect=[11, 22],
        ),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.suspend_hccl_comm",
            side_effect=_suspend,
        ),
    ):
        assert manager._suspend_all_comms() == 2

    assert calls == ["global", "barrier", "barrier", "suspend-11", "suspend-22"]


def test_hccl_resume_does_not_resume_unknown_current_handle_or_barrier():
    manager = HcclSleepWakeupManager(MagicMock(), MagicMock())
    group = SimpleNamespace(device_group=SimpleNamespace(size=lambda: 2))
    manager._suspended_comms = [11]
    with (
        patch.object(manager, "iter_alive_group_coordinators", return_value=[group]),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.resume_all_registered_pgs", return_value=0),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.get_backend_hccl_comm",
            return_value=22,
        ),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.resume_process_group_hccl",
            return_value=False,
        ),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.resume_hccl_comm",
            return_value=False,
        ) as mock_resume,
        patch.object(manager, "_safe_barrier") as mock_barrier,
        patch.object(manager, "_global_barrier") as mock_global_barrier,
    ):
        assert manager._resume_all_comms() == -1

    mock_resume.assert_called_once_with(11)
    mock_barrier.assert_not_called()
    mock_global_barrier.assert_not_called()
    assert manager._suspended_comms == [11]


def test_hccl_resume_keeps_group_handle_pending_when_current_handle_changes():
    manager = HcclSleepWakeupManager(MagicMock(), MagicMock())
    group = SimpleNamespace(device_group=SimpleNamespace(size=lambda: 2))
    manager._suspended_comms = [11]
    manager._suspended_group_handles = {id(group.device_group): 11}
    with (
        patch.object(manager, "iter_alive_group_coordinators", return_value=[group]),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.resume_all_registered_pgs", return_value=0),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.get_backend_hccl_comm",
            return_value=22,
        ),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.resume_process_group_hccl",
            return_value=False,
        ),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.resume_hccl_comm",
            return_value=True,
        ) as mock_resume,
        patch.object(manager, "_safe_barrier") as mock_barrier,
        patch.object(manager, "_global_barrier") as mock_global_barrier,
    ):
        assert manager._resume_all_comms() == -1

    mock_resume.assert_not_called()
    mock_barrier.assert_not_called()
    mock_global_barrier.assert_not_called()
    assert manager._suspended_comms == [11]
