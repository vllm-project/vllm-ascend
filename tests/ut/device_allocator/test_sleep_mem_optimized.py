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
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.device_allocator.sleep_mem_optimized import (
    AclGraphSleepPolicy,
    AclGraphSleepWakeupManager,
    HcclSleepPolicy,
    HcclSleepWakeupManager,
    SleepLifecycleState,
    SleepLifecycleStateMachine,
    SleepPolicyConfig,
    SleepWakeupManager,
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
    manager = AclGraphSleepWakeupManager(
        MagicMock(),
        lambda: model_runner,
        SleepPolicyConfig(acl_graph=AclGraphSleepPolicy.RECAPTURE),
    )

    assert manager.wakeup(tags=["weights"]) == "deferred"
    model_runner.capture_model.assert_not_called()

    manager.wakeup(tags=["kv_cache"])
    model_runner.capture_model.assert_called_once_with()


def test_acl_graph_inplace_keeps_entries_without_recapture():
    model_runner = MagicMock()
    manager = AclGraphSleepWakeupManager(
        MagicMock(),
        lambda: model_runner,
        SleepPolicyConfig(acl_graph=AclGraphSleepPolicy.INPLACE),
    )
    wrapper = MagicMock()
    wrapper.concrete_aclgraph_entries = {"keep": object()}
    with (
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.acl_graph._graph_params", None),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.acl_graph._draft_graph_params", None),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.acl_graph._draft_graph_prefill_params", None),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.acl_graph._acl_graph_wrappers",
            [wrapper],
        ),
    ):
        assert manager.sleep() == "inplace"
        assert manager.wakeup(tags=None) == "inplace"
    model_runner.capture_model.assert_not_called()


def test_sleep_wakeup_manager_skips_acl_sleep_when_aclgraph_disabled():
    model_runner = MagicMock()
    model_runner.use_aclgraph = False
    manager = SleepWakeupManager(MagicMock(), MagicMock(), lambda: model_runner)
    manager.acl_graph.sleep = MagicMock(return_value="skip")
    manager.hccl.sleep = MagicMock(return_value="destroy")
    with patch(
        "vllm_ascend.device_allocator.sleep_mem_optimized.torch.npu.mem_get_info",
        side_effect=[(10, 20), (12, 20)],
    ):
        manager.sleep()

    manager.acl_graph.sleep.assert_not_called()
    manager.hccl.sleep.assert_called_once_with()
    assert manager.lifecycle.state == SleepLifecycleState.ASLEEP


def test_sleep_wakeup_manager_cleans_acl_before_hccl_when_aclgraph_enabled():
    model_runner = MagicMock()
    model_runner.use_aclgraph = True
    manager = SleepWakeupManager(MagicMock(), MagicMock(), lambda: model_runner)
    calls: list[str] = []

    def _acl_sleep():
        calls.append("acl")
        return "inplace"

    def _hccl_sleep():
        calls.append("hccl")
        return "lightweight"

    manager.acl_graph.sleep = MagicMock(side_effect=_acl_sleep)
    manager.hccl.sleep = MagicMock(side_effect=_hccl_sleep)

    mem_info = [(10, 20), (12, 20)]
    with patch("vllm_ascend.device_allocator.sleep_mem_optimized.torch.npu.mem_get_info", side_effect=mem_info):
        manager.sleep()

    assert calls == ["acl", "hccl"]


def test_hccl_wakeup_restores_and_refreshes_moe_groups():
    manager = HcclSleepWakeupManager(
        MagicMock(),
        MagicMock(),
        SleepPolicyConfig(hccl=HcclSleepPolicy.DESTROY),
    )

    with (
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.set_current_vllm_config", return_value=nullcontext()),
        patch.object(manager, "restore_hccl", return_value=2) as mock_restore,
        patch.object(manager, "refresh_moe_hccl_groups") as mock_refresh,
    ):
        assert manager.wakeup() == "destroy"

    mock_restore.assert_called_once_with()
    mock_refresh.assert_called_once_with()


def test_hccl_lightweight_suspend_resume_path():
    worker = MagicMock()
    worker._pp_send_work = []
    manager = HcclSleepWakeupManager(
        MagicMock(),
        worker,
        SleepPolicyConfig(hccl=HcclSleepPolicy.LIGHTWEIGHT, allow_fallback=False),
    )
    with (
        patch.object(manager, "_drain_pp_and_sync"),
        patch.object(manager, "_suspend_all_comms", return_value=2),
        patch.object(manager, "_resume_all_comms", return_value=2),
        patch.object(manager, "refresh_moe_hccl_groups"),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.set_current_vllm_config", return_value=nullcontext()),
    ):
        assert manager.sleep() == "lightweight"
        assert manager.wakeup() == "lightweight"


def test_state_machine_rejects_illegal_transition():
    sm = SleepLifecycleStateMachine()
    with pytest.raises(RuntimeError, match="Illegal sleep lifecycle transition"):
        sm.transition(SleepLifecycleState.ASLEEP)


def test_state_machine_sleep_wake_happy_path():
    sm = SleepLifecycleStateMachine()
    sm.transition(SleepLifecycleState.SLEEPING)
    sm.transition(SleepLifecycleState.ASLEEP)
    sm.transition(SleepLifecycleState.WAKING)
    sm.transition(SleepLifecycleState.IDLE)
    assert sm.state == SleepLifecycleState.IDLE


def test_sleep_rollback_on_hccl_failure():
    model_runner = MagicMock()
    model_runner.use_aclgraph = True
    manager = SleepWakeupManager(MagicMock(), MagicMock(), lambda: model_runner)
    manager.acl_graph.sleep = MagicMock(return_value="inplace")
    manager.hccl.sleep = MagicMock(side_effect=RuntimeError("hccl boom"))
    manager.acl_graph.wakeup = MagicMock(return_value="inplace")
    manager.hccl.wakeup = MagicMock(return_value="lightweight")
    with (
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.torch.npu.mem_get_info",
            return_value=(10, 20),
        ),
        pytest.raises(RuntimeError, match="hccl boom"),
    ):
        manager.sleep()
    manager.hccl.wakeup.assert_called()
    assert manager.lifecycle.state == SleepLifecycleState.IDLE

def test_hccl_suspend_barriers_only_before_suspend():
    """Suspended HCCL cannot join collectives; no barriers after suspend."""
    worker = MagicMock()
    worker._pp_send_work = []
    vllm_config = MagicMock()
    manager = HcclSleepWakeupManager(
        vllm_config,
        worker,
        SleepPolicyConfig(hccl=HcclSleepPolicy.LIGHTWEIGHT, allow_fallback=False),
    )

    group = MagicMock()
    group.unique_name = "tp"
    group.device_group = MagicMock()
    calls: list[str] = []

    def _safe_barrier(_device_group=None):
        calls.append("safe_barrier")

    def _global_barrier():
        calls.append("global_barrier")

    def _suspend(comm):
        calls.append(f"suspend:{int(comm)}")
        return True

    with (
        patch.object(manager, "_iter_groups_stable", return_value=[group]),
        patch.object(manager, "_safe_barrier", side_effect=_safe_barrier),
        patch.object(manager, "_global_barrier", side_effect=_global_barrier),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.hccl_suspend_available",
            return_value=True,
        ),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.get_backend_hccl_comm",
            return_value=101,
        ),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.suspend_hccl_comm",
            side_effect=_suspend,
        ),
    ):
        assert manager._suspend_all_comms() == 1

    assert calls == ["global_barrier", "safe_barrier", "suspend:101"]
    assert "safe_barrier" not in calls[calls.index("suspend:101") + 1 :]


def test_hccl_resume_barriers_only_after_resume():
    """Barriers must not run while communicators are still suspended."""
    worker = MagicMock()
    worker._pp_send_work = []
    manager = HcclSleepWakeupManager(
        MagicMock(),
        worker,
        SleepPolicyConfig(hccl=HcclSleepPolicy.LIGHTWEIGHT, allow_fallback=False),
    )
    manager._suspended_comms = [202]
    group = MagicMock()
    group.unique_name = "tp"
    group.device_group = MagicMock()
    calls: list[str] = []

    def _safe_barrier(_device_group=None):
        calls.append("safe_barrier")

    def _global_barrier():
        calls.append("global_barrier")

    def _resume_pg(_device_group=None):
        calls.append("resume_pg")
        return True

    with (
        patch.object(manager, "_iter_groups_stable", return_value=[group]),
        patch.object(manager, "_safe_barrier", side_effect=_safe_barrier),
        patch.object(manager, "_global_barrier", side_effect=_global_barrier),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.resume_all_registered_pgs",
            return_value=0,
        ),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.resume_process_group_hccl",
            side_effect=_resume_pg,
        ),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.get_backend_hccl_comm",
            return_value=202,
        ),
        patch(
            "vllm_ascend.device_allocator.sleep_mem_optimized.clear_suspended_handles",
        ),
    ):
        assert manager._resume_all_comms() == 1

    assert calls[0] == "resume_pg"
    assert "safe_barrier" in calls
    assert calls.index("resume_pg") < calls.index("safe_barrier")
    assert calls[-1] == "global_barrier"
