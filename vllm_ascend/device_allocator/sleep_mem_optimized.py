#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import fields
from typing import Any

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.distributed.parallel_state import _groups, get_world_group, init_model_parallel_group
from vllm.logger import logger
from vllm.utils.mem_constants import GiB_bytes

from vllm_ascend.compilation import acl_graph
from vllm_ascend.utils import SLEEP_LIFECYCLE_ANCHOR_GROUP_NAME


class SleepCycleReport:
    """Evidence record for one sleep→wake cycle (review / regression gate).

    Transitional backend records destroy/restore and optional graph recapture.
    It does not claim native HCCL suspend or inplace graph reuse.

    Plain class (not @dataclass) so isolated importlib loaders used by UT can
    exec the module before it is fully registered in ``sys.modules``.
    """

    def __init__(
        self,
        round: int = 0,
        tp_size: int = 1,
        hccl_backend: str = "noop",
        hccl_rebuilt: bool = False,
        acl_graph_invalidated: bool = False,
        capture_model_called: bool = False,
        hbm_free_before_bytes: int = 0,
        hbm_free_after_sleep_bytes: int = 0,
        hbm_freed_bytes: int = 0,
        wake_completed: bool = False,
        notes: str = "",
    ):
        self.round = round
        self.tp_size = tp_size
        self.hccl_backend = hccl_backend
        self.hccl_rebuilt = hccl_rebuilt
        self.acl_graph_invalidated = acl_graph_invalidated
        self.capture_model_called = capture_model_called
        self.hbm_free_before_bytes = hbm_free_before_bytes
        self.hbm_free_after_sleep_bytes = hbm_free_after_sleep_bytes
        self.hbm_freed_bytes = hbm_freed_bytes
        self.wake_completed = wake_completed
        self.notes = notes

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "tp_size": self.tp_size,
            "hccl_backend": self.hccl_backend,
            "hccl_rebuilt": self.hccl_rebuilt,
            "acl_graph_invalidated": self.acl_graph_invalidated,
            "capture_model_called": self.capture_model_called,
            "hbm_free_before_bytes": self.hbm_free_before_bytes,
            "hbm_free_after_sleep_bytes": self.hbm_free_after_sleep_bytes,
            "hbm_freed_bytes": self.hbm_freed_bytes,
            "wake_completed": self.wake_completed,
            "notes": self.notes,
        }


class SleepWakeupManager:
    """Ascend ordering around the upstream communicator lifecycle contract."""

    def __init__(self, vllm_config: VllmConfig, worker: Any, model_runner_getter: Callable[[], Any]):
        self.acl_graph = AclGraphSleepWakeupManager(vllm_config, model_runner_getter)
        self.hccl = HcclSleepWakeupManager(vllm_config, worker)
        self._model_runner_getter = model_runner_getter
        self._sleeping = False
        self._weights_awake = False
        self._failed = False
        self._cycle_round = 0
        self.last_cycle_report: SleepCycleReport | None = None
        self._last_rebuilds_hccl = False
        self._last_invalidates_graphs = False
        self._hbm_free_before_bytes = 0
        self._hbm_free_after_sleep_bytes = 0

    def _validate(self) -> None:
        if self._failed:
            raise RuntimeError("Sleep lifecycle failed; restart the worker before retrying.")

    def validate_sleep(self) -> bool:
        self._validate()
        if self._sleeping and self._weights_awake:
            raise RuntimeError("Finish the staged KV-cache wake before starting another sleep cycle.")
        return not self._sleeping

    def _tp_size(self) -> int:
        try:
            parallel_config = getattr(self.hccl.vllm_config, "parallel_config", None)
            tp = getattr(parallel_config, "tensor_parallel_size", None)
            if tp is not None:
                return max(int(tp), 1)
        except Exception:
            pass
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return max(int(torch.distributed.get_world_size()), 1)
        except Exception:
            return 1
        return 1

    def _finalize_cycle_report(self, *, wake_completed: bool) -> SleepCycleReport:
        self._cycle_round += 1
        freed = max(self._hbm_free_after_sleep_bytes - self._hbm_free_before_bytes, 0)
        report = SleepCycleReport(
            round=self._cycle_round,
            tp_size=self._tp_size(),
            hccl_backend="destroy_restore" if self._last_rebuilds_hccl else "noop",
            hccl_rebuilt=self._last_rebuilds_hccl,
            acl_graph_invalidated=self._last_invalidates_graphs,
            capture_model_called=bool(self.acl_graph.capture_model_called),
            hbm_free_before_bytes=int(self._hbm_free_before_bytes),
            hbm_free_after_sleep_bytes=int(self._hbm_free_after_sleep_bytes),
            hbm_freed_bytes=int(freed),
            wake_completed=wake_completed,
            notes=(
                "transitional destroy/restore backend" if self._last_rebuilds_hccl else "no HCCL teardown this cycle"
            ),
        )
        self.last_cycle_report = report
        return report

    def sleep(self) -> None:
        if not self.validate_sleep():
            return
        free_before = torch.npu.mem_get_info()[0]
        self._hbm_free_before_bytes = int(free_before)
        try:
            rebuilds_hccl = self.hccl.prepare_sleep()
            # A destroy/restore backend cannot replay graphs captured against
            # the old native communicators. Decide before teardown, but leave
            # graphs/workspaces intact when this cycle has no HCCL work.
            invalidates_graphs = rebuilds_hccl and self._model_runner_getter().use_aclgraph
            if invalidates_graphs:
                self.acl_graph.sleep()
            self.hccl.sleep()
        except Exception:
            self._failed = True
            raise
        self._sleeping = True
        self._weights_awake = False
        self._last_rebuilds_hccl = bool(rebuilds_hccl)
        self._last_invalidates_graphs = bool(invalidates_graphs)
        free_after = torch.npu.mem_get_info()[0]
        self._hbm_free_after_sleep_bytes = int(free_after)
        released = free_after - free_before
        logger.info(
            "Sleep extra cleanup released %.3f GiB (HCCL rebuild=%s, ACL Graph invalidated=%s).",
            released / GiB_bytes,
            rebuilds_hccl,
            invalidates_graphs,
        )

    def validate_wakeup(self, tags: list[str] | None = None) -> None:
        self._validate()
        if self._sleeping and tags is not None and "kv_cache" in tags:
            if not self._weights_awake and "weights" not in tags:
                raise ValueError("Wake weights and finish reloading them before waking kv_cache, or wake all tags.")

    def wakeup(self, tags: list[str] | None = None) -> None:
        self.validate_wakeup(tags)
        if not self._sleeping:
            return
        try:
            # Errors must reach Worker/RPC; returning normally would advertise
            # a usable engine with missing communication or graph resources.
            self.hccl.wakeup()
            if tags is None or "weights" in tags:
                self._weights_awake = True
            if self._model_runner_getter().use_aclgraph:
                self.acl_graph.wakeup(tags)
            if tags is None or "kv_cache" in tags:
                self.hccl.release_lifecycle_anchor()
                self._sleeping = False
                self._finalize_cycle_report(wake_completed=True)
        except Exception:
            self._failed = True
            raise


class AclGraphSleepWakeupManager:
    def __init__(self, vllm_config: VllmConfig, model_runner_getter: Callable[[], Any]):
        self.vllm_config = vllm_config
        self._model_runner_getter = model_runner_getter
        self._needs_recapture = False
        self.capture_model_called = False

    @staticmethod
    def clear_attention_workspaces(params) -> None:
        if params is None:
            return
        for num_tokens in params.workspaces:
            params.workspaces[num_tokens] = None

    @classmethod
    def clear_all_attention_workspaces(cls) -> None:
        cls.clear_attention_workspaces(acl_graph._graph_params)
        cls.clear_attention_workspaces(acl_graph._draft_graph_params)
        cls.clear_attention_workspaces(acl_graph._draft_graph_prefill_params)

    @staticmethod
    def reset_graph_params(params) -> None:
        if params is None:
            return
        for graph_field in fields(params):
            attr_dict = getattr(params, graph_field.name, None)
            if not isinstance(attr_dict, MutableMapping):
                continue
            for num_tokens, value in attr_dict.items():
                if isinstance(value, list):
                    attr_dict[num_tokens] = []

    @classmethod
    def reset_all_graph_params(cls) -> None:
        cls.reset_graph_params(acl_graph._graph_params)
        cls.reset_graph_params(acl_graph._draft_graph_params)
        cls.reset_graph_params(acl_graph._draft_graph_prefill_params)
        for wrapper in list(acl_graph._acl_graph_wrappers):
            wrapper.concrete_aclgraph_entries.clear()
            wrapper.first_run_finished = False

    @staticmethod
    def reset_model_runner_graph_manager(model_runner: Any) -> None:
        manager = getattr(model_runner, "cudagraph_manager", None)
        if manager is None:
            return
        if hasattr(manager, "graphs"):
            manager.graphs.clear()
        if hasattr(manager, "_graphs_captured"):
            manager._graphs_captured = False

    def sleep(self) -> None:
        self.capture_model_called = False
        self.clear_all_attention_workspaces()
        self.reset_all_graph_params()
        self.reset_model_runner_graph_manager(self._model_runner_getter())
        self._needs_recapture = True

    def wakeup(self, tags: list[str] | None = None) -> None:
        if not self._needs_recapture or (tags is not None and "kv_cache" not in tags):
            # Level-2 wakeup restores weights before external weight loading;
            # recapture graphs only after KV cache is restored.
            return
        model_runner = self._model_runner_getter()
        with set_current_vllm_config(self.vllm_config):
            model_runner.capture_model()
            self.capture_model_called = True
        self._needs_recapture = False


class HcclSleepWakeupManager:
    def __init__(self, vllm_config: VllmConfig, worker: Any):
        self.vllm_config = vllm_config
        self.worker = worker
        self._lifecycle_anchor_group: Any | None = None
        self._prepared = False
        self._suspended = False
        self._manage_comms = False

    @staticmethod
    def iter_alive_group_coordinators():
        for group_ref in list(_groups.values()):
            group = group_ref()
            if group is None:
                continue
            yield group

    def _ensure_lifecycle_anchor(self) -> bool:
        """Create and physically initialize the HCCL sleep anchor.

        HCCL process-group creation can be lazy. Running a device barrier is
        therefore required before treating the group as a lifecycle anchor.
        The coordinator is registered with vLLM's normal distributed
        lifecycle. Its HCCL communicator is restored once per sleep cycle and
        released after business groups and ACL graphs are restored. Single-rank
        processes skip the anchor: there is no multi-rank HCCL control plane
        to keep alive.
        """
        try:
            if torch.distributed.get_world_size() <= 1:
                return False
            if self._lifecycle_anchor_group is None:
                world_size = torch.distributed.get_world_size()
                ranks = list(range(world_size))
                self._lifecycle_anchor_group = init_model_parallel_group(
                    [ranks],
                    get_world_group().local_rank,
                    "hccl",
                    use_device_communicator=False,
                    group_name=SLEEP_LIFECYCLE_ANCHOR_GROUP_NAME,
                )

            device_group = getattr(self._lifecycle_anchor_group, "device_group", None)
            if device_group is None:
                if not self._lifecycle_anchor_group.restore_hccl():
                    raise RuntimeError("failed to restore the HCCL lifecycle anchor")
                device_group = getattr(self._lifecycle_anchor_group, "device_group", None)

            torch.distributed.barrier(
                group=device_group,
                device_ids=[torch.npu.current_device()],
            )
            return True
        except Exception:
            logger.exception("Failed to initialize the HCCL sleep lifecycle anchor; refusing HCCL teardown.")
            return False

    def release_lifecycle_anchor(self) -> bool:
        """Release the temporary anchor after business groups and graphs recover."""
        anchor_group = self._lifecycle_anchor_group
        device_group = getattr(anchor_group, "device_group", None)
        if anchor_group is None or device_group is None:
            return False

        # All ranks must finish recapture before the collective anchor is
        # destroyed. The coordinator remains registered and is restored on the
        # next sleep cycle instead of creating another group with the same name.
        torch.distributed.barrier(
            group=device_group,
            device_ids=[torch.npu.current_device()],
        )
        destroyed = anchor_group.destroy_hccl()
        if destroyed:
            logger.info(
                "Released HCCL sleep lifecycle anchor '%s' after wake-up recovery.",
                SLEEP_LIFECYCLE_ANCHOR_GROUP_NAME,
            )
        return destroyed

    def _validate_group_plan(self, error: str | None = None) -> bool:
        """Agree on live PGs/order and report whether HCCL teardown is needed."""
        plan = []
        try:
            for group in self.iter_alive_group_coordinators():
                if group is self._lifecycle_anchor_group:
                    continue
                tears_down_hccl = group.backend == "hccl" and getattr(group, "device_group", None) is not None
                plan.append((group.unique_name, group.backend, group.group_ranks, tears_down_hccl))
                if group.backend == "hccl":
                    group._hccl_sleep_adapter.validate()
                    if not tears_down_hccl:
                        raise RuntimeError(f"HCCL owner {group.unique_name} has no live device group before sleep.")
        except Exception as exc:
            error = str(exc)
        world = get_world_group()
        local = (plan, error)
        gathered = [None] * torch.distributed.get_world_size(group=world.cpu_group)
        torch.distributed.all_gather_object(gathered, local, group=world.cpu_group)
        if any(item is None or item[1] is not None or item[0] != plan for item in gathered):
            raise RuntimeError(f"HCCL sleep preflight failed or group plans differ across ranks: {gathered}")
        return any(item[3] for item in plan)

    def prepare_sleep(self) -> bool:
        """Prepare the cycle and say whether its backend will rebuild HCCL."""
        if self._prepared or self._suspended:
            return self._manage_comms
        for handle in getattr(self.worker, "_pp_send_work", []):
            handle.wait()
        self.worker._pp_send_work = []
        torch.npu.synchronize()
        self._manage_comms = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        if self._manage_comms:
            self._manage_comms = self._validate_group_plan()
            if self._manage_comms and not self._ensure_lifecycle_anchor():
                raise RuntimeError("HCCL lifecycle anchor is unavailable; refusing sleep teardown.")
        self._prepared = True
        return self._manage_comms

    @staticmethod
    def refresh_moe_hccl_groups() -> None:
        from vllm_ascend.ops.fused_moe.moe_comm_method import _MoECommMethods

        for comm_method in _MoECommMethods.values():
            dispatcher = getattr(comm_method, "token_dispatcher", None)
            refresh_fn = getattr(dispatcher, "refresh_hccl_group", None)
            if callable(refresh_fn) and refresh_fn() is False:
                raise RuntimeError("MoE dispatcher refused HCCL group refresh.")

    @staticmethod
    def prepare_moe_hccl_teardown() -> bool:
        """Prepare MoE resources before their HCCL groups are destroyed."""
        from vllm_ascend.ops.fused_moe.moe_comm_method import _MoECommMethods

        try:
            for comm_method in _MoECommMethods.values():
                prepare_fn = getattr(comm_method, "prepare_hccl_teardown", None)
                if callable(prepare_fn) and prepare_fn() is False:
                    raise RuntimeError("MoE communication method refused HCCL teardown.")
            return True
        except Exception:
            logger.exception("Failed to prepare MoE state for HCCL teardown; refusing sleep teardown.")
            return False

    @staticmethod
    def refresh_moe_hccl_runtime_state() -> None:
        """Refresh MoE runtime resources after HCCL groups are restored."""
        from vllm_ascend.ops.fused_moe.moe_comm_method import _MoECommMethods

        for comm_method in _MoECommMethods.values():
            refresh_fn = getattr(comm_method, "refresh_hccl_runtime_state", None)
            if callable(refresh_fn) and refresh_fn() is False:
                raise RuntimeError("MoE communication method refused HCCL runtime refresh.")

    def sleep(self) -> None:
        if self._suspended:
            return
        self.prepare_sleep()
        if self._manage_comms:
            # Graphs have already been invalidated by SleepWakeupManager.
            # Agree on MoE readiness before any rank releases its first PG.
            error = None
            if not self.prepare_moe_hccl_teardown():
                error = "MoE HCCL teardown preparation failed"
            self._validate_group_plan(error)
            parallel_state.suspend_device_comms()
            logger.info("Suspended HCCL resources using the destroy/restore backend.")
        self._suspended = True

    def wakeup(self) -> None:
        if not self._suspended:
            return
        if self._manage_comms:
            with set_current_vllm_config(self.vllm_config):
                parallel_state.resume_device_comms()
                # Runtime refresh may issue collectives. Every PG must already
                # be restored before refreshing any MoE context.
                self.refresh_moe_hccl_groups()
                self.refresh_moe_hccl_runtime_state()
            logger.info("Restored HCCL resources after sleep.")
        self._suspended = False
        self._prepared = False
