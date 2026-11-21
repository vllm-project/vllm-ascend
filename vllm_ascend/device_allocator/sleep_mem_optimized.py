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
from datetime import timedelta
from enum import Enum
from typing import Any

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import _groups, get_world_group, init_model_parallel_group
from vllm.logger import logger
from vllm.utils.mem_constants import GiB_bytes

from vllm_ascend.compilation import acl_graph
from vllm_ascend.device_allocator.hccl_suspend import (
    SLEEP_HCCL_POLICIES,
    HcclSuspendBackend,
    HcclSuspendStats,
    resume_group,
    suspend_group,
)
from vllm_ascend.utils import SLEEP_LIFECYCLE_ANCHOR_GROUP_NAME


class SleepHcclState(str, Enum):
    """Lifecycle of the HCCL side of sleep mode.

    ``idle -> sleeping -> asleep -> waking -> idle`` is the happy path.
    A failed sleep/wake leaves the manager in ``failed`` so the next cycle
    cannot silently continue with a half-suspended communicator set.
    """

    IDLE = "idle"
    SLEEPING = "sleeping"
    ASLEEP = "asleep"
    WAKING = "waking"
    FAILED = "failed"


_ALLOWED_HCCL_TRANSITIONS: dict[SleepHcclState, set[SleepHcclState]] = {
    SleepHcclState.IDLE: {SleepHcclState.SLEEPING},
    SleepHcclState.SLEEPING: {SleepHcclState.ASLEEP, SleepHcclState.IDLE, SleepHcclState.FAILED},
    SleepHcclState.ASLEEP: {SleepHcclState.WAKING, SleepHcclState.FAILED},
    SleepHcclState.WAKING: {SleepHcclState.IDLE, SleepHcclState.FAILED},
    SleepHcclState.FAILED: {SleepHcclState.IDLE},
}


class SleepWakeupManager:
    def __init__(self, vllm_config: VllmConfig, worker: Any, model_runner_getter: Callable[[], Any]):
        self.acl_graph = AclGraphSleepWakeupManager(vllm_config, model_runner_getter)
        self.hccl = HcclSleepWakeupManager(vllm_config, worker)
        self._model_runner_getter = model_runner_getter

    @staticmethod
    def _measure_memory_released(cleanup: Callable[[], None]) -> int:
        free_bytes_before_cleanup = torch.npu.mem_get_info()[0]
        cleanup()
        free_bytes_after_cleanup = torch.npu.mem_get_info()[0]
        return max(free_bytes_after_cleanup - free_bytes_before_cleanup, 0)

    def sleep(self) -> None:
        model_runner = self._model_runner_getter()
        free_bytes_before_cleanup = torch.npu.mem_get_info()[0]
        if model_runner.use_aclgraph:
            self.acl_graph.sleep()
        self.hccl.sleep()
        free_bytes_after_cleanup = torch.npu.mem_get_info()[0]
        free_mem = free_bytes_after_cleanup - free_bytes_before_cleanup
        logger.info(
            "Sleep mode released HCCL and attention workspace memory: %.3f GiB.",
            free_mem / GiB_bytes,
        )

    def wakeup(self, tags: list[str] | None = None) -> None:
        self.hccl.wakeup()
        model_runner = self._model_runner_getter()
        if model_runner.use_aclgraph:
            self.acl_graph.wakeup(tags)
        if tags is None or "kv_cache" in tags:
            # Keep the anchor alive until graph recapture is complete. For
            # staged level-2 wakeup, weights are restored first and recapture
            # happens only after the KV cache is restored.
            self.hccl.release_lifecycle_anchor()


class AclGraphSleepWakeupManager:
    def __init__(self, vllm_config: VllmConfig, model_runner_getter: Callable[[], Any]):
        self.vllm_config = vllm_config
        self._model_runner_getter = model_runner_getter

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
        self.clear_all_attention_workspaces()
        self.reset_all_graph_params()
        self.reset_model_runner_graph_manager(self._model_runner_getter())

    def wakeup(self, tags: list[str] | None = None) -> None:
        if tags is not None and "kv_cache" not in tags:
            # Level-2 wakeup restores weights before external weight loading;
            # recapture graphs only after KV cache is restored.
            return
        model_runner = self._model_runner_getter()
        with set_current_vllm_config(self.vllm_config):
            model_runner.capture_model()


class HcclSleepWakeupManager:
    def __init__(self, vllm_config: VllmConfig, worker: Any):
        self.vllm_config = vllm_config
        self.worker = worker
        self._lifecycle_anchor_group: Any | None = None
        self._skip_hccl_cleanup_for_cycle = False
        self._state = SleepHcclState.IDLE
        self._suspend_backend = HcclSuspendBackend()
        self._suspended_groups: list[tuple[Any, list[int]]] = []
        self._destroyed_groups = False
        self._active_policy: str | None = None

    @property
    def state(self) -> SleepHcclState:
        return self._state

    @property
    def active_policy(self) -> str | None:
        return self._active_policy

    @property
    def suspend_stats(self) -> HcclSuspendStats:
        return self._suspend_backend.stats

    def _transition(self, target: SleepHcclState) -> None:
        allowed = _ALLOWED_HCCL_TRANSITIONS.get(self._state, set())
        if target not in allowed:
            raise RuntimeError(f"Illegal HCCL sleep transition: {self._state.value} -> {target.value}")
        logger.info("HCCL sleep state: %s -> %s", self._state.value, target.value)
        self._state = target

    def _require_state(self, expected: SleepHcclState, action: str) -> None:
        if self._state is not expected:
            raise RuntimeError(f"Cannot {action} HCCL sleep lifecycle while state={self._state.value}")

    @staticmethod
    def _resolve_sleep_hccl_policy() -> tuple[str, bool]:
        policy = "auto"
        allow_fallback = True
        try:
            from vllm_ascend.ascend_config import get_ascend_config

            rl_config = get_ascend_config().rl_config
            candidate = getattr(rl_config, "sleep_hccl_policy", "auto")
            if isinstance(candidate, str) and candidate in SLEEP_HCCL_POLICIES:
                policy = candidate
            fallback = getattr(rl_config, "sleep_allow_fallback", True)
            if isinstance(fallback, bool):
                allow_fallback = fallback
        except Exception:  # noqa: BLE001
            pass
        return policy, allow_fallback

    def _drain_and_sync(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for handle in getattr(self.worker, "_pp_send_work", []):
                handle.wait()
            self.worker._pp_send_work = []
            torch.npu.synchronize()

    def _suspend_all_groups(self) -> bool:
        suspended: list[tuple[Any, list[int]]] = []
        seen_device_groups: set[int] = set()
        seen_handles: set[int] = set()
        for group in self.iter_alive_group_coordinators():
            if group is self._lifecycle_anchor_group:
                continue
            device_group = getattr(group, "device_group", None)
            if device_group is None or id(device_group) in seen_device_groups:
                continue
            seen_device_groups.add(id(device_group))
            try:
                ok, handles = suspend_group(device_group, self._suspend_backend)
            except Exception:  # noqa: BLE001 - a timed-out/partial rank must fall back
                logger.exception("HCCL suspend raised for group '%s'.", getattr(group, "group_name", "anonymous"))
                ok, handles = False, []
            if not ok:
                for done_group, done_handles in reversed(suspended):
                    resume_group(getattr(done_group, "device_group", None), done_handles, self._suspend_backend)
                self._suspend_backend.stats.suspended_handles.clear()
                return False
            fresh_handles = [handle for handle in handles if handle not in seen_handles]
            if fresh_handles:
                seen_handles.update(fresh_handles)
                suspended.append((group, fresh_handles))
                logger.info(
                    "Suspended HCCL group '%s' communicator(s) %s for sleep mode.",
                    getattr(group, "group_name", "anonymous"),
                    fresh_handles,
                )
        self._suspended_groups = suspended
        if not suspended:
            logger.info("Sleep mode: no HCCL communicators to suspend (single-rank or lazy groups).")
        return True

    def _rollback_suspend(self) -> None:
        for group, handles in reversed(self._suspended_groups):
            resume_group(getattr(group, "device_group", None), handles, self._suspend_backend)
        self._suspended_groups = []
        self._suspend_backend.stats.suspended_handles.clear()

    _SUSPEND_CONSENSUS_TIMEOUT_S = 60.0

    def _all_rank_vote(self, local_ok: bool, stage: str) -> bool | None:
        """Run a CPU-process-group MIN vote for a suspend decision.

        Returns ``True`` when every rank agrees, ``False`` when the vote
        completed but at least one rank disagreed, and ``None`` when the vote
        itself failed (timeout/exception).  ``None`` must not fall back to
        destroy: a dead peer would make the destroy barrier hang as well.
        """
        try:
            world_size = torch.distributed.get_world_size()
        except Exception:  # noqa: BLE001
            return local_ok
        if world_size <= 1:
            return local_ok
        try:
            cpu_group = getattr(get_world_group(), "cpu_group", None)
        except Exception:  # noqa: BLE001
            cpu_group = None
        if cpu_group is None:
            logger.warning("No CPU process group available for the HCCL %s vote.", stage)
            return None
        try:
            vote = torch.tensor([1 if local_ok else 0], dtype=torch.int32)
            work = torch.distributed.all_reduce(
                vote,
                op=torch.distributed.ReduceOp.MIN,
                group=cpu_group,
                async_op=True,
            )
            work.wait(timedelta(seconds=self._SUSPEND_CONSENSUS_TIMEOUT_S))
            agreed = bool(int(vote.item()))
        except Exception:  # noqa: BLE001
            logger.exception("HCCL %s vote failed; failing the sleep cycle instead of mixing policies.", stage)
            return None
        if not agreed:
            logger.warning("HCCL %s vote rejected: local_ok=%s.", stage, local_ok)
        return agreed

    def _agree_on_suspend(self, local_ok: bool) -> bool | None:
        """Require every rank to suspend before committing the lightweight path.

        HCCL suspension is per-process state.  If one rank falls back to
        destroy/restore while another keeps a suspended communicator, the job
        is left in a mixed state that can hang on wake-up.
        """
        return self._all_rank_vote(local_ok, "suspend")

    def _force_destroy_and_restore(self) -> None:
        """Last-resort recovery when a suspended group cannot be resumed."""
        logger.warning("HCCL resume failed; recreating suspended process groups.")
        for group, _ in self._suspended_groups:
            try:
                group.destroy_hccl()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to destroy an unresumable HCCL group.")
        self._suspended_groups = []
        self.restore_hccl()

    @staticmethod
    def iter_alive_group_coordinators():
        seen: set[int] = set()
        for group_ref in list(_groups.values()):
            group = group_ref()
            if group is None or id(group) in seen:
                continue
            seen.add(id(group))
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
            logger.exception(
                "Failed to initialize the HCCL sleep lifecycle anchor; sleep_hccl_policy=skip for this sleep cycle."
            )
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

    def destroy_hccl(self) -> int:
        groups = list(self.iter_alive_group_coordinators())
        self._skip_hccl_cleanup_for_cycle = False
        if not self._ensure_lifecycle_anchor():
            self._skip_hccl_cleanup_for_cycle = True
            return 0

        if not self.prepare_moe_hccl_teardown():
            self._skip_hccl_cleanup_for_cycle = True
            return 0

        num_destroyed = 0
        for group in groups:
            if group is self._lifecycle_anchor_group:
                continue
            if group.destroy_hccl():
                num_destroyed += 1
        return num_destroyed

    def restore_hccl(self) -> int:
        if self._skip_hccl_cleanup_for_cycle:
            self._skip_hccl_cleanup_for_cycle = False
            return 0

        num_restored = 0
        for group in self.iter_alive_group_coordinators():
            if group is self._lifecycle_anchor_group:
                continue
            if group.restore_hccl():
                num_restored += 1
        return num_restored

    @staticmethod
    def refresh_moe_hccl_groups() -> None:
        from vllm_ascend.ops.fused_moe.moe_comm_method import _MoECommMethods

        for comm_method in _MoECommMethods.values():
            dispatcher = getattr(comm_method, "token_dispatcher", None)
            refresh_fn = getattr(dispatcher, "refresh_hccl_group", None)
            if callable(refresh_fn):
                refresh_fn()

    @staticmethod
    def prepare_moe_hccl_teardown() -> bool:
        """Prepare MoE resources before their HCCL groups are destroyed."""
        from vllm_ascend.ops.fused_moe.moe_comm_method import _MoECommMethods

        try:
            for comm_method in _MoECommMethods.values():
                prepare_fn = getattr(comm_method, "prepare_hccl_teardown", None)
                if callable(prepare_fn):
                    prepare_fn()
            return True
        except Exception:
            logger.exception(
                "Failed to prepare MoE state for HCCL teardown; skipping HCCL cleanup for this sleep cycle."
            )
            return False

    @staticmethod
    def refresh_moe_hccl_runtime_state() -> None:
        """Refresh MoE runtime resources after HCCL groups are restored."""
        from vllm_ascend.ops.fused_moe.moe_comm_method import _MoECommMethods

        for comm_method in _MoECommMethods.values():
            refresh_fn = getattr(comm_method, "refresh_hccl_runtime_state", None)
            if callable(refresh_fn):
                refresh_fn()

    def sleep(self) -> None:
        if self._state is SleepHcclState.ASLEEP:
            logger.info("HCCL sleep ignored: already asleep.")
            return
        self._require_state(SleepHcclState.IDLE, "sleep")
        self._transition(SleepHcclState.SLEEPING)
        policy, allow_fallback = self._resolve_sleep_hccl_policy()
        self._active_policy = policy
        try:
            if policy == "skip":
                logger.info(
                    "HCCL sleep skipped (policy=skip, distributed_ready=%s).",
                    torch.distributed.is_available() and torch.distributed.is_initialized(),
                )
                self._transition(SleepHcclState.ASLEEP)
                return
            self._drain_and_sync()
            if policy in ("auto", "lightweight"):
                support = self._suspend_backend.available()
                support_vote = self._all_rank_vote(support, "capability")
                if support_vote is None:
                    raise RuntimeError("HCCL suspend capability vote failed (timeout/exception)")
                if not (support and support_vote):
                    if policy == "lightweight" and not allow_fallback:
                        raise RuntimeError(
                            "HCCL suspend/resume is unavailable on at least one rank "
                            "and sleep_allow_fallback is disabled"
                        )
                    self._suspend_backend.stats.fallbacks += 1
                    logger.info("HCCL suspend unavailable on at least one rank; using destroy/restore sleep path.")
                else:
                    local_ok = self._suspend_all_groups()
                    agreed = self._agree_on_suspend(local_ok)
                    if local_ok and agreed:
                        logger.info("HCCL sleep stats: %s", self._suspend_backend.stats.as_dict())
                        self._transition(SleepHcclState.ASLEEP)
                        return
                    if agreed is None:
                        raise RuntimeError(
                            "HCCL suspend vote failed (timeout/exception); refusing to mix suspend and destroy"
                        )
                    if local_ok:
                        self._rollback_suspend()
                    if not allow_fallback:
                        raise RuntimeError("HCCL lightweight suspend failed and sleep_allow_fallback is disabled")
                    self._suspend_backend.stats.fallbacks += 1
                    logger.warning("HCCL lightweight suspend failed; falling back to destroy/restore.")
            num_destroyed = self.destroy_hccl()
            self._destroyed_groups = True
            logger.info("Destroyed %d HCCL process groups for sleep mode (policy=%s).", num_destroyed, policy)
            self._transition(SleepHcclState.ASLEEP)
        except Exception:
            self._rollback_suspend()
            self._transition(SleepHcclState.FAILED)
            raise

    def wakeup(self) -> None:
        if self._state is SleepHcclState.IDLE:
            logger.info("HCCL wake-up ignored: not asleep.")
            return
        self._require_state(SleepHcclState.ASLEEP, "wake up")
        self._transition(SleepHcclState.WAKING)
        try:
            with set_current_vllm_config(self.vllm_config):
                if self._suspended_groups:
                    failed: list[Any] = []
                    for group, handles in self._suspended_groups:
                        try:
                            resumed = resume_group(getattr(group, "device_group", None), handles, self._suspend_backend)
                        except Exception:  # noqa: BLE001 - fallback below
                            logger.exception(
                                "HCCL resume raised for group '%s'.", getattr(group, "group_name", "anonymous")
                            )
                            resumed = False
                        if not resumed:
                            failed.append(group)
                    if failed:
                        if not self._resolve_sleep_hccl_policy()[1]:
                            raise RuntimeError(f"Failed to resume {len(failed)} suspended HCCL group(s)")
                        self._suspend_backend.stats.fallbacks += 1
                        self._force_destroy_and_restore()
                    self._suspended_groups = []
                    self._suspend_backend.stats.suspended_handles.clear()
                    logger.info("HCCL wake-up stats: %s", self._suspend_backend.stats.as_dict())
                elif self._destroyed_groups:
                    num_restored = self.restore_hccl()
                    self._destroyed_groups = False
                    logger.info("Restored %d HCCL process groups after sleep mode.", num_restored)
                self.refresh_moe_hccl_groups()
                self.refresh_moe_hccl_runtime_state()
            self._transition(SleepHcclState.IDLE)
        except Exception:
            self._transition(SleepHcclState.FAILED)
            raise
