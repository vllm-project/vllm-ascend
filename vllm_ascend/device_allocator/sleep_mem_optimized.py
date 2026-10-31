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
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import _groups
from vllm.logger import logger
from vllm.utils.mem_constants import GiB_bytes

from vllm_ascend.compilation import acl_graph
from vllm_ascend.device_allocator.hccl_suspend import (
    clear_suspended_handles,
    comm_to_int,
    get_backend_hccl_comm,
    hccl_suspend_available,
    resume_all_registered_pgs,
    resume_hccl_comm,
    resume_process_group_hccl,
    suspend_hccl_comm,
)


class SleepLifecycleState(str, Enum):
    IDLE = "idle"
    SLEEPING = "sleeping"
    ASLEEP = "asleep"
    WAKING = "waking"
    FAILED = "failed"


class HcclSleepPolicy(str, Enum):
    LIGHTWEIGHT = "lightweight"  # HcclCommSuspend / Resume
    DESTROY = "destroy"  # release PG + recreate (legacy)
    SKIP = "skip"


class AclGraphSleepPolicy(str, Enum):
    INPLACE = "inplace"  # keep graph entries; free workspaces only
    WORKSPACE_ONLY = "workspace_only"  # free workspaces; keep params/handles
    RECAPTURE = "recapture"  # invalidate + capture_model (legacy)
    SKIP = "skip"


@dataclass
class SleepPolicyConfig:
    hccl: HcclSleepPolicy = HcclSleepPolicy.LIGHTWEIGHT
    acl_graph: AclGraphSleepPolicy = AclGraphSleepPolicy.INPLACE
    allow_fallback: bool = True


def resolve_sleep_policy(ascend_config: Any | None = None) -> SleepPolicyConfig:
    """Read sleep policies from AscendConfig / additional_config with safe defaults."""
    if ascend_config is None:
        try:
            from vllm_ascend.ascend_config import get_ascend_config

            ascend_config = get_ascend_config()
        except Exception:  # noqa: BLE001
            return SleepPolicyConfig()

    hccl_raw = getattr(ascend_config, "sleep_hccl_policy", None) or "lightweight"
    acl_raw = getattr(ascend_config, "sleep_acl_graph_policy", None) or "inplace"
    allow_fallback = bool(getattr(ascend_config, "sleep_allow_fallback", True))
    try:
        hccl = HcclSleepPolicy(str(hccl_raw).lower())
    except ValueError:
        logger.warning("Unknown sleep_hccl_policy=%s, fallback to lightweight", hccl_raw)
        hccl = HcclSleepPolicy.LIGHTWEIGHT
    try:
        acl_graph_policy = AclGraphSleepPolicy(str(acl_raw).lower())
    except ValueError:
        logger.warning("Unknown sleep_acl_graph_policy=%s, fallback to inplace", acl_raw)
        acl_graph_policy = AclGraphSleepPolicy.INPLACE
    return SleepPolicyConfig(hccl=hccl, acl_graph=acl_graph_policy, allow_fallback=allow_fallback)


class SleepLifecycleStateMachine:
    """Single owner of sleep/wake transitions with illegal-transition guards."""

    _ALLOWED = {
        SleepLifecycleState.IDLE: {SleepLifecycleState.SLEEPING},
        SleepLifecycleState.SLEEPING: {
            SleepLifecycleState.ASLEEP,
            SleepLifecycleState.FAILED,
            SleepLifecycleState.IDLE,  # rollback
        },
        SleepLifecycleState.ASLEEP: {SleepLifecycleState.WAKING},
        SleepLifecycleState.WAKING: {
            SleepLifecycleState.IDLE,
            SleepLifecycleState.FAILED,
            SleepLifecycleState.ASLEEP,  # rollback to asleep then cold rebuild
        },
        SleepLifecycleState.FAILED: {
            SleepLifecycleState.IDLE,
            SleepLifecycleState.SLEEPING,
            SleepLifecycleState.WAKING,
        },
    }

    def __init__(self) -> None:
        self.state = SleepLifecycleState.IDLE
        self.last_error: str | None = None
        self.completed_phases: list[str] = []

    def transition(self, new_state: SleepLifecycleState) -> None:
        allowed = self._ALLOWED.get(self.state, set())
        if new_state not in allowed:
            raise RuntimeError(f"Illegal sleep lifecycle transition: {self.state} -> {new_state}")
        self.state = new_state

    def mark_phase(self, name: str) -> None:
        self.completed_phases.append(name)

    def reset_phases(self) -> None:
        self.completed_phases = []
        self.last_error = None


class SleepWakeupManager:
    def __init__(self, vllm_config: VllmConfig, worker: Any, model_runner_getter: Callable[[], Any]):
        self.vllm_config = vllm_config
        self.policy = resolve_sleep_policy()
        self.lifecycle = SleepLifecycleStateMachine()
        self.acl_graph = AclGraphSleepWakeupManager(vllm_config, model_runner_getter, self.policy)
        self.hccl = HcclSleepWakeupManager(vllm_config, worker, self.policy)
        self._model_runner_getter = model_runner_getter

    def sleep(self) -> None:
        self.policy = resolve_sleep_policy()
        self.acl_graph.policy = self.policy
        self.hccl.policy = self.policy
        self.lifecycle.reset_phases()
        self.lifecycle.transition(SleepLifecycleState.SLEEPING)
        model_runner = self._model_runner_getter()
        free_bytes_before_cleanup = torch.npu.mem_get_info()[0]
        try:
            # Ownership order (sleep): ACL Graph/workspace -> HCCL -> (weights/KV by allocator)
            if model_runner is not None and getattr(model_runner, "use_aclgraph", False):
                used = self.acl_graph.sleep()
                self.lifecycle.mark_phase(f"acl_graph:{used}")
            used_hccl = self.hccl.sleep()
            self.lifecycle.mark_phase(f"hccl:{used_hccl}")
            self.lifecycle.transition(SleepLifecycleState.ASLEEP)
        except Exception as exc:  # noqa: BLE001
            self.lifecycle.last_error = str(exc)
            logger.exception("Sleep cleanup failed; attempting rollback to servable state")
            self._rollback_sleep()
            self.lifecycle.transition(SleepLifecycleState.FAILED)
            self.lifecycle.transition(SleepLifecycleState.IDLE)
            raise
        free_bytes_after_cleanup = torch.npu.mem_get_info()[0]
        free_mem = free_bytes_after_cleanup - free_bytes_before_cleanup
        logger.info(
            "Sleep mode released HCCL/ACL resources (policy hccl=%s acl=%s): %.3f GiB.",
            self.policy.hccl.value,
            self.policy.acl_graph.value,
            free_mem / GiB_bytes,
        )

    def wakeup(self, tags: list[str] | None = None) -> None:
        self.policy = resolve_sleep_policy()
        self.acl_graph.policy = self.policy
        self.hccl.policy = self.policy
        if self.lifecycle.state == SleepLifecycleState.IDLE:
            # Idempotent wake: nothing to do.
            return
        self.lifecycle.transition(SleepLifecycleState.WAKING)
        try:
            # Ownership order (wake): HCCL -> ACL Graph -> (weights/KV by allocator)
            used_hccl = self.hccl.wakeup()
            self.lifecycle.mark_phase(f"hccl_wake:{used_hccl}")
            model_runner = self._model_runner_getter()
            if model_runner is not None and getattr(model_runner, "use_aclgraph", False):
                used = self.acl_graph.wakeup(tags)
                self.lifecycle.mark_phase(f"acl_graph_wake:{used}")
            self.lifecycle.transition(SleepLifecycleState.IDLE)
        except Exception as exc:  # noqa: BLE001
            self.lifecycle.last_error = str(exc)
            logger.exception("Wakeup failed; attempting cold rebuild fallback")
            self._cold_rebuild(tags)
            self.lifecycle.transition(SleepLifecycleState.FAILED)
            self.lifecycle.transition(SleepLifecycleState.IDLE)
            raise

    def _rollback_sleep(self) -> None:
        """Best-effort undo of partial sleep so the worker stays servable."""
        try:
            self.hccl.wakeup()
        except Exception:  # noqa: BLE001
            logger.exception("HCCL rollback during sleep failed")
        try:
            self.acl_graph.wakeup(tags=None)
        except Exception:  # noqa: BLE001
            logger.exception("ACL graph rollback during sleep failed")

    def _cold_rebuild(self, tags: list[str] | None) -> None:
        """Force destroy/recapture path after a failed lightweight wake."""
        try:
            self.hccl.force_destroy_and_restore()
        except Exception:  # noqa: BLE001
            logger.exception("HCCL cold rebuild failed")
        try:
            self.acl_graph.force_recapture(tags)
        except Exception:  # noqa: BLE001
            logger.exception("ACL graph cold rebuild failed")


class AclGraphSleepWakeupManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        model_runner_getter: Callable[[], Any],
        policy: SleepPolicyConfig | None = None,
    ):
        self.vllm_config = vllm_config
        self._model_runner_getter = model_runner_getter
        self.policy = policy or SleepPolicyConfig()
        self._workspace_snapshot: dict[str, dict[int, Any]] = {}
        self._inplace_kept = False

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

    def _snapshot_workspace_keys(self) -> None:
        self._workspace_snapshot = {}
        for name, params in (
            ("main", acl_graph._graph_params),
            ("draft", acl_graph._draft_graph_params),
            ("draft_prefill", acl_graph._draft_graph_prefill_params),
        ):
            if params is None or not hasattr(params, "workspaces"):
                continue
            self._workspace_snapshot[name] = {k: None for k in list(params.workspaces.keys())}

    def sleep(self) -> str:
        mode = self.policy.acl_graph
        if mode == AclGraphSleepPolicy.SKIP:
            return "skip"
        if mode in (AclGraphSleepPolicy.INPLACE, AclGraphSleepPolicy.WORKSPACE_ONLY):
            self._snapshot_workspace_keys()
            self.clear_all_attention_workspaces()
            self._inplace_kept = True
            # Keep concrete_aclgraph_entries / events for in-place restore.
            if mode == AclGraphSleepPolicy.WORKSPACE_ONLY:
                return "workspace_only"
            return "inplace"
        # Legacy recapture path.
        self.clear_all_attention_workspaces()
        self.reset_all_graph_params()
        self.reset_model_runner_graph_manager(self._model_runner_getter())
        self._inplace_kept = False
        return "recapture"

    def wakeup(self, tags: list[str] | None = None) -> str:
        if tags is not None and "kv_cache" not in tags:
            # Level-2 wakeup restores weights before external weight loading;
            # restore/recapture graphs only after KV cache is restored.
            return "deferred"
        mode = self.policy.acl_graph
        if mode == AclGraphSleepPolicy.SKIP:
            return "skip"
        model_runner = self._model_runner_getter()
        if mode in (AclGraphSleepPolicy.INPLACE, AclGraphSleepPolicy.WORKSPACE_ONLY) and self._inplace_kept:
            # In-place: keep captured graphs; workspaces are lazily rebuilt on
            # next capture/update path. If graph entries were lost, fallback.
            wrappers = list(acl_graph._acl_graph_wrappers)
            has_entries = any(getattr(w, "concrete_aclgraph_entries", None) for w in wrappers)
            if has_entries or mode == AclGraphSleepPolicy.WORKSPACE_ONLY:
                self._inplace_kept = False
                return mode.value
            if self.policy.allow_fallback:
                logger.warning("In-place ACL graph entries missing; falling back to recapture")
            else:
                raise RuntimeError("In-place ACL graph restore failed and fallback disabled")
        with set_current_vllm_config(self.vllm_config):
            model_runner.capture_model()
        self._inplace_kept = False
        return "recapture"

    def force_recapture(self, tags: list[str] | None = None) -> None:
        self._inplace_kept = False
        self.clear_all_attention_workspaces()
        self.reset_all_graph_params()
        self.reset_model_runner_graph_manager(self._model_runner_getter())
        if tags is not None and "kv_cache" not in tags:
            return
        model_runner = self._model_runner_getter()
        with set_current_vllm_config(self.vllm_config):
            model_runner.capture_model()


class HcclSleepWakeupManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        worker: Any,
        policy: SleepPolicyConfig | None = None,
    ):
        self.vllm_config = vllm_config
        self.worker = worker
        self.policy = policy or SleepPolicyConfig()
        self._suspended_comms: list[Any] = []
        self._used_lightweight = False
        self._multi_rank_destroyed = False

    @staticmethod
    def iter_alive_group_coordinators():
        seen: set[int] = set()
        for group_ref in list(_groups.values()):
            group = group_ref()
            if group is None or id(group) in seen:
                continue
            seen.add(id(group))
            yield group

    @classmethod
    def destroy_hccl(cls) -> int:
        num_destroyed = 0
        for group in cls.iter_alive_group_coordinators():
            if group.destroy_hccl():
                num_destroyed += 1
        return num_destroyed

    @classmethod
    def restore_hccl(cls) -> int:
        num_restored = 0
        for group in cls.iter_alive_group_coordinators():
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

    def _drain_pp_and_sync(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for handle in getattr(self.worker, "_pp_send_work", []):
                handle.wait()
            self.worker._pp_send_work = []
            torch.npu.synchronize()

    def _iter_groups_stable(self):
        groups = list(self.iter_alive_group_coordinators())
        try:
            from vllm_ascend.device_allocator.hccl_suspend import group_sort_key

            groups.sort(key=group_sort_key)
        except Exception:  # noqa: BLE001
            groups.sort(key=lambda g: str(getattr(g, "unique_name", id(g))))
        return groups

    @staticmethod
    def _group_world_size(group) -> int:
        device_group = getattr(group, "device_group", None)
        if device_group is None:
            return int(getattr(group, "world_size", 1) or 1)
        world = getattr(device_group, "size", None)
        try:
            world = world() if callable(world) else world
            return int(world or 1)
        except Exception:  # noqa: BLE001
            return int(getattr(group, "world_size", 1) or 1)

    def _has_multi_rank_group(self) -> bool:
        return any(self._group_world_size(g) > 1 for g in self._iter_groups_stable())

    @staticmethod
    def _safe_barrier(device_group) -> None:
        if device_group is None:
            return
        try:
            import torch.distributed as dist

            world = getattr(device_group, "size", 1)
            world = world() if callable(world) else world
            if int(world) <= 1:
                return
            dist.barrier(group=device_group)
        except Exception as exc:  # noqa: BLE001
            logger.warning("HCCL sleep barrier skipped: %s", exc)

    def _global_barrier(self) -> None:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Global HCCL barrier skipped: %s", exc)

    def _suspend_all_comms(self) -> int:
        if not hccl_suspend_available():
            return 0
        suspended = 0
        self._suspended_comms = []
        groups = self._iter_groups_stable()
        # Barrier only before any suspend: after HcclCommSuspend the PG can no
        # longer join collectives, so post-suspend barriers hang.
        self._global_barrier()
        seen_handles: set[int] = set()
        for group in groups:
            device_group = getattr(group, "device_group", None)
            self._safe_barrier(device_group)
            comm = get_backend_hccl_comm(device_group)
            if comm is None:
                logger.warning(
                    "No HCCL handle for group %s on this rank; skip suspend",
                    getattr(group, "unique_name", group),
                )
                continue
            handle = comm_to_int(comm)
            if handle in seen_handles:
                continue
            if suspend_hccl_comm(comm):
                self._suspended_comms.append(comm)
                seen_handles.add(handle)
                suspended += 1
        return suspended

    def _resume_all_comms(self) -> int:
        """Resume via ProcessGroupHCCL.resume_hccl_comm (preferred) then raw.

        Multi-NPU TP generate hangs if only raw HcclCommResume is used, because
        torch_npu ProcessGroup state is not rebound. Prefer the official
        reinit_process_group(rebuild_link=False) / resume_hccl_comm path.

        Barriers must run only after resume succeeds: suspended communicators
        cannot participate in collectives.
        """
        groups = self._iter_groups_stable()

        resumed = resume_all_registered_pgs()
        if resumed > 0:
            for group in groups:
                device_group = getattr(group, "device_group", None)
                resume_process_group_hccl(device_group)
                self._safe_barrier(device_group)
            self._global_barrier()
            clear_suspended_handles()
            self._suspended_comms = []
            return max(resumed, len(groups))

        resumed = 0
        pending_by_handle = {comm_to_int(comm): comm for comm in self._suspended_comms}
        for group in groups:
            device_group = getattr(group, "device_group", None)
            if resume_process_group_hccl(device_group):
                resumed += 1
                handle = get_backend_hccl_comm(device_group)
                if handle is not None:
                    pending_by_handle.pop(comm_to_int(handle), None)
                self._safe_barrier(device_group)
                continue
            comm = get_backend_hccl_comm(device_group)
            target = None
            if comm is not None:
                target = pending_by_handle.pop(comm_to_int(comm), None)
            if (target is not None and resume_hccl_comm(target)) or (
                comm is not None and resume_hccl_comm(comm)
            ):
                resumed += 1
            self._safe_barrier(device_group)
        for comm in list(pending_by_handle.values()):
            if resume_hccl_comm(comm):
                resumed += 1
        self._global_barrier()
        clear_suspended_handles()
        self._suspended_comms = []
        return resumed

    def _verify_tp_collectives(self) -> bool:
        try:
            import torch
            import torch.distributed as dist

            ok = True
            for group in self._iter_groups_stable():
                if self._group_world_size(group) <= 1:
                    continue
                device_group = getattr(group, "device_group", None)
                if device_group is None:
                    continue
                self._safe_barrier(device_group)
                t = torch.ones(8, device="npu", dtype=torch.float32)
                dist.all_reduce(t, group=device_group)
                torch.npu.synchronize()
                expected = float(self._group_world_size(group))
                got = float(t[0].item())
                if abs(got - expected) > 1e-3:
                    logger.warning(
                        "TP collective verify mismatch group=%s got=%s expected=%s",
                        getattr(group, "unique_name", group),
                        got,
                        expected,
                    )
                    ok = False
                self._safe_barrier(device_group)
            return ok
        except Exception as exc:  # noqa: BLE001
            logger.warning("TP collective verify failed: %s", exc)
            return False

    def sleep(self) -> str:
        mode = self.policy.hccl
        if mode == HcclSleepPolicy.SKIP:
            return "skip"
        self._drain_pp_and_sync()
        self._multi_rank_destroyed = False
        if mode == HcclSleepPolicy.LIGHTWEIGHT:
            # Single-rank: true HcclCommSuspend + ProcessGroup resume_hccl_comm.
            # Multi-rank: Suspend+Resume leaves TP collectives stale on current
            # CANN/torch_npu; Suspend then destroy/restore can hang generate.
            # So multi-rank lightweight uses destroy/restore (#9585/#3172) while
            # still exposing the same sleep_hccl_policy=lightweight knob.
            if self._has_multi_rank_group():
                destroyed = self.destroy_hccl()
                self._used_lightweight = True
                self._multi_rank_destroyed = True
                logger.info(
                    "Lightweight multi-NPU sleep via destroy/restore "
                    "(destroyed_pgs=%d; Suspend reserved for single-rank)",
                    destroyed,
                )
                return "lightweight"
            num = self._suspend_all_comms()
            if num > 0:
                self._used_lightweight = True
                logger.info(
                    "Suspended %d HCCL communicators (lightweight, multi=False).",
                    num,
                )
                return "lightweight"
            if self.policy.allow_fallback:
                logger.warning("HCCL lightweight suspend unavailable; falling back to destroy")
                destroyed = self.destroy_hccl()
                self._used_lightweight = True
                self._multi_rank_destroyed = True
                logger.info(
                    "Lightweight fallback destroy/restore (destroyed_pgs=%d)",
                    destroyed,
                )
                return "lightweight"
            raise RuntimeError("HCCL lightweight suspend failed and fallback disabled")
        num_destroyed = self.destroy_hccl()
        self._used_lightweight = False
        if num_destroyed > 0:
            logger.info("Destroyed %d HCCL process groups for sleep mode.", num_destroyed)
        return "destroy"

    def wakeup(self) -> str:
        mode = self.policy.hccl
        if mode == HcclSleepPolicy.SKIP:
            return "skip"
        with set_current_vllm_config(self.vllm_config):
            if self._used_lightweight or mode == HcclSleepPolicy.LIGHTWEIGHT:
                if self._multi_rank_destroyed:
                    num_restored = self.restore_hccl()
                    self.refresh_moe_hccl_groups()
                    self._multi_rank_destroyed = False
                    self._used_lightweight = False
                    self._suspended_comms = []
                    clear_suspended_handles()
                    logger.info(
                        "Restored %d HCCL process groups after lightweight destroy fallback.",
                        num_restored,
                    )
                    return "lightweight"

                num = self._resume_all_comms()
                self.refresh_moe_hccl_groups()
                ok = self._verify_tp_collectives()
                if not ok and self.policy.allow_fallback:
                    logger.warning("Post-resume TP verify failed; falling back to destroy/restore")
                    self.destroy_hccl()
                    num_restored = self.restore_hccl()
                    self.refresh_moe_hccl_groups()
                    self._multi_rank_destroyed = False
                    self._used_lightweight = False
                    clear_suspended_handles()
                    logger.info(
                        "Fallback restored %d HCCL process groups after failed Suspend/Resume.",
                        num_restored,
                    )
                    return "lightweight"

                self._used_lightweight = False
                logger.info(
                    "Resumed %d HCCL communicators after sleep mode (verify_ok=%s).",
                    num,
                    ok,
                )
                return "lightweight"
            num_restored = self.restore_hccl()
            self.refresh_moe_hccl_groups()
        logger.info("Restored %d HCCL process groups after sleep mode.", num_restored)
        return "destroy"

    def force_destroy_and_restore(self) -> None:
        self._used_lightweight = False
        self._suspended_comms = []
        self._multi_rank_destroyed = False
        clear_suspended_handles()
        self._drain_pp_and_sync()
        self.destroy_hccl()
        with set_current_vllm_config(self.vllm_config):
            self.restore_hccl()
            self.refresh_moe_hccl_groups()
