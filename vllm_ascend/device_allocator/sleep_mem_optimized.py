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
from enum import Enum
from typing import Any

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    _groups,
    get_world_group,
    init_model_parallel_group,
)
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
from vllm_ascend.utils import SLEEP_LIFECYCLE_ANCHOR_GROUP_NAME


class HcclSleepPolicy(str, Enum):
    LIGHTWEIGHT = "lightweight"
    DESTROY = "destroy"
    SKIP = "skip"


class AclGraphSleepPolicy(str, Enum):
    INPLACE = "inplace"
    WORKSPACE_ONLY = "workspace_only"
    RECAPTURE = "recapture"
    SKIP = "skip"


def _policy_from_rl(vllm_config: VllmConfig) -> tuple[HcclSleepPolicy, AclGraphSleepPolicy, bool]:
    """Read sleep policies from Ascend RlConfig when available."""
    hccl = HcclSleepPolicy.DESTROY
    acl = AclGraphSleepPolicy.RECAPTURE
    allow_fallback = True
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        rl = getattr(get_ascend_config(), "rl_config", None)
        if rl is not None:
            hccl = HcclSleepPolicy(str(getattr(rl, "sleep_hccl_policy", "destroy")))
            acl = AclGraphSleepPolicy(str(getattr(rl, "sleep_acl_graph_policy", "recapture")))
            allow_fallback = bool(getattr(rl, "sleep_allow_fallback", True))
    except Exception:
        pass
    return hccl, acl, allow_fallback


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
        _, self.acl_policy, self.allow_fallback = _policy_from_rl(vllm_config)
        self._used_inplace = False

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
        if self.acl_policy == AclGraphSleepPolicy.SKIP:
            return
        self.clear_all_attention_workspaces()
        if self.acl_policy in (AclGraphSleepPolicy.INPLACE, AclGraphSleepPolicy.WORKSPACE_ONLY):
            # Keep captured entries; only free workspaces for HBM reclaim.
            self._used_inplace = True
            return
        self.reset_all_graph_params()
        self.reset_model_runner_graph_manager(self._model_runner_getter())
        self._used_inplace = False

    def wakeup(self, tags: list[str] | None = None) -> None:
        if tags is not None and "kv_cache" not in tags:
            # Level-2 wakeup restores weights before external weight loading;
            # recapture graphs only after KV cache is restored.
            return
        if self.acl_policy == AclGraphSleepPolicy.SKIP:
            return
        if self._used_inplace and self.acl_policy in (
            AclGraphSleepPolicy.INPLACE,
            AclGraphSleepPolicy.WORKSPACE_ONLY,
        ):
            has_captured = False
            if acl_graph._acl_graph_wrappers:
                has_captured = any(
                    len(getattr(w, "concrete_aclgraph_entries", {}) or {}) > 0 for w in acl_graph._acl_graph_wrappers
                )
            if has_captured or not self.allow_fallback:
                # Lazy workspace rebuild happens on first replay; no capture_model.
                self._used_inplace = False
                return
            logger.warning("In-place ACL Graph entries missing; falling back to recapture.")
            self._used_inplace = False
        model_runner = self._model_runner_getter()
        with set_current_vllm_config(self.vllm_config):
            model_runner.capture_model()


class HcclSleepWakeupManager:
    def __init__(self, vllm_config: VllmConfig, worker: Any):
        self.vllm_config = vllm_config
        self.worker = worker
        self._lifecycle_anchor_group: Any | None = None
        self._skip_hccl_cleanup_for_cycle = False
        self.hccl_policy, _, self.allow_fallback = _policy_from_rl(vllm_config)
        self._used_lightweight = False
        self._suspended_comms: list[Any] = []
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

    def _group_world_size(self, group: Any) -> int:
        device_group = getattr(group, "device_group", None)
        if device_group is None:
            return 1
        size = getattr(device_group, "size", 1)
        return int(size() if callable(size) else size)

    def _has_multi_rank_group(self) -> bool:
        return any(self._group_world_size(g) > 1 for g in self.iter_alive_group_coordinators())

    def _safe_barrier(self, device_group: Any) -> None:
        if device_group is None:
            return
        try:
            world = getattr(device_group, "size", 1)
            world = world() if callable(world) else world
            if int(world) <= 1:
                return
            torch.distributed.barrier(group=device_group)
        except Exception as exc:  # noqa: BLE001
            logger.warning("HCCL sleep barrier skipped: %s", exc)

    def _global_barrier(self) -> None:
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Global HCCL barrier skipped: %s", exc)

    def _suspend_all_comms(self) -> int:
        if not hccl_suspend_available():
            return 0
        suspended = 0
        self._suspended_comms = []
        groups = list(self.iter_alive_group_coordinators())
        # Barrier only before suspend; suspended HCCL cannot join collectives.
        self._global_barrier()
        seen_handles: set[int] = set()
        for group in groups:
            if group is self._lifecycle_anchor_group:
                continue
            device_group = getattr(group, "device_group", None)
            self._safe_barrier(device_group)
            comm = get_backend_hccl_comm(device_group)
            if comm is None:
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
        groups = list(self.iter_alive_group_coordinators())
        resumed = resume_all_registered_pgs()
        if resumed > 0:
            for group in groups:
                if group is self._lifecycle_anchor_group:
                    continue
                device_group = getattr(group, "device_group", None)
                resume_process_group_hccl(device_group)
                self._safe_barrier(device_group)
            self._global_barrier()
            clear_suspended_handles()
            self._suspended_comms = []
            return max(resumed, 1)

        resumed = 0
        pending_by_handle = {comm_to_int(comm): comm for comm in self._suspended_comms}
        for group in groups:
            if group is self._lifecycle_anchor_group:
                continue
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
            if (target is not None and resume_hccl_comm(target)) or (comm is not None and resume_hccl_comm(comm)):
                resumed += 1
            self._safe_barrier(device_group)
        for comm in list(pending_by_handle.values()):
            if resume_hccl_comm(comm):
                resumed += 1
        self._global_barrier()
        clear_suspended_handles()
        self._suspended_comms = []
        return resumed

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
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        for handle in getattr(self.worker, "_pp_send_work", []):
            handle.wait()
        self.worker._pp_send_work = []
        torch.npu.synchronize()
        self._used_lightweight = False
        self._multi_rank_destroyed = False

        if self.hccl_policy == HcclSleepPolicy.SKIP:
            return

        if self.hccl_policy == HcclSleepPolicy.LIGHTWEIGHT:
            # Single-rank: HcclCommSuspend. Multi-rank: destroy/restore under
            # the same lightweight knob (Suspend+Resume currently unreliable).
            if self._has_multi_rank_group():
                destroyed = self.destroy_hccl()
                self._used_lightweight = True
                self._multi_rank_destroyed = True
                logger.info(
                    "Lightweight multi-NPU sleep via destroy/restore (destroyed_pgs=%d).",
                    destroyed,
                )
                return
            num = self._suspend_all_comms()
            if num > 0:
                self._used_lightweight = True
                logger.info("Suspended %d HCCL communicators (lightweight).", num)
                return
            if self.allow_fallback:
                logger.warning("HCCL lightweight suspend unavailable; falling back to destroy")
            else:
                return

        num_destroyed = self.destroy_hccl()
        if num_destroyed > 0:
            logger.info("Destroyed %d HCCL process groups for sleep mode.", num_destroyed)

    def wakeup(self) -> None:
        with set_current_vllm_config(self.vllm_config):
            if self._used_lightweight and not self._multi_rank_destroyed:
                num = self._resume_all_comms()
                self.refresh_moe_hccl_groups()
                self.refresh_moe_hccl_runtime_state()
                logger.info("Resumed %d HCCL communicators after lightweight sleep.", num)
                self._used_lightweight = False
                return
            num_restored = self.restore_hccl()
            self.refresh_moe_hccl_groups()
            self.refresh_moe_hccl_runtime_state()
        logger.info("Restored %d HCCL process groups after sleep mode.", num_restored)
        self._used_lightweight = False
        self._multi_rank_destroyed = False
