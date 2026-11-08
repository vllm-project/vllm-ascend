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
"""Ascend sleep-mode backends (RFC #34303).

``--sleep-mode-backend`` keeps upstream's default name ``cumem``: it means
"this platform's built-in allocator-based backend", and no new user-facing
name is added here. Upstream's own ``CuMemBackend`` finds its allocator with
``get_mem_allocator_instance()``, which only resolves on CUDA and XPU, so on
NPU that name selects the backends in this module, which drive
``CaMemAllocator`` instead. ``NPUWorker`` resolves it; the upstream registry
is left alone, so any other name is a third-party backend and still comes
from ``SleepModeBackendFactory``.

An Ascend sleep releases more than allocator memory: Mooncake buffer
registrations, HCCL process groups, ACL graphs and the host copies a level-2
sleep keeps of the model. In ``CaMemSleepBackend`` each of those is a
``SleepPhase``. The backend owns the order the phases run in and what happens
when one of them fails; the worker only contributes the phases that need
worker or model-runner context.

``LegacyCaMemSleepBackend`` handles device memory alone and leaves the rest
of the sequence to the worker, as it was before ``CaMemSleepBackend``. The
``additional_config`` key ``rl_config.sleep_mode_legacy`` selects it.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass

import torch
from vllm.device_allocator.sleep_mode_backend import SleepModeBackend
from vllm.logger import logger
from vllm.platforms import current_platform

#: Upstream's default ``--sleep-mode-backend``. On NPU it selects the backends
#: in this module rather than upstream's ``CuMemBackend``.
UPSTREAM_DEFAULT_SLEEP_MODE_BACKEND = "cumem"

# The tags one full sleep puts to sleep. Mirrors the set vLLM's executor tracks.
SLEEP_TAGS = frozenset(("weights", "kv_cache"))


@dataclass(frozen=True)
class SuspendRequest:
    """What one ``suspend`` call asks of every phase."""

    level: int = 1
    """1 keeps the weights on the host and restores them on resume. 2 drops
    them: the caller reloads the weights after waking."""


def suspend_level(discardable: Collection[int] | int) -> int:
    """Sleep level of a ``SleepModeBackend.suspend`` argument.

    ``NPUWorker`` passes the level. vLLM after v0.33.0 defines the argument as
    the storage addresses the caller rewrites after waking up (vLLM #66010).
    ``CaMemAllocator`` releases a whole pool at a time, so any discardable
    storage means the weights pool is dropped, which is level 2.
    """
    if isinstance(discardable, int):
        return discardable
    return 2 if discardable else 1


def _ignore_request(request: SuspendRequest) -> None:
    pass


def _ignore_tags(tags: list[str] | None) -> None:
    pass


@dataclass(frozen=True)
class SleepPhase:
    """One group of resources that sleep releases and wake-up restores.

    ``prepare`` runs for every phase before any phase suspends. It rejects a
    request the phase cannot honour and copies out what ``suspend`` is about
    to destroy, but releases nothing, so a failure there needs no rollback.

    ``suspend`` runs in registration order and ``resume`` in reverse.
    ``resume`` also rolls back a suspend that failed part-way, and vLLM calls
    it once per tag when it wakes in stages, so it must cope with resources
    that are not suspended.
    """

    name: str
    prepare: Callable[[SuspendRequest], None] = _ignore_request
    suspend: Callable[[SuspendRequest], None] = _ignore_request
    resume: Callable[[list[str] | None], None] = _ignore_tags


def _suspend_device_memory(request: SuspendRequest) -> None:
    # Imported here on purpose: ``camem`` pulls in ``acl.rt``, and upstream
    # requires that resolving or constructing a backend never touches the device.
    from vllm_ascend.device_allocator.camem import CaMemAllocator
    from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te

    # Earlier phases queue device work such as HCCL teardown and parameter
    # copies. Let it finish before the memory it uses is unmapped.
    torch.npu.synchronize()
    # Mooncake hands raw KV-cache device addresses to the NIC for one-sided
    # reads, so a remote peer can read this memory without the local CPU
    # taking part. Unregister before the allocator releases it, otherwise
    # those reads land on freed - possibly reallocated - device memory.
    global_te.unregister_buffer()
    allocator = CaMemAllocator.get_instance()
    allocator.sleep(offload_tags=("weights",) if request.level == 1 else tuple())


def _resume_device_memory(tags: list[str] | None) -> None:
    from vllm_ascend.device_allocator.camem import CaMemAllocator
    from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te

    allocator = CaMemAllocator.get_instance()
    allocator.wake_up(tags=tags)
    # Re-register only once the KV-cache pool is actually back: the
    # recorded pointers describe that pool and nothing else.
    if tags is None or "kv_cache" in tags:
        global_te.reregister_buffer()


DEVICE_MEMORY_PHASE = SleepPhase(
    name="device_memory",
    suspend=_suspend_device_memory,
    resume=_resume_device_memory,
)


class CaMemSleepBackend(SleepModeBackend):
    """What ``cumem`` means on NPU, run as an ordered list of phases.

    Device memory is always the last phase to suspend and the first to resume.
    Phases added with ``add_phase`` run ahead of it, in the order they were
    added.

    Selective discard (``release_kv_cache_memory``) is not wired up on Ascend,
    so ``discard`` keeps upstream's "not supported" error. It was already
    unreachable here before this backend existed, because ``NPUWorker`` had no
    sleep-mode backend to forward it to.
    """

    def __init__(self) -> None:
        super().__init__()
        self._resource_phases: list[SleepPhase] = []
        self._asleep_tags: set[str] = set()
        self._closed = False

    @property
    def phases(self) -> tuple[SleepPhase, ...]:
        """Every phase, in suspend order."""
        return (*self._resource_phases, DEVICE_MEMORY_PHASE)

    def add_phase(self, phase: SleepPhase) -> None:
        """Register a resource this backend cannot reach on its own.

        The factory builds a backend from the model config alone, so whoever
        owns worker or model-runner state hands it over here, before the
        first suspend.
        """
        self._check_open()
        if self._asleep_tags:
            raise RuntimeError(f"Cannot add sleep phase {phase.name!r} while the backend is suspended.")
        if any(existing.name == phase.name for existing in self.phases):
            raise ValueError(f"Sleep phase {phase.name!r} is already registered.")
        self._resource_phases.append(phase)

    def suspend(self, discardable: Collection[int] | int = ()) -> None:
        self._check_open()
        level = suspend_level(discardable)
        # Same rule as vLLM's executor: once the weights are asleep a second
        # sleep has nothing to release, and its phases would read freed memory.
        if "weights" in self._asleep_tags:
            logger.warning("Ascend sleep-mode backend is already suspended; ignoring suspend(level=%s).", level)
            return

        request = SuspendRequest(level=level)
        phases = self.phases
        for phase in phases:
            phase.prepare(request)

        # After a staged wake-up some tags are still asleep. A rollback wakes
        # only what this call put to sleep, so those stay as they were.
        rollback_tags = sorted(SLEEP_TAGS - self._asleep_tags) if self._asleep_tags else None
        entered: list[SleepPhase] = []
        try:
            for phase in phases:
                # Recorded before it runs: a phase that fails half-way is
                # rolled back as well.
                entered.append(phase)
                phase.suspend(request)
        except Exception:
            self._roll_back(entered, rollback_tags)
            raise
        self._asleep_tags.update(SLEEP_TAGS)
        self._state = "SUSPENDED"

    def _roll_back(self, entered: list[SleepPhase], tags: list[str] | None) -> None:
        logger.warning(
            "Sleep phase '%s' failed to suspend; resuming %d phase(s) to undo the sleep.",
            entered[-1].name,
            len(entered),
        )
        for phase in reversed(entered):
            try:
                phase.resume(tags)
            except Exception:
                # Keep undoing the other phases; the caller gets the error
                # that started the rollback.
                logger.exception("Sleep phase '%s' failed to roll back and may still be suspended.", phase.name)

    def resume(self, tags: list[str] | None = None) -> None:
        self._check_open()
        # Not skipped when nothing is recorded asleep: if that record is ever
        # wrong, a wake-up that does nothing would strand device memory.
        self._state = "RESUMING"
        for phase in reversed(self.phases):
            phase.resume(tags)
        if tags is None:
            self._asleep_tags.clear()
        else:
            self._asleep_tags.difference_update(tags)
        self._state = "SUSPENDED" if self._asleep_tags else "RUNNING"

    def close(self) -> None:
        """Drop the registered phases; later suspend or resume calls fail.

        Phases hold references back into the worker that added them. Safe to
        call more than once.
        """
        self._closed = True
        self._resource_phases.clear()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Ascend sleep-mode backend is closed.")

    @classmethod
    def is_supported(cls) -> bool:
        return current_platform.device_type == "npu"

    @classmethod
    def preserves_communicators(cls) -> bool:
        # Ascend sleep destroys and recreates HCCL process groups when RL
        # extra cleanup is enabled. A capability flag cannot see per-instance
        # config, so report the conservative answer and let callers assume the
        # groups always need re-initialization after a wake-up.
        return False


class LegacyCaMemSleepBackend(SleepModeBackend):
    """Device memory only; the worker runs the rest of the sleep around it.

    This is the sequence Ascend used before ``CaMemSleepBackend``: no phases,
    no rollback and no check for a repeated sleep. It is the fallback for the
    migration and runs under the same ``cumem`` name when
    ``rl_config.sleep_mode_legacy`` is set. It goes away, together with that
    key, once the migration is complete.
    """

    def suspend(self, discardable: Collection[int] | int = ()) -> None:
        self._state = "SUSPENDED"
        _suspend_device_memory(SuspendRequest(level=suspend_level(discardable)))

    def resume(self, tags: list[str] | None = None) -> None:
        self._state = "RESUMING"
        _resume_device_memory(tags)
        self._state = "RUNNING"

    @classmethod
    def is_supported(cls) -> bool:
        return current_platform.device_type == "npu"

    @classmethod
    def preserves_communicators(cls) -> bool:
        return CaMemSleepBackend.preserves_communicators()
