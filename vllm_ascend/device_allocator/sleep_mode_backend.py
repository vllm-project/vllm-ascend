# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend mechanisms for the upstream sleep-mode dispatch interface.

The backend is constructed without a device context by the upstream factory.
Each worker subsequently binds its own runtime; no process-global worker is used.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from threading import RLock
from typing import TYPE_CHECKING

from vllm.device_allocator.sleep_mode_backend import SleepModeBackend, SleepModeBackendFactory
from vllm.logger import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from vllm_ascend.device_allocator.sleep_runtime import AscendSleepRuntime


class AscendLegacySleepModeBackend(SleepModeBackend):
    """Compatibility adapter for the pre-backend Ascend ordering."""

    def __init__(self) -> None:
        super().__init__()
        self._runtime: AscendSleepRuntime | None = None

    @classmethod
    def is_supported(cls) -> bool:
        from vllm.platforms import current_platform

        return current_platform.device_type == "npu"

    def bind_runtime(self, runtime: AscendSleepRuntime) -> None:
        if self._runtime is not None:
            raise RuntimeError("Sleep backend is already bound to a worker")
        self._runtime = runtime

    @property
    def runtime(self) -> AscendSleepRuntime:
        if self._runtime is None:
            raise RuntimeError("Sleep backend has no worker runtime")
        return self._runtime

    def suspend(self, level: int = 1) -> None:
        self.runtime.suspend_legacy(level)
        self._state = "SUSPENDED"

    def resume(self, tags: list[str] | None = None) -> None:
        self._state = "RESUMING"
        self.runtime.resume_legacy(tags)
        self._state = "RUNNING"

    def close(self) -> None:
        # Runtime resources are borrowed. Do not wake unmapped model storage
        # or destroy shared allocator/communication objects during shutdown.
        self._runtime = None


class AscendSleepModeBackend(AscendLegacySleepModeBackend):
    """Opt-in serialized lifecycle with explicit partial-resume readiness.

    Upstream only exposes three states. A failed or closed backend therefore
    remains SUSPENDED/RESUMING and rejects execution; it never advertises RUNNING.
    Errors after cleanup starts are not automatically retried: allocator copies
    or communicator destruction can have partially completed on another rank.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = RLock()
        self._level: int | None = None
        self._pending_tags: set[str] = set()
        self._known_tags: set[str] = set()
        self._failed = False
        self._closed = False
        self._completed_stages: list[str] = []

    def _check_open(self) -> None:
        if self._closed or self._failed:
            raise RuntimeError("Sleep backend is closed or failed; restart the worker")

    def _stage(self, name, operation) -> None:
        """Agree on local results before any rank enters the next stage.

        The runtime uses the surviving CPU world group, not HCCL. A failure
        *inside* a device collective still requires the distributed timeout;
        this protocol cannot repair a lost process or a corrupted device.
        """
        error = None
        try:
            operation()
        except Exception as exc:
            error = exc
        self.runtime.agree(name, error)
        if error is not None:
            raise error
        self._completed_stages.append(name)
        logger.info("Ascend sleep backend stage=%s status=complete", name)

    def suspend(self, level: int = 1) -> None:
        with self._lock:
            self._check_open()
            self.runtime.agree(f"sleep-request:{level}:{self._state}:{self._level}", None)
            if level not in (1, 2):
                raise ValueError("Ascend sleep level must be 1 or 2")
            if self._state == "SUSPENDED" and self._level == level:
                return
            if self._state != "RUNNING":
                raise RuntimeError("Finish the current sleep/wake cycle first")
            self._completed_stages = []
            self._state = "SUSPENDED"
            destructive = False
            try:
                self._stage("drain", self.runtime.synchronize)
                self._stage("backup", lambda: self.runtime.backup(level))

                # Snapshot before unmapping. Require both public pools even
                # when a small/test workload has no allocation in one of them.
                def inventory():
                    self._known_tags = self.runtime.allocation_tags() | {"weights", "kv_cache"}
                    self._pending_tags = self._known_tags.copy()

                self._stage("inventory", inventory)
                self.runtime.agree(f"sleep-pools:{sorted(self._known_tags)}", None)
                destructive = True
                self._stage("cleanup", self.runtime.cleanup)
                self._stage("quiesce", self.runtime.synchronize)
                self._stage("unregister", self.runtime.unregister)
                self._stage("unmap", lambda: self.runtime.sleep_allocator(level))
                self._level = level
            except Exception:
                if not destructive and "drain" in self._completed_stages:
                    # Backup only creates CPU copies. Clearing them is safe;
                    # no device resources have been invalidated yet.
                    try:
                        self._stage("rollback_backup", self.runtime.clear_backup)
                        self._state = "RUNNING"
                    except Exception:
                        self._failed = True
                else:
                    self._failed = True
                raise

    def resume(self, tags: list[str] | None = None) -> None:
        with self._lock:
            self._check_open()
            requested_tags = None if tags is None else sorted(set(tags))
            self.runtime.agree(f"wake-request:{requested_tags}:{self._state}:{sorted(self._pending_tags)}", None)
            if self._state == "RUNNING":
                return
            requested = self._pending_tags.copy() if tags is None else set(tags)
            unknown = requested - self._known_tags
            if unknown:
                raise ValueError(f"Unknown or inactive sleep allocation tags: {sorted(unknown)}")
            selected = requested & self._pending_tags
            if not selected:
                return
            # Capturing with unmapped weights would dereference invalid storage.
            if "kv_cache" in selected and self._pending_tags - selected - {"kv_cache"}:
                raise ValueError("Restore weights before the KV cache, including all other pending pools")
            self._stage("validate_wake", self.runtime.validate_wake)
            self._state = "RESUMING"
            actual_tags = None if tags is None else sorted(selected)
            try:
                self._stage("remap", lambda: self.runtime.wake_allocator(actual_tags))
                if "kv_cache" in selected:
                    self._stage("reregister", self.runtime.reregister)
                if "weights" in selected:
                    self._stage("restore", self.runtime.restore)
                # Keep the existing weights-first HCCL initialization for
                # external weight loading; graphs only capture on KV wake.
                if selected & {"weights", "kv_cache"}:
                    self._stage("rebuild", lambda: self.runtime.rebuild(actual_tags))
                self._stage("ready", self.runtime.synchronize)
                self._pending_tags.difference_update(selected)
                if not self._pending_tags:
                    self._state = "RUNNING"
                    self._level = None
            except Exception:
                self._failed = True
                raise

    @contextmanager
    def execution(self) -> Iterator[None]:
        with self._lock:
            self._check_open()
            if self._state != "RUNNING":
                raise RuntimeError(f"Cannot execute while sleep backend is {self._state}")
            yield

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._state = "SUSPENDED"
            super().close()


def register_sleep_mode_backends() -> None:
    """Idempotent discovery without changing the built-in GPU cumem mapping."""
    for name, backend in (
        ("ascend", AscendSleepModeBackend),
        ("ascend_legacy", AscendLegacySleepModeBackend),
    ):
        try:
            SleepModeBackendFactory.register_backend(name, __name__, backend.__name__)
        except ValueError:
            if SleepModeBackendFactory.get_backend_class(name) is not backend:
                raise


def normalize_sleep_mode_backend(model_config) -> None:
    register_sleep_mode_backends()
    if model_config.sleep_mode_backend == "cumem":
        model_config.sleep_mode_backend = "ascend_legacy"
    backend = SleepModeBackendFactory.get_backend_class(model_config.sleep_mode_backend)
    if not issubclass(backend, AscendLegacySleepModeBackend):
        raise ValueError(f"Unsupported Ascend sleep backend: {model_config.sleep_mode_backend}")


def guard_sleep_execution(method):
    """Serialize new-backend inference with lifecycle RPCs in the same worker."""

    @wraps(method)
    def guarded(worker, *args, **kwargs):
        backend = getattr(worker, "sleep_mode_backend", None)
        if isinstance(backend, AscendSleepModeBackend):
            with backend.execution():
                return method(worker, *args, **kwargs)
        return method(worker, *args, **kwargs)

    return guarded
