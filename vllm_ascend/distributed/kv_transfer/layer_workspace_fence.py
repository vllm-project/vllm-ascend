# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequestChunkKey:
    request_id: str
    request_generation: int
    chunk_id: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id is required")
        if self.request_generation < 0 or self.chunk_id < 0:
            raise ValueError("request_generation and chunk_id must be non-negative")


@dataclass(frozen=True)
class WorkspaceKey:
    execution_id: int
    layer_id: int
    arena_id: int
    workspace_generation: int

    def __post_init__(self) -> None:
        if min(
            self.execution_id,
            self.layer_id,
            self.arena_id,
            self.workspace_generation,
        ) < 0:
            raise ValueError("Workspace identity fields must be non-negative")


class WorkspaceIdentityError(RuntimeError):
    pass


class WorkspaceTransferError(RuntimeError):
    pass


class LayerWorkspaceFence:
    """Protect one workspace use from transfer readers and device compute."""

    def __init__(self, key: WorkspaceKey):
        self.key = key
        self._pending = 0
        self._closed = False
        self._failure: BaseException | None = None
        self._compute_release_event: Any | None = None
        self._condition = threading.Condition()

    @property
    def pending(self) -> int:
        with self._condition:
            return self._pending

    @property
    def registration_closed(self) -> bool:
        with self._condition:
            return self._closed

    def add_source_future(
        self,
        future: Future[Any],
        identity: WorkspaceKey,
    ) -> None:
        if identity != self.key:
            raise WorkspaceIdentityError(
                f"Future identity {identity!r} does not match fence {self.key!r}"
            )
        with self._condition:
            if self._closed:
                raise RuntimeError("Cannot add a source future after registration closes")
            self._pending += 1
        future.add_done_callback(self._complete)

    def set_compute_release_event(
        self,
        event: Any,
        identity: WorkspaceKey,
    ) -> None:
        if identity != self.key:
            raise WorkspaceIdentityError(
                f"Compute release identity {identity!r} does not match "
                f"fence {self.key!r}"
            )
        with self._condition:
            if self._closed:
                raise RuntimeError(
                    "Cannot set a compute release event after registration closes"
                )
            if self._compute_release_event is not None:
                raise RuntimeError("Compute release event is already set")
            self._compute_release_event = event

    def close_registration(self) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("Workspace future registration is already closed")
            self._closed = True
            if self._pending == 0:
                self._condition.notify_all()

    def _complete(self, done: Future[Any]) -> None:
        try:
            error = done.exception()
        except BaseException as callback_error:
            error = callback_error
        with self._condition:
            if error is not None and self._failure is None:
                self._failure = error
            self._pending -= 1
            if self._pending < 0:
                raise RuntimeError("A source future completed more than once")
            if self._closed and self._pending == 0:
                self._condition.notify_all()

    def wait_source_safe(self, timeout: float | None = None) -> None:
        with self._condition:
            completed = self._condition.wait_for(
                lambda: self._closed and self._pending == 0,
                timeout=timeout,
            )
            if not completed:
                raise TimeoutError(f"Timed out waiting for workspace fence {self.key!r}")
            if self._failure is not None:
                raise WorkspaceTransferError(
                    f"Workspace source transfer failed for {self.key!r}"
                ) from self._failure

    def wait_workspace_reusable(self, timeout: float | None = None) -> None:
        """Wait until neither transfer nor compute can access the workspace."""
        source_error: BaseException | None = None
        try:
            self.wait_source_safe(timeout=timeout)
        except BaseException as error:
            source_error = error

        with self._condition:
            compute_release_event = self._compute_release_event

        compute_error: BaseException | None = None
        if compute_release_event is None:
            compute_error = RuntimeError(
                f"Compute release event is missing for workspace fence {self.key!r}"
            )
        else:
            try:
                compute_release_event.synchronize()
            except BaseException as error:
                compute_error = error

        if source_error is not None:
            raise source_error
        if compute_error is not None:
            raise compute_error
