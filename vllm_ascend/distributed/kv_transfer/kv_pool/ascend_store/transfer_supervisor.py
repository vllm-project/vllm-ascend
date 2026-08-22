from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from vllm.logger import logger


class TransferWorkerError(RuntimeError):
    """Raised when an asynchronous transfer worker cannot make progress."""


class TransferTimeoutError(TimeoutError):
    """Raised when a transfer wait exceeds its configured deadline."""


@dataclass
class _CommandState:
    status: str = "new"
    error: BaseException | None = None


class TransferSupervisor:
    """Own transfer failure state and wake every dependent waiter.

    The supervisor deliberately lives in the model worker process.  It is
    shared by thread and planner backends, so switching backends does not
    change connector error or timeout semantics.
    """

    def __init__(self, timeout_s: float = 30.0) -> None:
        self.timeout_s = max(float(timeout_s), 0.001)
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._fatal_error: BaseException | None = None
        self._generation = 0
        self._commands: dict[int, _CommandState] = {}
        self._ready_workers: set[str] = set()
        self._heartbeat_ns: dict[str, int] = {}

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def submit(self, command_id: int) -> None:
        with self._condition:
            self._commands[command_id] = _CommandState(status="submitted")

    def worker_ready(self, worker_name: str) -> None:
        with self._condition:
            self._ready_workers.add(worker_name)
            self._heartbeat_ns[worker_name] = time.monotonic_ns()
            self._condition.notify_all()

    def worker_heartbeat(self, worker_name: str) -> None:
        with self._condition:
            self._heartbeat_ns[worker_name] = time.monotonic_ns()

    def mark_running(self, command_id: int) -> None:
        with self._condition:
            state = self._commands.setdefault(command_id, _CommandState())
            state.status = "running"

    def complete(self, command_id: int) -> None:
        with self._condition:
            state = self._commands.setdefault(command_id, _CommandState())
            state.status = "succeeded"
            self._condition.notify_all()

    def fail(self, command_id: int, error: BaseException) -> None:
        with self._condition:
            state = self._commands.setdefault(command_id, _CommandState())
            state.status = "failed"
            state.error = error
            self._condition.notify_all()

    def cancel(self, command_id: int) -> None:
        with self._condition:
            state = self._commands.setdefault(command_id, _CommandState())
            state.status = "cancelled"
            self._condition.notify_all()

    def report_fatal(self, error: BaseException, worker_name: str) -> None:
        with self._condition:
            if self._fatal_error is None:
                self._fatal_error = error
                self._generation += 1
                logger.error(
                    "Transfer worker %s entered fatal state (generation=%d): %s",
                    worker_name,
                    self._generation,
                    error,
                )
            self._stop_event.set()
            for state in self._commands.values():
                if state.status in {"submitted", "running"}:
                    state.status = "lost"
                    state.error = error
            self._condition.notify_all()

    def clear_failure_for_restart(self) -> int:
        with self._condition:
            self._fatal_error = None
            self._stop_event.clear()
            self._generation += 1
            self._condition.notify_all()
            return self._generation

    def raise_if_failed(self) -> None:
        with self._condition:
            error = self._fatal_error
        if error is not None:
            raise TransferWorkerError("asynchronous transfer worker failed") from error

    def wait_for_event(
        self,
        event: threading.Event,
        *,
        description: str,
        timeout_s: float | None = None,
    ) -> None:
        deadline = time.monotonic() + (self.timeout_s if timeout_s is None else max(timeout_s, 0.001))
        while not event.wait(timeout=min(0.1, max(deadline - time.monotonic(), 0.0))):
            self.raise_if_failed()
            if self._stop_event.is_set():
                if self._fatal_error is not None:
                    raise TransferWorkerError("asynchronous transfer worker failed") from self._fatal_error
                raise TransferWorkerError(f"transfer supervisor stopped while waiting for {description}")
            if time.monotonic() >= deadline:
                raise TransferTimeoutError(f"timed out waiting for {description}")

    def wait_for_command(self, command_id: int, timeout_s: float | None = None) -> str:
        deadline = time.monotonic() + (self.timeout_s if timeout_s is None else max(timeout_s, 0.001))
        with self._condition:
            while True:
                state = self._commands.get(command_id)
                if state is not None and state.status in {"succeeded", "failed", "cancelled", "lost"}:
                    if state.error is not None:
                        raise TransferWorkerError(f"transfer command {command_id} failed") from state.error
                    return state.status
                if self._fatal_error is not None:
                    raise TransferWorkerError("asynchronous transfer worker failed") from self._fatal_error
                if self._stop_event.is_set():
                    raise TransferWorkerError(f"transfer supervisor stopped for command {command_id}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TransferTimeoutError(f"timed out waiting for transfer command {command_id}")
                self._condition.wait(timeout=min(remaining, 0.1))

    def shutdown(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
