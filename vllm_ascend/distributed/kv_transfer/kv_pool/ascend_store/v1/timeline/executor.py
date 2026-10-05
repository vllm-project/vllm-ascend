"""Run timeline commands serially on one process-owned thread."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Generic, TypeVar, cast

from vllm.logger import logger

CommandT = TypeVar("CommandT")
_STOP = object()


class TimelineExecutor(threading.Thread, Generic[CommandT]):
    """Own the common thread, queue and terminal failure of a timeline."""

    def __init__(
        self,
        name: str,
        initializer: Callable[[], None],
        execute: Callable[[CommandT], None],
        complete: Callable[[CommandT], None] | None = None,
    ) -> None:
        super().__init__(daemon=True, name=name)
        self._initializer = initializer
        self._execute = execute
        self._complete = complete
        self._ready = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._has_started = False
        self._closed = False
        self._join_complete = False
        self._stop_requested = False
        self._queue: queue.Queue[CommandT | object] = queue.Queue()
        self._failure: BaseException | None = None
        self._failed_command: CommandT | None = None
        self._discarded_commands: tuple[CommandT, ...] = ()

    @property
    def failure(self) -> BaseException | None:
        with self._lifecycle_lock:
            return self._failure

    @property
    def closed(self) -> bool:
        with self._lifecycle_lock:
            return self._closed

    @property
    def stopped(self) -> bool:
        """Whether close has confirmed that no executor thread can run."""
        with self._lifecycle_lock:
            return self._join_complete

    @property
    def failed_command(self) -> CommandT | None:
        with self._lifecycle_lock:
            return self._failed_command

    @property
    def discarded_commands(self) -> tuple[CommandT, ...]:
        with self._lifecycle_lock:
            return self._discarded_commands

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError(f"{self.name} is closed")
            if self._failure is not None:
                return
            if not self._has_started:
                super().start()
                self._has_started = True
        self._ready.wait()

    def check_running(self) -> None:
        with self._lifecycle_lock:
            if self._failure is not None:
                raise RuntimeError(f"{self.name} has failed") from self._failure
            if not self._has_started:
                raise RuntimeError(f"{self.name} has not started")
            if self._closed:
                raise RuntimeError(f"{self.name} is closed")

    def submit(self, command: CommandT) -> None:
        with self._lifecycle_lock:
            if self._failure is not None:
                raise RuntimeError(f"{self.name} has failed") from self._failure
            if not self._has_started:
                raise RuntimeError(f"{self.name} has not started")
            if self._closed:
                raise RuntimeError(f"{self.name} is closed")
            self._queue.put(command)

    def terminate(self, error: BaseException) -> None:
        with self._lifecycle_lock:
            self._record_failure(error, None)
            self._request_stop()

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._join_complete:
                return
            self._closed = True
            if not self._has_started:
                self._join_complete = True
                return
            self._request_stop()
        self.join()
        if self.is_alive():
            raise RuntimeError(f"{self.name} did not stop after join")
        with self._lifecycle_lock:
            self._join_complete = True

    def run(self) -> None:
        try:
            self._initializer()
        except BaseException as error:
            with self._lifecycle_lock:
                self._record_failure(error, None)
            logger.exception("Failed to initialize %s", self.name)
        finally:
            self._ready.set()
        if self.failure is not None:
            return

        while True:
            queued = self._queue.get()
            try:
                if queued is _STOP:
                    return
                command = cast(CommandT, queued)
                failed = False
                try:
                    self._execute(command)
                except BaseException as error:
                    with self._lifecycle_lock:
                        self._record_failure(error, command)
                    logger.exception("Error in %s", self.name)
                    failed = True
                if not self._notify_completion(command) or failed:
                    return
            finally:
                self._queue.task_done()

    def _notify_completion(self, command: CommandT) -> bool:
        if self._complete is None:
            return True
        try:
            self._complete(command)
        except BaseException as error:
            with self._lifecycle_lock:
                self._record_failure(error, command)
            logger.exception("Failed to publish completion in %s", self.name)
            return False
        return True

    def _request_stop(self) -> None:
        if self._has_started and self.is_alive() and not self._stop_requested:
            self._stop_requested = True
            self._queue.put(_STOP)

    def _record_failure(self, error: BaseException, command: CommandT | None) -> None:
        if self._failure is not None:
            return
        self._failure = error
        self._failed_command = command
        discarded_commands = []
        while True:
            try:
                queued = self._queue.get_nowait()
            except queue.Empty:
                break
            if queued is not _STOP:
                discarded_commands.append(cast(CommandT, queued))
            self._queue.task_done()
        self._discarded_commands = tuple(discarded_commands)
