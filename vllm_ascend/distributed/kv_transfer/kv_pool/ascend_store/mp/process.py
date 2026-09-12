"""Lifetime owner for one worker-local KV transfer subprocess."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from .client import TRANSFER_TIMEOUT_SECONDS, TransferClient

PROCESS_EXIT_TIMEOUT_SECONDS = 5.0


class TransferProcess:
    """Spawn, monitor and reap the child that serves ``TransferClient``."""

    def __init__(self, timeout: float = TRANSFER_TIMEOUT_SECONDS, command: list[str] | None = None):
        self._close_lock = threading.Lock()
        self._closed = False
        self._directory = tempfile.TemporaryDirectory(prefix="kv-")
        endpoint = f"ipc://{self._directory.name}/worker"
        parent_read, parent_write = os.pipe()
        self._parent_write: int | None = parent_write
        if command is None:
            command = [sys.executable, str(Path(__file__).with_name("worker.py"))]
        try:
            self.process = subprocess.Popen(
                [*command, endpoint, str(parent_read)], pass_fds=(parent_read,), close_fds=True
            )
        except BaseException:
            os.close(parent_write)
            self._parent_write = None
            self._directory.cleanup()
            raise
        finally:
            os.close(parent_read)

        try:
            self.client = TransferClient(endpoint, self.process.poll, timeout)
        except BaseException:
            self._close_parent_pipe()
            self._reap_child()
            self._directory.cleanup()
            raise

    @property
    def stopped(self) -> bool:
        return self.process.poll() is not None

    def close(self) -> None:
        """Close transport, signal parent departure and always reap the child."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.client.close()
            finally:
                self._close_parent_pipe()
                self._reap_child()
                self._directory.cleanup()

    def _close_parent_pipe(self) -> None:
        if self._parent_write is not None:
            os.close(self._parent_write)
            self._parent_write = None

    def _reap_child(self) -> None:
        try:
            self.process.wait(PROCESS_EXIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(PROCESS_EXIT_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(PROCESS_EXIT_TIMEOUT_SECONDS)


__all__ = ["TransferProcess"]
