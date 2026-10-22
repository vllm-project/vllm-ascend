from __future__ import annotations

import queue

from py_monitor.control_api import ControlCommand


class MonitorCommandQueue:
    def __init__(self) -> None:
        self._q: "queue.Queue[ControlCommand]" = queue.Queue()

    @property
    def queue(self) -> "queue.Queue[ControlCommand]":
        return self._q
