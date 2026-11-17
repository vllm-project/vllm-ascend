#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Background queue: wait rare logprobs ready, D2H, then pack+feed.

Separate from ActionQueue so feed completion does not contend with report/dump
jobs and a short ActionQueue cannot drop rare_chars feeds under high concurrency.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

_MAXSIZE = 512


@dataclass
class RareLogprobBatch:
    """One step of device ``[rows, 2]`` = (top1_logprob, sampled_logprob) + ready event."""

    device_logprobs: torch.Tensor
    ready_event: Any


class RareLogprobFeedQueue:
    """One worker thread for rare_chars logprob wait / D2H / feed jobs."""

    def __init__(self, maxsize: int = _MAXSIZE) -> None:
        self._q: queue.Queue[Callable[[], None] | None] = queue.Queue(maxsize=maxsize)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._started = False
        self.dropped = 0
        self.submitted = 0

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._thread = threading.Thread(
                target=self._loop,
                name="rare-logprob-feed",
                daemon=True,
            )
            self._started = True
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            with contextlib.suppress(queue.Full):
                self._q.put_nowait(None)
            self._started = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def qsize(self) -> int:
        return int(self._q.qsize())

    def submit(self, job: Callable[[], None]) -> bool:
        self.start()
        try:
            self._q.put_nowait(job)
            self.submitted += 1
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                item()
            except Exception:
                logger.exception("[rare_logprob_feed] job failed")
            finally:
                self._q.task_done()


_QUEUE: RareLogprobFeedQueue | None = None
_QUEUE_LOCK = threading.Lock()


def get_rare_logprob_feed_queue() -> RareLogprobFeedQueue:
    global _QUEUE
    with _QUEUE_LOCK:
        if _QUEUE is None:
            _QUEUE = RareLogprobFeedQueue()
        return _QUEUE


def reset_rare_logprob_feed_queue_for_tests() -> None:
    """Drop singleton (unit tests only)."""
    global _QUEUE
    with _QUEUE_LOCK:
        if _QUEUE is not None:
            _QUEUE.stop()
        _QUEUE = None
