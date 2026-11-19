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
import os
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

_MAXSIZE = 512


def rare_sync_feed_enabled() -> bool:
    """Debug: run rare D2H+feed on the caller thread (skip RareLogprobFeedQueue).

    Default is async hide via the feed queue. Prefer
    ``vllm_ascend.envs.VLLM_ASCEND_RG_RARE_SYNC_FEED`` when present.
    """
    try:
        from vllm_ascend import envs as _envs

        if hasattr(_envs, "VLLM_ASCEND_RG_RARE_SYNC_FEED"):
            return bool(_envs.VLLM_ASCEND_RG_RARE_SYNC_FEED)
    except Exception:
        pass
    return os.getenv("VLLM_ASCEND_RG_RARE_SYNC_FEED", "0").strip() == "1"


@dataclass
class RareLogprobBatch:
    """One step of device ``[rows, 2]`` = (top1_logprob, sampled_logprob) + ready event.

    ``device_token_ids`` is aligned 1:1 with ``device_logprobs`` rows (including
    ``-1`` pads under MTP / adaptive verification). ``slots_per_req`` is 1 for
    non-spec, ``1 + num_speculative_tokens`` for fixed-slot MTP, and ``0`` when
    spans come from ``cu_num_logits`` (adaptive verification).
    """

    device_logprobs: torch.Tensor
    ready_event: Any
    device_token_ids: torch.Tensor | None = None
    slots_per_req: int = 1
    cu_num_logits: list[int] | None = None


class RareLogprobFeedQueue:
    """One worker thread for rare_chars logprob wait / D2H / feed jobs."""

    def __init__(self, maxsize: int = _MAXSIZE) -> None:
        self._jobs: queue.Queue[Callable[[], None] | None] = queue.Queue(maxsize=maxsize)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._started = False
        self.dropped = 0
        self.submitted = 0

    # Back-compat for tests that touch the private queue attribute.
    @property
    def _q(self) -> queue.Queue[Callable[[], None] | None]:
        return self._jobs

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._thread = threading.Thread(
                target=self._worker_loop,
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
                self._jobs.put_nowait(None)
            self._started = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def qsize(self) -> int:
        return int(self._jobs.qsize())

    def submit(self, job: Callable[[], None]) -> bool:
        self.start()
        try:
            self._jobs.put_nowait(job)
            self.submitted += 1
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _worker_loop(self) -> None:
        while True:
            item = self._jobs.get()
            try:
                if item is None:
                    return
                item()
            except Exception:
                logger.exception("[rare_logprob_feed] job failed")
            finally:
                self._jobs.task_done()


_FEED_QUEUE: RareLogprobFeedQueue | None = None
_FEED_QUEUE_LOCK = threading.Lock()


def get_rare_logprob_feed_queue() -> RareLogprobFeedQueue:
    global _FEED_QUEUE
    with _FEED_QUEUE_LOCK:
        if _FEED_QUEUE is None:
            _FEED_QUEUE = RareLogprobFeedQueue()
        return _FEED_QUEUE


def reset_rare_logprob_feed_queue_for_tests() -> None:
    """Drop singleton (unit tests only)."""
    global _FEED_QUEUE
    with _FEED_QUEUE_LOCK:
        if _FEED_QUEUE is not None:
            _FEED_QUEUE.stop()
        _FEED_QUEUE = None
