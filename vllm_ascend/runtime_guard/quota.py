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

"""Dump quota / cooldown (no pending state)."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from vllm_ascend.logger import init_logger_ascend

if TYPE_CHECKING:
    from vllm_ascend.runtime_config.config import RuntimeConfig

logger = init_logger_ascend(__name__)


class DumpQuota:
    def __init__(self, runtime_config: RuntimeConfig) -> None:
        self._runtime_config = runtime_config
        self._total_count = 0
        self._last_ts: float | None = None
        self._lock = threading.Lock()
        self.sync_from_config()

    def sync_from_config(self) -> None:
        self._max_times = int(self._runtime_config.dump_max_times())
        self._cooldown = float(self._runtime_config.dump_cooldown_seconds())

    @property
    def total_count(self) -> int:
        return self._total_count

    @property
    def max_times(self) -> int:
        return self._max_times

    def can_consume(self, *, consume_quota: bool = True) -> bool:
        if not consume_quota:
            return True
        if self._max_times <= 0:
            return False
        if self._total_count >= self._max_times:
            return False
        if self._last_ts is not None and self._cooldown > 0:
            if time.time() - self._last_ts < self._cooldown:
                return False
        return True

    def try_consume(self, *, consume_quota: bool = True) -> bool:
        """Atomic check + consume (B'4).

        ``can_consume()`` followed by a separate ``consume()`` leaves a window
        where a hot-reloaded quota/cooldown boundary flips between the two —
        snapshots already captured then get dropped without being counted.
        """
        with self._lock:
            if not self.can_consume(consume_quota=consume_quota):
                return False
            if consume_quota:
                self._total_count += 1
                self._last_ts = time.time()
            return True

    def refund(self, *, consume_quota: bool = True) -> None:
        """Give back one unit when a consumed dump captured nothing.

        Also clears cooldown (``_last_ts``): empty queue / all-skip arms must
        not block the next auto dump for ``auto_cooldown_seconds``.
        """
        if not consume_quota:
            return
        with self._lock:
            if self._total_count > 0:
                self._total_count -= 1
            self._last_ts = None


    def snapshot(self) -> tuple[int, int]:
        return self._total_count, self._max_times
