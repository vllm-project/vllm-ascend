# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Cumulative observations without transfer waits or admission changes."""

from __future__ import annotations

import json
import threading
import time

from vllm.logger import logger

_LOG_INTERVAL_SECONDS = 5.0


class ECMetrics:
    def __init__(self, component: str) -> None:
        self.component = component
        self._lock = threading.Lock()
        self._values: dict[str, float] = {}
        self._last_log = time.monotonic()

    def add(self, **values: float) -> None:
        with self._lock:
            for key, value in values.items():
                self._values[key] = self._values.get(key, 0) + value

    def duration(self, name: str, seconds: float) -> None:
        with self._lock:
            for key, value in ((name + "_count", 1), (name + "_seconds", seconds)):
                self._values[key] = self._values.get(key, 0) + value
            key = name + "_max_seconds"
            self._values[key] = max(self._values.get(key, 0), seconds)

    def snapshot(self) -> dict:
        with self._lock:
            return {"component": self.component, "cumulative": True, **self._values}

    def emit(self, *, force: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
            if not force and now - self._last_log < _LOG_INTERVAL_SECONDS:
                return
            self._last_log = now
            result = {"component": self.component, "cumulative": True, "forced_snapshot": force, **self._values}
        logger.info("EC_OBSERVATION %s", json.dumps(result, sort_keys=True))
