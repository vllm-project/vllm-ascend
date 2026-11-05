# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Exclusive wall-clock stage accounting for RFork startup."""

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager


class StageTimer:
    def __init__(self, initial_stage: str) -> None:
        self.started = self.last = time.perf_counter_ns()
        self.current = initial_stage
        self.stages: dict[str, int] = {}

    def mark(self, stage: str) -> None:
        now = time.perf_counter_ns()
        self.stages[self.current] = self.stages.get(self.current, 0) + now - self.last
        self.last = now
        self.current = stage

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.mark(name)
        try:
            yield
        finally:
            self.mark("coordination_and_overhead")

    @contextmanager
    def track_context_exit(self) -> Iterator[None]:
        try:
            yield
        finally:
            self.mark("context_exit")

    def finish(self) -> tuple[float, str]:
        self.mark("done")
        return (self.last - self.started) / 1e9, json.dumps(
            {name: round(value / 1e9, 6) for name, value in self.stages.items()}, separators=(",", ":")
        )
