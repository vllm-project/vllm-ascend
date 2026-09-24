# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from math import ceil
from time import perf_counter
from typing import Any, Callable

import torch
from vllm.logger import logger


class EplbDeviceTimingWindow:
    """Batch device-event timings without synchronizing every forward."""

    def __init__(self, phase: str, window_size: int) -> None:
        self.phase = phase
        self.window_size = window_size
        self.samples: list[tuple[torch.Event, torch.Event, float]] = []

    @contextmanager
    def measure(self) -> Iterator[None]:
        start_event = torch.Event(enable_timing=True)
        end_event = torch.Event(enable_timing=True)
        started_at = perf_counter()
        start_event.record()
        try:
            yield
        finally:
            end_event.record()
            self.samples.append((start_event, end_event, (perf_counter() - started_at) * 1000))
            if len(self.samples) >= self.window_size:
                self._report()

    def _report(self) -> None:
        self.samples[-1][1].synchronize()
        cpu_ms = [sample[2] for sample in self.samples]
        device_ms = sorted(start.elapsed_time(end) for start, end, _ in self.samples)
        p95_index = ceil(0.95 * len(device_ms)) - 1
        logger.info(
            "EPLB device timing window: phase=%s samples=%d cpu_total_ms=%.3f "
            "device_total_ms=%.3f device_mean_ms=%.3f device_p95_ms=%.3f device_max_ms=%.3f",
            self.phase,
            len(device_ms),
            sum(cpu_ms),
            sum(device_ms),
            sum(device_ms) / len(device_ms),
            device_ms[p95_index],
            device_ms[-1],
        )
        self.samples.clear()


class EplbCpuTimingWindow:
    """Report host time in fixed-size windows."""

    def __init__(self, phase: str, window_size: int) -> None:
        self.phase = phase
        self.window_size = window_size
        self.samples: list[float] = []

    @contextmanager
    def measure(self) -> Iterator[None]:
        started_at = perf_counter()
        try:
            yield
        finally:
            self.samples.append((perf_counter() - started_at) * 1000)
            if len(self.samples) >= self.window_size:
                sorted_ms = sorted(self.samples)
                p95_index = ceil(0.95 * len(sorted_ms)) - 1
                logger.info(
                    "EPLB host timing window: phase=%s samples=%d total_ms=%.3f "
                    "mean_ms=%.3f p95_ms=%.3f max_ms=%.3f",
                    self.phase,
                    len(sorted_ms),
                    sum(sorted_ms),
                    sum(sorted_ms) / len(sorted_ms),
                    sorted_ms[p95_index],
                    sorted_ms[-1],
                )
                self.samples.clear()


def measure_eplb_device_calls(timer_attribute: str) -> Callable:
    """Measure a runner method when its diagnostic timer is enabled."""

    def decorate(method: Callable) -> Callable:
        @wraps(method)
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            timer = getattr(self, timer_attribute, None)
            if timer is None:
                return method(self, *args, **kwargs)
            with timer.measure():
                return method(self, *args, **kwargs)

        return wrapped

    return decorate
