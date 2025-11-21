# Copyright (c) HuaWei Technologies Co., Ltd. 2025-2025. All rights reserved

import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from functools import wraps
from typing import ClassVar, Dict, Optional

from omni.adaptors.vllm.ems.ems_env import EmsEnv
from vllm.logger import logger


class MetricsType(Enum):
    LOAD = auto()
    ASYNC_SAVE = auto()
    SYNC_SAVE_EVENT = auto()
    CAL_KEYS = auto()
    CAL_VALUES = auto()
    LOAD_KV_CACHE = auto()
    PRE_CC_HANDLE = auto()


@dataclass
class MetricsStat:

    def __init__(self):
        self.total_count: int = 0
        self.total_latency_us: int = 0
        self.max_latency_us: int = -sys.maxsize - 1
        self.min_latency_us: int = sys.maxsize

    def reset(self) -> None:
        self.__init__()

    @property
    def average_latency_us(self) -> Optional[int]:
        return self.total_latency_us // self.total_count if self.total_count > 0 else None

    def update_stat(self, elapsed_time_us) -> None:
        self.total_count += 1
        self.total_latency_us += elapsed_time_us
        self.min_latency_us = min(self.min_latency_us, elapsed_time_us)
        self.max_latency_us = max(self.max_latency_us, elapsed_time_us)


class MetricsManager:
    NS_TO_US: ClassVar[int] = 1_000

    def __init__(self, log_interval_second: int = 30):
        self.log_interval_second = log_interval_second
        self.metrics_stats: Dict[MetricsType, MetricsStat] = {
            metric: MetricsStat()
            for metric in MetricsType
        }

        if EmsEnv.enable_ems_profiling:
            self._start_log_task()

    def metrics_start(self, metrics_type: MetricsType) -> Optional[int]:
        return time.perf_counter_ns() if EmsEnv.enable_ems_profiling else None

    def metrics_end(self, metrics_type: MetricsType,
                    start_time: Optional[int]) -> None:
        if not EmsEnv.enable_ems_profiling or start_time is None:
            return

        elapsed_time_us = (time.perf_counter_ns() -
                           start_time) // self.NS_TO_US
        self.metrics_stats[metrics_type].update_stat(elapsed_time_us)

    def _start_log_task(self):
        background_thread = threading.Thread(target=self._log_task,
                                             name="ems-metrics")
        background_thread.daemon = True
        background_thread.start()

    def _log_task(self):
        while True:
            time.sleep(self.log_interval_second)
            self._log_metrics()
            self._reset()

    def _log_metrics(self) -> None:
        for metrics_type, stat in self.metrics_stats.items():
            if stat.total_count <= 0:
                continue

            values = [
                metrics_type.name, stat.total_count, stat.average_latency_us,
                stat.max_latency_us, stat.min_latency_us, stat.total_latency_us
            ]
            logger.info(f"{' '.join(str(value) for value in values)}")

    def _reset(self) -> None:
        for stat in self.metrics_stats.values():
            stat.reset()


def collect_metric(metrics_type: MetricsType):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = g_metrics_manager.metrics_start(metrics_type)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                g_metrics_manager.metrics_end(metrics_type, start_time)

        return wrapper

    return decorator


g_metrics_manager = MetricsManager()
