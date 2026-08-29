#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, runtime_checkable

import torch


class DeviceMetadataStage(IntEnum):
    COMPRESSOR = 0
    INDEXER = 1
    ATTENTION = 2


@dataclass(frozen=True, slots=True)
class DeviceMetadataTask:
    stage: DeviceMetadataStage
    run: Callable[[], None]


@dataclass(frozen=True, slots=True)
class DeviceMetadataTaskContext:
    common_attn_metadata: object
    layer_attn_metadata: object
    for_cudagraph_capture: bool


@runtime_checkable
class DeviceMetadataTaskProvider(Protocol):
    def build_device_metadata_tasks(self, context: DeviceMetadataTaskContext) -> tuple[DeviceMetadataTask, ...]: ...


class DeviceMetadataExecutor:
    """Submit device metadata tasks on a worker-owned NPU stream."""

    def __init__(self) -> None:
        self.stream = torch.npu.Stream()
        self._inputs_ready = torch.npu.Event()
        self._stage_ready = {stage: torch.npu.Event() for stage in DeviceMetadataStage}
        self._buffer_reusable = torch.npu.Event()
        self._has_reuse_fence = False
        self._submission_in_flight = False

    def submit(self, tasks: Iterable[DeviceMetadataTask]) -> None:
        if self._submission_in_flight:
            raise RuntimeError("The previous device metadata submission has not been released")
        ordered_tasks = tuple(sorted(tasks, key=lambda task: task.stage))
        if not ordered_tasks:
            raise ValueError("At least one device metadata task is required")

        self._inputs_ready.record(torch.npu.current_stream())
        with torch.npu.stream(self.stream):
            self.stream.wait_event(self._inputs_ready)
            if self._has_reuse_fence:
                self.stream.wait_event(self._buffer_reusable)

            task_index = 0
            for stage in DeviceMetadataStage:
                while task_index < len(ordered_tasks) and ordered_tasks[task_index].stage == stage:
                    ordered_tasks[task_index].run()
                    task_index += 1
                self._stage_ready[stage].record(self.stream)

        self._submission_in_flight = True

    def wait(self, stage: DeviceMetadataStage) -> None:
        if not self._submission_in_flight:
            raise RuntimeError("No device metadata submission is in flight")
        torch.npu.current_stream().wait_event(self._stage_ready[stage])

    def release(self) -> None:
        if not self._submission_in_flight:
            raise RuntimeError("No device metadata submission is in flight")
        self._buffer_reusable.record(torch.npu.current_stream())
        self._has_reuse_fence = True
        self._submission_in_flight = False
