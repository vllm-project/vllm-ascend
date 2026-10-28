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
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, runtime_checkable

import torch
from vllm.forward_context import BatchDescriptor, get_forward_context, is_forward_context_available


class DeviceMetadataStage(IntEnum):
    COMPRESSOR = 0
    ATTENTION = 1
    INDEXER = 2


@dataclass(frozen=True, slots=True)
class DeviceMetadataTask:
    stage: DeviceMetadataStage
    run: Callable[[], None]
    group_id: int


@runtime_checkable
class DeviceMetadataTaskProvider(Protocol):
    def enable_device_metadata(self) -> None: ...

    def take_device_metadata_tasks(self) -> tuple[DeviceMetadataTask, ...]: ...


class DeviceMetadataExecutor:
    """Submit device metadata tasks on a worker-owned NPU stream."""

    def __init__(self, *, capture_producers: bool = False) -> None:
        # MRV1 keeps its graph-external producer protocol unchanged. MRV2
        # captures both sides of the dependency and joins them in one graph.
        self.capture_producers = capture_producers
        self.stream = torch.npu.Stream()
        self._inputs_ready = torch.npu.Event()
        self._stage_ready: dict[tuple[DeviceMetadataStage, int], torch.npu.Event] = {}
        self._external_stage_ready: dict[tuple[BatchDescriptor, DeviceMetadataStage, int], torch.npu.ExternalEvent] = {}
        self._external_frontiers: dict[BatchDescriptor, set[tuple[DeviceMetadataStage, int]]] = {}
        self._buffer_reusable = torch.npu.Event()
        self._has_reuse_fence = False
        self._submission_in_flight = False
        self._waited_stages: set[tuple[DeviceMetadataStage, int]] = set()
        self._batch_descriptor: BatchDescriptor | None = None
        # Reuse each SAS group's first-consumption order across batch shapes.
        self._attention_order: dict[int, int] = {}

    @property
    def submission_in_flight(self) -> bool:
        return self._submission_in_flight

    @property
    def uses_external_events(self) -> bool:
        return self._batch_descriptor is not None

    def submit(
        self,
        tasks: Iterable[DeviceMetadataTask],
        batch_descriptor: BatchDescriptor | None = None,
    ) -> None:
        if self.capture_producers and batch_descriptor is not None:
            raise ValueError("Graph-side producers must not use graph-external events")
        if self._submission_in_flight:
            raise RuntimeError("The previous device metadata submission has not been released")
        ordered_tasks = sorted(
            tasks,
            key=lambda task: (
                task.stage,
                self._attention_order.get(task.group_id, len(self._attention_order))
                if task.stage == DeviceMetadataStage.ATTENTION
                else 0,
            ),
        )
        if not ordered_tasks:
            raise ValueError("At least one device metadata task is required")
        # Graph waits bind to event keys, independently of producer order.
        submitted_frontiers = {(task.stage, task.group_id) for task in ordered_tasks}
        expected_frontiers = self._external_frontiers.get(batch_descriptor) if batch_descriptor is not None else None
        if expected_frontiers is not None and expected_frontiers != submitted_frontiers:
            raise RuntimeError("Device metadata frontiers changed for an existing full-graph batch descriptor")
        for task in ordered_tasks:
            frontier = (task.stage, task.group_id)
            external_frontier = (batch_descriptor, *frontier) if batch_descriptor is not None else None
            if external_frontier is not None and external_frontier not in self._external_stage_ready:
                self._external_stage_ready[external_frontier] = torch.npu.ExternalEvent()
            elif external_frontier is None and frontier not in self._stage_ready:
                self._stage_ready[frontier] = torch.npu.Event()
        if batch_descriptor is not None and expected_frontiers is None:
            self._external_frontiers[batch_descriptor] = submitted_frontiers

        self._submission_in_flight = True
        self._batch_descriptor = batch_descriptor
        self._waited_stages.clear()
        self._inputs_ready.record(torch.npu.current_stream())
        with torch.npu.stream(self.stream):
            self.stream.wait_event(self._inputs_ready)
            if self._has_reuse_fence and not self.capture_producers:
                self.stream.wait_event(self._buffer_reusable)

            task_index = 0
            for stage in DeviceMetadataStage:
                while task_index < len(ordered_tasks) and ordered_tasks[task_index].stage == stage:
                    task = ordered_tasks[task_index]
                    task.run()
                    frontier = (stage, task.group_id)
                    if batch_descriptor is None:
                        self._stage_ready[frontier].record(self.stream)
                    else:
                        self._external_stage_ready[(batch_descriptor, *frontier)].record(self.stream)
                    task_index += 1

    def wait(self, stage: DeviceMetadataStage, group_id: int) -> None:
        if not self._submission_in_flight:
            raise RuntimeError("No device metadata submission is in flight")
        frontier = (stage, group_id)
        if frontier not in self._waited_stages:
            stream = torch.npu.current_stream()
            if self._batch_descriptor is None:
                stream.wait_event(self._stage_ready[frontier])
            else:
                event = self._external_stage_ready[(self._batch_descriptor, *frontier)]
                event.wait(stream)
                event.reset(stream)
            self._waited_stages.add(frontier)
            if stage == DeviceMetadataStage.ATTENTION:
                self._attention_order.setdefault(group_id, len(self._attention_order))

    def release(self) -> None:
        if not self._submission_in_flight:
            raise RuntimeError("No device metadata submission is in flight")
        if not self.capture_producers:
            self._buffer_reusable.record(torch.npu.current_stream())
            self._has_reuse_fence = True
        # Captured producers are joined before release. The next input-ready
        # event follows all consumers on the main stream and fences reuse.
        self._submission_in_flight = False
        self._batch_descriptor = None


def wait_for_device_metadata(stage: DeviceMetadataStage, group_id: int) -> None:
    if not is_forward_context_available():
        return
    context = get_forward_context()
    executor = getattr(context, "device_metadata_executor", None)
    if executor is None:
        # MRV2 carries the producer with the attention metadata for this
        # forward. Draft metadata cannot inherit a target execution scope.
        metadata = getattr(context, "attn_metadata", None)
        if isinstance(metadata, dict):
            resource = next(iter(metadata.values()), None)
            executor = getattr(resource, "device_metadata_executor", None)
    if executor is not None:
        executor.wait(stage, group_id)


class TargetDeviceMetadata:
    """Target-owned producer stream, isolated from DSpark's metadata builders.

    FULL warmup/capture runs producers in ModelWithContext.forward; replay runs
    those captured nodes, not Python builders. Inputs/outputs remain in the
    builders' persistent padded buffers. Eager execution submits after prepare.
    Every producer is joined before the next async step may reuse its inputs.
    """

    def __init__(self):
        self.executor = DeviceMetadataExecutor(capture_producers=True)
        self._tasks: tuple[DeviceMetadataTask, ...] = ()
        self._failed = False

    def run_build(self, build_fn, **kwargs):
        full_graph = kwargs.get("full_graph_mode", False) or kwargs.get("for_cudagraph_capture", False)
        with self.build(kwargs["attn_groups"], full_graph):
            metadata = build_fn(**kwargs)
            for resource in metadata.values():
                resource.device_metadata_executor = self.executor
            return metadata

    @contextmanager
    def build(self, attn_groups, full_graph: bool):
        if self._failed:
            raise RuntimeError("Metadata producer failed; recreate the target model state before retrying")
        if self.executor.submission_in_flight or self._tasks:
            raise RuntimeError("Target metadata was not retired before rebuilding inputs")
        providers = {
            id(builder): builder
            for groups in attn_groups
            for group in groups
            for builder in (group.get_metadata_builder(0),)
            if hasattr(builder, "defer_device_metadata")
        }
        with ExitStack() as stack:
            for provider in providers.values():
                stack.enter_context(provider.defer_device_metadata(in_graph=full_graph))
            try:
                yield
            except BaseException:
                for provider in providers.values():
                    provider.take_device_metadata_tasks()
                raise
            self._tasks = tuple(
                task for provider in providers.values() for task in provider.take_device_metadata_tasks()
            )
        if not full_graph:
            self.begin_forward()

    def begin_forward(self):
        """Called inside target graph warmup/capture, or after eager prepare."""
        if not self._tasks or self.executor.submission_in_flight:
            return
        try:
            self.executor.submit(self._tasks)
        except BaseException:
            self._failed = True
            if self.executor.submission_in_flight:
                torch.npu.current_stream().wait_stream(self.executor.stream)
                self.executor.release()
            self._tasks = ()
            raise

    def finish(self):
        if self.executor.submission_in_flight:
            # Capture the join too, including any producer without a consumer
            # in a dummy path. Never globally synchronize the device/host.
            for task in self._tasks:
                self.executor.wait(task.stage, task.group_id)
            self.executor.release()
        self._tasks = ()

    def finish_replay(self):
        # Producers, waits and joins are all graph nodes. Runtime preparation
        # only refreshed their persistent inputs; do not re-submit on the host.
        if self.executor.submission_in_flight:
            raise RuntimeError("Graph-side metadata was unexpectedly submitted outside capture")
        self._tasks = ()
