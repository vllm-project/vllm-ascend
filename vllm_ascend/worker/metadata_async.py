# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Opt-in graph-external V4.1 target metadata on A3 Model Runner V2."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger

from vllm_ascend.worker.device_metadata import DeviceMetadataExecutor, DeviceMetadataStage, DeviceMetadataTask

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_ascend.worker.device_metadata import TargetDeviceMetadata


def finish_metadata(
    metadata: TargetDeviceMetadata | AsyncTargetDeviceMetadata | None,
    *,
    failed: bool = False,
    replay: bool = False,
) -> None:
    """Retire metadata without replacing an original forward/capture error."""
    if metadata is None:
        return
    if failed and isinstance(metadata, AsyncTargetDeviceMetadata):
        metadata.abort_after_failure()
    elif replay:
        metadata.finish_replay()
    else:
        metadata.finish()


def create_async_target_metadata(
    config: dict[str, Any],
    vllm_config: VllmConfig,
    *,
    packed_cache: bool,
    pcp_enabled: bool,
    is_a3: bool,
) -> AsyncTargetDeviceMetadata | None:
    """Leave the existing A5/disabled path alone; reject unsupported opt-ins."""
    if not isinstance(config, dict) or set(config) - {"enabled", "mode"}:
        raise ValueError("dsv41_metadata_async_config only accepts enabled and mode")
    enabled, mode = config.get("enabled", False), config.get("mode", "M2")
    if type(enabled) is not bool or not isinstance(mode, str) or mode not in {"M1", "M2"}:
        raise ValueError("Metadata enabled must be a bool and mode must be M1 or M2")
    if not enabled:
        return None
    transfer = vllm_config.kv_transfer_config
    parallel = vllm_config.parallel_config
    if (
        not is_a3
        or packed_cache
        or vllm_config.model_config.architecture != "DeepseekV41ForCausalLM"
        or transfer is None
        or transfer.kv_role != "kv_consumer"
    ):
        raise ValueError("Asynchronous target metadata requires A3 DeepSeek V4.1 with kv_role=kv_consumer")
    if (
        pcp_enabled
        or vllm_config.lora_config is not None
        or parallel.pipeline_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
    ):
        raise ValueError("Asynchronous target metadata does not support LoRA, PP or CP")
    return AsyncTargetDeviceMetadata(mode=mode)


class SelectedMetadataExecutor(DeviceMetadataExecutor):
    """A synchronous frontier needs no event; an unknown frontier is an error."""

    def __init__(self) -> None:
        super().__init__(capture_producers=False)
        self.synchronous_frontiers: set[tuple[DeviceMetadataStage, int]] = set()
        self.asynchronous_frontiers: set[tuple[DeviceMetadataStage, int]] = set()

    def wait(self, stage: DeviceMetadataStage, group_id: int) -> None:
        if not self.submission_in_flight:
            raise RuntimeError("No metadata submission is in flight")
        frontier = (stage, group_id)
        if frontier in self.synchronous_frontiers:
            return
        if frontier not in self.asynchronous_frontiers:
            raise RuntimeError("Consumer references an unregistered metadata buffer")
        super().wait(stage, group_id)


class AsyncTargetDeviceMetadata:
    """Submit current-step producers outside the graph and capture their joins.

    A3 metadata operators consume dynamic Python scalars, so producers must be
    rebuilt on every step. FULL graphs contain per-frontier wait/reset nodes,
    never the producer kernels. M1 overlaps SAS and QLI; M2 only overlaps QLI.
    """

    def __init__(self, *, mode: str = "M2") -> None:
        if not isinstance(mode, str) or mode not in {"M1", "M2"}:
            raise ValueError("Metadata mode must be M1 or M2")
        self.mode = mode
        self.executor = SelectedMetadataExecutor()
        self._tasks: tuple[DeviceMetadataTask, ...] = ()
        self._failed = False
        self._descriptor: BatchDescriptor | None = None
        self._capture_sizes: frozenset[int] = frozenset()
        self._frontiers_by_bucket: dict[int, frozenset[tuple[DeviceMetadataStage, int]]] = {}
        self._captured_tokens: set[int] = set()

    def configure_capture_sizes(self, sizes: Iterable[int]) -> None:
        sizes = frozenset(sizes)
        if not sizes or any(type(size) is not int or size <= 0 for size in sizes):
            raise ValueError("Metadata capture sizes must be positive descriptor token counts")
        if self._capture_sizes and self._capture_sizes != sizes:
            raise ValueError("Cannot change capture sizes after binding metadata events")
        self._capture_sizes = sizes

    def _check_idle(self) -> None:
        if self._failed:
            raise RuntimeError("Metadata executor failed; recreate the target model state")
        if self.executor.submission_in_flight or self._tasks:
            raise RuntimeError("The previous metadata submission has not been retired")

    def run_build(self, build_fn: Callable[..., dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self._check_idle()
        full_graph = kwargs.get("full_graph_mode", False) or kwargs.get("for_cudagraph_capture", False)
        num_tokens = kwargs["num_input_tokens"]
        if full_graph and num_tokens not in self._capture_sizes:
            raise ValueError("Input token count is not a configured capture descriptor")
        self._descriptor = BatchDescriptor(num_tokens=num_tokens) if full_graph else None
        providers = {
            id(builder): builder
            for groups in kwargs["attn_groups"]
            for group in groups
            for builder in (group.get_metadata_builder(0),)
            if hasattr(builder, "defer_device_metadata")
        }
        try:
            with ExitStack() as stack:
                for provider in providers.values():
                    stack.enter_context(provider.defer_device_metadata(in_graph=False))
                resources = build_fn(**kwargs)
                tasks = tuple(task for provider in providers.values() for task in provider.take_device_metadata_tasks())
            frontiers = frozenset((task.stage, task.group_id) for task in tasks)
            if len(frontiers) != len(tasks):
                raise RuntimeError("The same metadata buffer was submitted twice")
            if full_graph:
                expected = self._frontiers_by_bucket.setdefault(num_tokens, frontiers)
                if frontiers != expected:
                    raise RuntimeError("Metadata buffer frontiers changed for an existing capture bucket")
            async_stages = {DeviceMetadataStage.INDEXER}
            if self.mode == "M1":
                async_stages.add(DeviceMetadataStage.ATTENTION)
            self._tasks = tuple(task for task in tasks if task.stage in async_stages)
            synchronous = tuple(task for task in tasks if task.stage not in async_stages)
            if not self._tasks:
                raise RuntimeError("No asynchronous metadata tasks; refusing an empty event graph")
            self.executor.synchronous_frontiers = {(task.stage, task.group_id) for task in synchronous}
            self.executor.asynchronous_frontiers = {(task.stage, task.group_id) for task in self._tasks}
            for task in synchronous:
                task.run()
            self.executor.submit(self._tasks, self._descriptor)
            for resource in resources.values():
                resource.device_metadata_executor = self.executor
            return resources
        except BaseException:
            # Retain the original build/producer error if cleanup also fails.
            for provider in providers.values():
                try:
                    provider.take_device_metadata_tasks()
                except Exception:
                    logger.exception("Failed to discard metadata tasks after a build error")
            self.abort_after_failure()
            raise

    def begin_forward(self) -> None:
        if self._failed:
            raise RuntimeError("Metadata executor is invalid")
        if self._descriptor is not None:
            context = get_forward_context()
            descriptor = context.batch_descriptor
            actual_tokens = descriptor.num_tokens if descriptor is not None else None
            if descriptor is None and context.cudagraph_runtime_mode == CUDAGraphMode.NONE:
                # V2 warmup and the FULL capture body both run in NONE mode.
                padding = context.is_padding
                if padding is not None and padding.ndim == 1:
                    actual_tokens = padding.shape[0]
            expected_tokens = self._descriptor.num_tokens
            if actual_tokens != expected_tokens:
                self.abort_after_failure()
                raise ValueError(
                    f"Metadata capture bucket mismatch: actual={actual_tokens}, expected={expected_tokens}"
                )
        if not self.executor.submission_in_flight:
            raise RuntimeError("Metadata must be submitted before forward")

    def finish(self) -> None:
        if not self.executor.submission_in_flight:
            return
        try:
            # Dummy paths may not consume every group. Capture their waits too.
            for task in self._tasks:
                self.executor.wait(task.stage, task.group_id)
            if self._descriptor is not None and torch.npu.is_current_stream_capturing():
                self._captured_tokens.add(self._descriptor.num_tokens)
            self.executor.release()
            self._tasks = ()
            self._descriptor = None
        except BaseException:
            self.abort_after_failure()
            raise

    def validate_replay(self, num_tokens: int) -> None:
        if self._failed or not self.executor.submission_in_flight:
            raise RuntimeError("Replay requires a valid graph-external metadata submission")
        if self._descriptor is None or num_tokens != self._descriptor.num_tokens:
            raise ValueError("Replay bucket does not match the current metadata submission")
        if num_tokens not in self._captured_tokens:
            raise RuntimeError("Metadata wait/reset nodes were not captured for this bucket")

    def finish_replay(self) -> None:
        try:
            if self._descriptor is None:
                raise RuntimeError("Missing metadata replay descriptor")
            self.validate_replay(self._descriptor.num_tokens)
            # The graph contains wait/reset. Record reuse after all consumers;
            # returning from replay only means the graph has been submitted.
            self.executor.release()
            self._tasks = ()
            self._descriptor = None
        except BaseException:
            self.abort_after_failure()
            raise

    def abort_after_failure(self) -> None:
        try:
            self.abort()
        except Exception:
            logger.exception("Failed to retire metadata after an execution error; retaining the original exception")

    def abort(self) -> None:
        self._failed = True
        if torch.npu.is_current_stream_capturing():
            # Do not introduce graph-external wait_stream inside capture. Keep
            # the references until the outer capture finally can retire them.
            return
        if self.executor.submission_in_flight:
            # A producer can fail before recording its ready event. Joining the
            # stream avoids waiting on an event that may never be recorded.
            torch.npu.current_stream().wait_stream(self.executor.stream)
            self.executor.release()
        self._tasks = ()
        self._descriptor = None
