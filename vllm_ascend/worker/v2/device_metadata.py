# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 metadata lifecycle: graph-side producers for FULL, side-stream eager."""

from contextlib import ExitStack, contextmanager

import torch

from vllm_ascend.worker.device_metadata import DeviceMetadataExecutor, use_device_metadata_executor


class TargetDeviceMetadata:
    """Target-owned producer stream, isolated from DSpark's metadata builders.

    FULL warmup/capture runs producers in ModelWithContext.forward; replay runs
    those captured nodes, not Python builders. Inputs/outputs remain in the
    builders' persistent padded buffers. Eager execution submits after prepare.
    Every producer is joined before the next async step may reuse its inputs.
    """

    def __init__(self):
        self.executor = DeviceMetadataExecutor(capture_producers=True)
        self._tasks = ()
        self._failed = False

    @contextmanager
    def activate(self):
        with use_device_metadata_executor(self.executor):
            try:
                yield
            finally:
                self.finish()

    def run_build(self, build_fn, **kwargs):
        full_graph = kwargs.get("full_graph_mode", False) or kwargs.get("for_cudagraph_capture", False)
        with self.build(kwargs["attn_groups"], full_graph):
            return build_fn(**kwargs)

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
