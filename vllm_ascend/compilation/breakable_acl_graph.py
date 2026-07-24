# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Breakable ACL graph capture and replay.

This is the Ascend counterpart of vLLM's breakable CUDA graph. A model
forward is captured as alternating ACL graph and eager segments. Eager
segments must update caller-provided static output buffers so later graph
segments keep reading from the addresses used during capture.
"""

from __future__ import annotations

import functools
import gc
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, NoReturn, TypeVar

import torch
import vllm.envs as envs
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import logger
from vllm.platforms import current_platform

from vllm_ascend.compilation.acl_graph import (
    _is_old_hdk_capture_error,
    _is_stream_resource_capture_error,
    get_draft_graph_params,
    get_draft_graph_prefill_params,
    get_graph_params,
    weak_ref_workspaces,
)
from vllm_ascend.utils import weak_ref_tensor, weak_ref_tensors

F = TypeVar("F", bound=Callable[..., Any])


def is_breakable_aclgraph_enabled() -> bool:
    return bool(envs.VLLM_USE_BREAKABLE_CUDAGRAPH)


def eager_break_during_capture(fn: F) -> F:
    """Run a custom-op implementation eagerly during breakable capture.

    The decorated operation must write its result to a caller-provided output
    tensor. Returning a newly allocated tensor would make later graph segments
    depend on an address that is not updated during replay.
    """
    if not is_breakable_aclgraph_enabled():
        return fn

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        capture = BreakableACLGraphCapture.current()
        if capture is None or not capture.is_capturing:
            return fn(*args, **kwargs)

        if is_forward_context_available():
            mode = get_forward_context().cudagraph_runtime_mode
            if mode == CUDAGraphMode.FULL:
                return fn(*args, **kwargs)

        weak_args = tuple(weak_ref_tensor(arg) if isinstance(arg, torch.Tensor) else arg for arg in args)
        weak_kwargs = {
            key: weak_ref_tensor(value) if isinstance(value, torch.Tensor) else value for key, value in kwargs.items()
        }
        return capture.add_eager(lambda: fn(*weak_args, **weak_kwargs))

    return wrapper  # type: ignore[return-value]


class BreakableACLGraphCapture:
    """NPU stream-capture context that supports eager break points."""

    _tls = threading.local()

    @classmethod
    def current(cls) -> BreakableACLGraphCapture | None:
        return getattr(cls._tls, "active", None)

    @classmethod
    def is_active(cls) -> bool:
        return cls.current() is not None

    def __init__(self, pool: Any | None = None) -> None:
        self.pool = pool
        self.segments: list[Callable[[], Any]] = []
        self._num_graphs = 0
        self._num_eager_breaks = 0
        self._current_graph: torch.npu.NPUGraph | None = None
        self._capturing = False
        self.capture_stream = torch.npu.Stream()
        self._stream_context: Any | None = None

    @property
    def is_capturing(self) -> bool:
        return self._capturing

    def __enter__(self) -> BreakableACLGraphCapture:
        if self.current() is not None:
            raise RuntimeError("Nested BreakableACLGraphCapture is not supported.")

        BreakableACLGraphCapture._tls.active = self
        try:
            self._stream_context = torch.npu.stream(self.capture_stream)
            self._stream_context.__enter__()
            self._begin_segment()
        except BaseException as exc:
            if self._stream_context is not None:
                self._stream_context.__exit__(
                    type(exc),
                    exc,
                    exc.__traceback__,
                )
                self._stream_context = None
            BreakableACLGraphCapture._tls.active = None
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._end_segment()
        finally:
            try:
                if self._stream_context is not None:
                    self._stream_context.__exit__(exc_type, exc, tb)
                    self._stream_context = None
            finally:
                BreakableACLGraphCapture._tls.active = None

    def _begin_segment(self) -> None:
        assert not self._capturing
        graph = torch.npu.NPUGraph()
        if self.pool is None:
            graph.capture_begin()
        else:
            graph.capture_begin(pool=self.pool)
        self._current_graph = graph
        self._capturing = True

    def _end_segment(self) -> None:
        if not self._capturing:
            return

        assert self._current_graph is not None
        self._current_graph.capture_end()
        self.segments.append(self._current_graph.replay)
        self._num_graphs += 1
        self._current_graph = None
        self._capturing = False

    def add_eager(self, fn: Callable[[], Any]) -> Any:
        """End capture, run and record ``fn``, then start a new segment."""
        self._end_segment()
        result = fn()
        self.segments.append(fn)
        self._num_eager_breaks += 1
        self._begin_segment()
        return result

    def replay(self) -> None:
        for segment in self.segments:
            segment()

    @property
    def num_graphs(self) -> int:
        return self._num_graphs

    @property
    def num_eager_breaks(self) -> int:
        return self._num_eager_breaks

    def __repr__(self) -> str:
        return f"BreakableACLGraphCapture(graphs={self.num_graphs}, eager_breaks={self.num_eager_breaks})"


@dataclass
class BreakableACLGraphEntry:
    batch_descriptor: BatchDescriptor
    capture: BreakableACLGraphCapture | None = None
    output: Any | None = None
    input_addresses: list[int] | None = None


class BreakableACLGraphWrapper:
    """Capture a model forward as alternating ACL graph/eager segments."""

    _all_instances: ClassVar[weakref.WeakSet[BreakableACLGraphWrapper]] = weakref.WeakSet()

    @classmethod
    def clear_all_graphs(cls) -> None:
        for instance in list(cls._all_instances):
            instance.clear_graphs()

    def __init__(
        self,
        runnable: Callable[..., Any],
        vllm_config: VllmConfig,
    ) -> None:
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.graph_pool = current_platform.get_global_graph_pool()
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self._runnable_str = str(runnable) if self.is_debugging_mode else None
        self.concrete_aclgraph_entries: dict[BatchDescriptor, BreakableACLGraphEntry] = {}

        self._all_instances.add(self)
        logger.info_once("Breakable ACL graph enabled")

    def __getattr__(self, key: str) -> Any:
        runnable = self.__dict__.get("runnable")
        if runnable is not None and hasattr(runnable, key):
            return getattr(runnable, key)
        if self.__dict__.get("is_debugging_mode", False):
            raise AttributeError(
                f"Attribute {key} not exists in the runnable of breakable ACL graph wrapper: {self._runnable_str}"
            )
        raise AttributeError(key)

    def unwrap(self) -> Callable[..., Any]:
        return self.runnable

    @property
    def cudagraph_wrapper(self) -> BreakableACLGraphWrapper:
        return self

    def clear_graphs(self) -> None:
        self.concrete_aclgraph_entries.clear()

    @staticmethod
    def _collect_tensor_addresses(
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> list[int]:
        addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
        addresses.extend(value.data_ptr() for value in kwargs.values() if isinstance(value, torch.Tensor))
        return addresses

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not is_forward_context_available():
            return self.runnable(*args, **kwargs)

        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        runtime_mode = forward_context.cudagraph_runtime_mode
        if runtime_mode == CUDAGraphMode.NONE:
            return self.runnable(*args, **kwargs)

        assert batch_descriptor is not None
        entry = self.concrete_aclgraph_entries.get(batch_descriptor)
        if entry is None:
            entry = BreakableACLGraphEntry(batch_descriptor=batch_descriptor)
            self.concrete_aclgraph_entries[batch_descriptor] = entry

        if entry.capture is None:
            return self._capture(entry, args, kwargs)
        return self._replay(entry, args, kwargs)

    def _capture(
        self,
        entry: BreakableACLGraphEntry,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        validate_cudagraph_capturing_enabled()
        entry.input_addresses = self._collect_tensor_addresses(args, kwargs)

        if self.graph_pool is None:
            set_graph_pool_id(current_platform.graph_pool_handle())
        else:
            set_graph_pool_id(self.graph_pool)

        gc.collect()
        torch.npu.empty_cache()

        from vllm.model_executor.offloader.base import get_offloader

        offloader = get_offloader()
        offloader.sync_prev_onload()
        capture = BreakableACLGraphCapture(pool=self.graph_pool)
        forward_context = get_forward_context()
        previous_capturing = forward_context.capturing
        forward_context.capturing = True
        try:
            with capture:
                output = self.runnable(*args, **kwargs)
                offloader.join_after_forward()
                output = weak_ref_tensors(output)
        except RuntimeError as exc:
            self._raise_capture_error(exc)
        finally:
            forward_context.capturing = previous_capturing

        for params in (
            get_graph_params(),
            get_draft_graph_params(),
            get_draft_graph_prefill_params(),
        ):
            weak_ref_workspaces(params)

        entry.capture = capture
        entry.output = weak_ref_tensors(output)
        logger.debug(
            "Captured breakable ACL graph for %s: %r",
            entry.batch_descriptor,
            capture,
        )
        return output

    @staticmethod
    def _raise_capture_error(exc: RuntimeError) -> NoReturn:
        if _is_old_hdk_capture_error(exc):
            raise RuntimeError(
                "ACL graph capture failed with an old Ascend HDK/CANN stack "
                "signature (`Alloc sq cq fail`). Please upgrade Ascend HDK to "
                "25.5.1 or later and use the matching CANN stack.\n"
                f"Original error:\n{exc}"
            ) from exc
        if _is_stream_resource_capture_error(exc):
            raise RuntimeError(
                "ACL graph capture failed with a known stream-resource "
                "exhaustion signature. Consider reducing "
                "cudagraph_capture_sizes, lowering "
                "max_cudagraph_capture_size, or temporarily disabling graph "
                "mode to confirm the failure is capture-related.\n"
                f"Original error:\n{exc}"
            ) from exc
        raise exc

    def _replay(
        self,
        entry: BreakableACLGraphEntry,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if self.is_debugging_mode and entry.input_addresses is not None:
            new_addresses = self._collect_tensor_addresses(args, kwargs)
            assert new_addresses == entry.input_addresses, (
                "Input tensor addresses changed between capture and replay "
                f"for {entry.batch_descriptor}. Expected "
                f"{entry.input_addresses}, got {new_addresses}."
            )

        from vllm.model_executor.offloader.base import get_offloader

        get_offloader().sync_prev_onload()
        logger.info_once("Replaying breakable ACL graph")
        assert entry.capture is not None
        entry.capture.replay()
        return entry.output
