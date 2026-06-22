# SPDX-License-Identifier: Apache-2.0
"""Breakable ACL graph capture for eager Python kernels between NPU segments."""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available


def _sparse_attn_graph_safe_in_full() -> bool:
    """Return True when sparse attention may stay inside FULL NPUGraph segments.

    Ascend still uses Python torch fallbacks for indexer/sparse-attn; they must
    re-run every decode step and cannot be replayed from NPUGraph yet.
    """
    return False


class BreakableACLGraphCapture:
    """Multi-segment NPUGraph capture with eager breaks between segments."""

    _tls = threading.local()

    @classmethod
    def current(cls) -> BreakableACLGraphCapture | None:
        return getattr(cls._tls, "active", None)

    def __init__(self, pool: Any | None = None) -> None:
        self.pool = pool
        self.segments: list[Callable[[], Any]] = []
        self._current_graph: torch.npu.NPUGraph | None = None
        self._graph_ctx: AbstractContextManager[Any] | None = None
        self._capturing = False
        self._replaying = False
        self._num_graphs = 0
        self._num_eager_breaks = 0

    def __enter__(self) -> BreakableACLGraphCapture:
        if BreakableACLGraphCapture.current() is not None:
            raise RuntimeError("Nested BreakableACLGraphCapture is not supported.")
        BreakableACLGraphCapture._tls.active = self
        self._begin_segment()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._end_segment()
        finally:
            BreakableACLGraphCapture._tls.active = None

    def _begin_segment(self) -> None:
        assert not self._capturing
        graph = torch.npu.NPUGraph()
        self._current_graph = graph
        self._graph_ctx = torch.npu.graph(graph, pool=self.pool)
        self._graph_ctx.__enter__()
        self._capturing = True

    def _end_segment(self) -> None:
        if not self._capturing:
            return
        assert self._current_graph is not None
        assert self._graph_ctx is not None
        self._graph_ctx.__exit__(None, None, None)
        self.segments.append(self._current_graph.replay)
        self._num_graphs += 1
        self._current_graph = None
        self._graph_ctx = None
        self._capturing = False

    def add_eager(self, fn: Callable[[], Any]) -> Any:
        self._end_segment()
        torch.npu.current_stream().synchronize()
        result = fn()
        torch.npu.current_stream().synchronize()
        self.segments.append(fn)
        self._num_eager_breaks += 1
        self._begin_segment()
        return result

    def replay(self) -> None:
        BreakableACLGraphCapture._tls.active = self
        self._replaying = True
        try:
            for segment in self.segments:
                segment()
                torch.npu.current_stream().synchronize()
        finally:
            self._replaying = False
            BreakableACLGraphCapture._tls.active = None

    @property
    def num_graphs(self) -> int:
        return self._num_graphs

    @property
    def num_eager_breaks(self) -> int:
        return self._num_eager_breaks

    @property
    def replaying(self) -> bool:
        return self._replaying


def breakable_acl_graph_eager(fn: Callable) -> Callable:
    """Turn a custom-op kernel into an eager break during breakable capture.

    Aligns with vllm ``eager_break_during_capture``: PIECEWISE (prefill)
    breaks out to eager; FULL (decode) may stay inside the graph segment when
    ``_sparse_attn_graph_safe_in_full()`` is True.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        capture = BreakableACLGraphCapture.current()
        if capture is None or not capture._capturing:
            return fn(*args, **kwargs)

        if is_forward_context_available():
            mode = get_forward_context().cudagraph_runtime_mode
            if mode == CUDAGraphMode.FULL and _sparse_attn_graph_safe_in_full():
                return fn(*args, **kwargs)

        bound_args = args
        bound_kwargs = kwargs
        return capture.add_eager(lambda: fn(*bound_args, **bound_kwargs))

    return wrapper
