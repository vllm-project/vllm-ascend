# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from vllm_ascend.utils import weak_ref_tensors


class RuntimeParameterProvider(Protocol):
    def resolve(self, forward_context) -> dict[str, Any]: ...


_ACTIVE_GRAPH: ContextVar["UpdatableGraph | None"] = ContextVar(
    "capturing_updatable_graph", default=None
)


@dataclass(frozen=True, slots=True)
class GraphUpdateTask:
    operation: Callable[..., Any]
    kwargs: dict[str, Any]
    provider: RuntimeParameterProvider
    handle: Any
    event: Any

    def update(self, update_stream, forward_context) -> None:
        runtime_kwargs = {**self.kwargs, **self.provider.resolve(forward_context)}
        torch.npu.graph_task_update_begin(update_stream, self.handle)
        self.operation(**runtime_kwargs)
        torch.npu.graph_task_update_end(update_stream)
        self.event.record(update_stream)


class UpdatableGraph(torch.npu.NPUGraph):
    def __init__(self) -> None:
        super().__init__()
        self.update_tasks: list[GraphUpdateTask] = []

    def register_task(
        self,
        operation: Callable[..., Any],
        kwargs: dict[str, Any],
        provider: RuntimeParameterProvider,
    ) -> None:
        stream = torch.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        torch.npu.graph_task_group_begin(stream)
        operation(**kwargs)
        handle = torch.npu.graph_task_group_end(stream)
        weak_kwargs = weak_ref_tensors(kwargs)
        self.update_tasks.append(
            GraphUpdateTask(operation, weak_kwargs, provider, handle, event)
        )

    def update(self, update_stream, forward_context) -> None:
        with torch.npu.stream(update_stream):
            for task in self.update_tasks:
                task.update(update_stream, forward_context)


@contextmanager
def capture_updatable_graph(graph: UpdatableGraph):
    token = _ACTIVE_GRAPH.set(graph)
    try:
        yield
    finally:
        _ACTIVE_GRAPH.reset(token)


def register_task(
    operation: Callable[..., Any],
    kwargs: dict[str, Any],
    provider: RuntimeParameterProvider,
) -> None:
    graph = _ACTIVE_GRAPH.get()
    if graph is None:
        operation(**kwargs)
    else:
        graph.register_task(operation, kwargs, provider)
