# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Any, Protocol

import torch

from vllm_ascend.utils import weak_ref_tensors


class RuntimeParameterProvider(Protocol):
    def resolve(self, runtime_context) -> dict[str, Any]: ...


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

    def resolve(self, runtime_context) -> "GraphUpdateTask":
        runtime_kwargs = {**self.kwargs, **self.provider.resolve(runtime_context)}
        return replace(self, kwargs=runtime_kwargs)

    def apply(self, update_stream) -> None:
        torch.npu.graph_task_update_begin(update_stream, self.handle)
        self.operation(**self.kwargs)
        torch.npu.graph_task_update_end(update_stream)
        self.event.record(update_stream)


class UpdatableGraph(torch.npu.NPUGraph):
    def __init__(self) -> None:
        super().__init__()
        self.tasks: list[GraphUpdateTask] = []
        self.capture_token: Token[UpdatableGraph | None] | None = None

    def capture_begin(self, pool=None, capture_error_mode: str = "global") -> None:
        super().capture_begin(pool=pool, capture_error_mode=capture_error_mode)
        assert self.capture_token is None
        self.capture_token = _ACTIVE_GRAPH.set(self)

    def capture_end(self) -> None:
        try:
            super().capture_end()
        finally:
            assert self.capture_token is not None
            _ACTIVE_GRAPH.reset(self.capture_token)
            self.capture_token = None

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
        self.tasks.append(
            GraphUpdateTask(operation, weak_kwargs, provider, handle, event)
        )

    def resolve_tasks(self, runtime_context) -> tuple[GraphUpdateTask, ...]:
        return tuple(task.resolve(runtime_context) for task in self.tasks)

    def update(
        self,
        update_stream,
        resolved_tasks: tuple[GraphUpdateTask, ...],
    ) -> None:
        with torch.npu.stream(update_stream):
            for task in resolved_tasks:
                task.apply(update_stream)


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
