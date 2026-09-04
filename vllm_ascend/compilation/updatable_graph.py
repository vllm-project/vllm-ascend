# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Hashable, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Any, Protocol

import torch

from vllm_ascend.utils import weak_ref_tensors

Params = dict[str, Any]


class ParamProvider(Protocol):
    def resolve(self, context) -> Params: ...


class ParamSource(Protocol):
    def get(
        self,
        provider: ParamProvider,
    ) -> Sequence[Params]: ...


@dataclass(frozen=True, slots=True)
class ContextSource:
    context: Any

    def get(
        self,
        provider: ParamProvider,
    ) -> Sequence[Params]:
        return (provider.resolve(self.context),)


@dataclass(frozen=True, slots=True)
class SharedSource:
    params: Sequence[Params]

    def get(
        self,
        _provider: ParamProvider,
    ) -> Sequence[Params]:
        return self.params


_ACTIVE_GRAPH: ContextVar["UpdatableGraph | None"] = ContextVar("capturing_updatable_graph", default=None)


@dataclass(frozen=True, slots=True)
class GraphUpdateTask:
    operation: Callable[..., Any]
    kwargs: dict[str, Any]
    provider: ParamProvider
    provider_index: int
    handle: Any
    event: Any

    def bind(self, params: Params) -> "GraphUpdateTask":
        runtime_kwargs = {**self.kwargs, **params}
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
        self.provider_sizes: dict[ParamProvider, int] = {}
        self.capture_resources: dict[Hashable, Any] = {}
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
            self.capture_resources = weak_ref_tensors(self.capture_resources)

    def get_capture_resource(
        self,
        key: Hashable,
        factory: Callable[[], Any],
    ) -> Any:
        if key not in self.capture_resources:
            self.capture_resources[key] = factory()
        return self.capture_resources[key]

    def register_task(
        self,
        operation: Callable[..., Any],
        kwargs: dict[str, Any],
        provider: ParamProvider,
    ) -> None:
        stream = torch.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        torch.npu.graph_task_group_begin(stream)
        operation(**kwargs)
        handle = torch.npu.graph_task_group_end(stream)
        weak_kwargs = weak_ref_tensors(kwargs)
        provider_index = self.provider_sizes.get(provider, 0)
        self.provider_sizes[provider] = provider_index + 1
        self.tasks.append(
            GraphUpdateTask(
                operation,
                weak_kwargs,
                provider,
                provider_index,
                handle,
                event,
            )
        )

    def resolve_tasks(
        self,
        source: ParamSource,
    ) -> tuple[GraphUpdateTask, ...]:
        params_by_provider = {provider: source.get(provider) for provider in self.provider_sizes}
        for provider, size in self.provider_sizes.items():
            assert len(params_by_provider[provider]) == size
        return tuple(task.bind(params_by_provider[task.provider][task.provider_index]) for task in self.tasks)

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
    provider: ParamProvider,
) -> None:
    graph = _ACTIVE_GRAPH.get()
    if graph is None:
        operation(**kwargs)
    else:
        graph.register_task(operation, kwargs, provider)


def get_capture_resource(
    key: Hashable,
    factory: Callable[[], Any],
) -> Any:
    graph = _ACTIVE_GRAPH.get()
    if graph is None:
        return factory()
    return graph.get_capture_resource(key, factory)
