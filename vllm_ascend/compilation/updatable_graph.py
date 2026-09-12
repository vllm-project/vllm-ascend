# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Hashable, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Any, Protocol

import torch
import torch_npu
from vllm.logger import logger

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


@dataclass(slots=True)
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

    def apply(self, update_stream, workspace_buffer) -> None:
        if self.operation == torch_npu._npu_paged_attention:
            ws_key = (
                self.kwargs["context_lens"].data_ptr(),
                tuple(self.kwargs["context_lens"].shape),
                self.kwargs["query"].shape,
                self.kwargs["query"].dtype,
                self.kwargs["key_cache"].shape,
                self.kwargs["key_cache"].dtype,
                self.kwargs["value_cache"].shape,
                self.kwargs["value_cache"].dtype,
                self.kwargs["block_table"].shape if self.kwargs["block_table"] is not None else None,
                self.kwargs["block_table"].dtype if self.kwargs["block_table"] is not None else None,
                self.kwargs["out"].shape,
                self.kwargs["out"].dtype,
                self.kwargs["num_kv_heads"],
                self.kwargs["num_heads"],
                self.kwargs["scale_value"],
            )
            if workspace_buffer.get(ws_key) is None:
                workspace_kwargs = self.kwargs.copy()
                workspace_kwargs.pop("workspace")
                workspace = torch_npu._npu_paged_attention_get_workspace(**workspace_kwargs)
                workspace_buffer[ws_key] = workspace
            self.kwargs["workspace"] = workspace_buffer[ws_key]

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
        use_max_workspace: bool = False,
    ) -> Any:
        if use_max_workspace:
            # Some models mix attention layer shapes under the same graph size.
            # During capture, keep the largest required workspace for that size.
            candidate_workspace = factory()
        if key not in self.capture_resources:
            self.capture_resources[key] = factory()
        if (
            use_max_workspace
            and candidate_workspace.numel() * candidate_workspace.element_size()
            > self.capture_resources[key].numel() * self.capture_resources[key].element_size()
        ):
            self.capture_resources[key] = candidate_workspace
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
        logger.debug_once("Updating host-side attention metadata with UpdatableGraph.")
        with torch.npu.stream(update_stream):
            # This is specially designed for PA.
            workspace_buffer: dict[Hashable, Any] = {}
            for task in resolved_tasks:
                task.apply(update_stream, workspace_buffer)


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
    use_max_workspace: bool = False,
) -> Any:
    graph = _ACTIVE_GRAPH.get()
    if graph is None:
        return factory()
    return graph.get_capture_resource(key, factory, use_max_workspace)
