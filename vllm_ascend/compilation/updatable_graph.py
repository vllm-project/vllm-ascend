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
        _index: int,
    ) -> Params:
        return provider.resolve(self.context)


@dataclass(frozen=True, slots=True)
class SharedSource:
    params: Sequence[Params]

    def get(
        self,
        _provider: ParamProvider,
        index: int,
    ) -> Params:
        # 草稿模型必须连续注册
        return self.params[index]


_ACTIVE_GRAPH: ContextVar["UpdatableGraph | None"] = ContextVar("capturing_updatable_graph", default=None)


@dataclass(slots=True)
class GraphUpdateTask:
    operation: Callable[..., Any]
    kwargs: dict[str, Any]
    provider: ParamProvider
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
        use_max_workspace: bool = False,
    ) -> Any:
        if key not in self.capture_resources:
            self.capture_resources[key] = factory()
        if use_max_workspace:
            # Some models mix attention layer shapes under the same graph size.
            # During capture, keep the largest required workspace for that size.
            candidate_workspace = factory()
            if (
                candidate_workspace.numel() * candidate_workspace.element_size()
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
        # provider_index = self.provider_sizes.get(provider, 0)
        # self.provider_sizes[provider] = provider_index + 1
        self.tasks.append(
            GraphUpdateTask(
                operation,
                weak_kwargs,
                provider,
                handle,
                event,
            )
        )

    def resolve_tasks(
        self,
        source: ParamSource,
    ) -> tuple[GraphUpdateTask, ...]:
        """
        SharedSource
        providers           [p0,p1,p2,p3,p0,p1,p2,p3]                               # 混合注意力 f1,f2,g1,f3  update_params: u1,u2,u3 草稿模型不能是这种类型
        self.tasks          [t1,t2,t3,t4,t5,t6,t7,t8]
        source              update_params       [u0,u1,u2,u3,u4,u5,u6,u7]

        ContextSource
        providers           [p0,p1,p2,p3,p4,p5,p6,p7]
        self.tasks          [t0,t1,t2,t3,t4,t5,t6,t7]
        source              attn_metadata       [a0,a1,a2,a3,a4,a5,a6,a7]

        """

        return tuple(task.bind(source.get(task.provider, index)) for index,task in enumerate(self.tasks))

    def update(
        self,
        update_stream,
        resolved_tasks: tuple[GraphUpdateTask, ...],
    ) -> None:
        logger.debug_once("Updating host-side attention metadata with UpdatableGraph.")
        with torch.npu.stream(update_stream):
            # This is specially designed for PA.
            ws_buffer: dict[Hashable, Any] = {}
            for task in resolved_tasks:
                if task.operation == torch_npu._npu_paged_attention:
                    ws_key = _get_ws_key(task.kwargs)
                    if ws_buffer.get(ws_key) is None:
                        ws_kwargs = task.kwargs.copy()
                        ws_kwargs.pop("workspace")
                        workspace = torch_npu._npu_paged_attention_get_workspace(**ws_kwargs)
                        ws_buffer[ws_key] = workspace
                    task.kwargs["workspace"] = ws_buffer[ws_key]
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
    use_max_workspace: bool = False,
) -> Any:
    graph = _ACTIVE_GRAPH.get()
    if graph is None:
        return factory()
    return graph.get_capture_resource(key, factory, use_max_workspace)


def _get_ws_key(kwargs):
    def _sig(t: Any) -> tuple | None:
        if t is None:
            return (None, None)
        return (t.shape, t.dtype)

    return (
        kwargs["context_lens"].data_ptr(),
        tuple(kwargs["context_lens"].shape),
        _sig(kwargs["query"]),
        _sig(kwargs["key_cache"]),
        _sig(kwargs["value_cache"]),
        _sig(kwargs["block_table"]),
        _sig(kwargs["out"]),
        kwargs["num_kv_heads"],
        kwargs["num_heads"],
        kwargs["scale_value"],
    )
