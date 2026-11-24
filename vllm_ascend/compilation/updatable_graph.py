# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Hashable, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, fields, replace
from typing import Any, Protocol, cast

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
        layer_name = getattr(_provider, "layer_name", None)
        return [
            {k: v for k, v in param.items() if k != "layer_name"}
            for param in self.params
            if layer_name == param.get("layer_name")
        ]


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
        runtime_kwargs = {k: params.get(k, v) for k, v in self.kwargs.items()}
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
        self._resolved_tasks: tuple[GraphUpdateTask, ...] | None = None
        self._resolved_base_key_counts: tuple[int, ...] = ()
        self._single_context_mode: bool = False
        self._provider_cache_specs: tuple[tuple[tuple[Any, ...], str] | None, ...] = ()

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
        self._resolved_tasks = None

    def _ensure_resolved_tasks(self) -> tuple[GraphUpdateTask, ...]:
        resolved = self._resolved_tasks
        if resolved is None:
            resolved = tuple(
                GraphUpdateTask(
                    t.operation,
                    t.kwargs.copy(),
                    t.provider,
                    t.provider_index,
                    t.handle,
                    t.event,
                )
                for t in self.tasks
            )
            self._resolved_tasks = resolved
            self._resolved_base_key_counts = tuple(len(t.kwargs) for t in self.tasks)
            self._single_context_mode = len(self.provider_sizes) == len(self.tasks) and all(
                size == 1 for size in self.provider_sizes.values()
            )
            specs: list[tuple[tuple[Any, ...], str] | None] = []
            for t in self.tasks:
                prov = t.provider
                dc_fields = getattr(prov, "__dataclass_fields__", None)
                layer_name = getattr(prov, "layer_name", None)
                if (
                    dc_fields is not None
                    and "layer_name" in dc_fields
                    and "metadata_layer_name" not in dc_fields
                    and isinstance(layer_name, str)
                ):
                    static_sig = (
                        type(prov),
                        tuple(
                            (f.name, getattr(prov, f.name)) for f in fields(cast(Any, prov)) if f.name != "layer_name"
                        ),
                    )
                    try:
                        hash(static_sig)
                        specs.append((static_sig, layer_name))
                    except TypeError:
                        specs.append(None)
                else:
                    specs.append(None)
            self._provider_cache_specs = tuple(specs)
        return resolved

    def resolve_tasks(
        self,
        source: ParamSource,
    ) -> tuple[GraphUpdateTask, ...]:
        if not isinstance(self, UpdatableGraph):
            params_by_provider = {provider: source.get(provider) for provider in self.provider_sizes}
            for provider, size in self.provider_sizes.items():
                assert len(params_by_provider[provider]) == size
            return tuple(task.bind(params_by_provider[task.provider][task.provider_index]) for task in self.tasks)
        resolved = self._ensure_resolved_tasks()
        if type(source) is ContextSource and self._single_context_mode:
            ctx = source.context
            if isinstance(ctx, dict):
                param_cache: dict[tuple[Any, int], Params] = {}
                for task, base_task, base_n, spec in zip(
                    resolved, self.tasks, self._resolved_base_key_counts, self._provider_cache_specs
                ):
                    meta = ctx.get(spec[1]) if spec is not None else None
                    if spec is not None and meta is not None:
                        cache_key = (spec[0], id(meta))
                        params = param_cache.get(cache_key)
                        if params is None:
                            params = task.provider.resolve(ctx)
                            param_cache[cache_key] = params
                    else:
                        params = task.provider.resolve(ctx)
                    if len(task.kwargs) != base_n:
                        task.kwargs = base_task.kwargs.copy()
                    task.kwargs.update(params)
                return resolved
            for task, base_task, base_n in zip(resolved, self.tasks, self._resolved_base_key_counts):
                params = task.provider.resolve(ctx)
                if len(task.kwargs) != base_n:
                    task.kwargs = base_task.kwargs.copy()
                task.kwargs.update(params)
            return resolved
        params_by_provider = {provider: source.get(provider) for provider in self.provider_sizes}
        for provider, size in self.provider_sizes.items():
            assert len(params_by_provider[provider]) == size
        for task, base_task, base_n in zip(resolved, self.tasks, self._resolved_base_key_counts):
            params = params_by_provider[task.provider][task.provider_index]
            if len(task.kwargs) != base_n:
                task.kwargs = base_task.kwargs.copy()
            task.kwargs.update(params)
        return resolved

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
