# SPDX-License-Identifier: Apache-2.0
"""Registered cache geometry, separate from model and transport policy.

Adapters supply complete semantic membership from the loaded model. Cache
registration supplies physical groups and tensors. Neither source alone proves
that every required model component was included.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, cast

from .layout import TensorLayout
from .pages import PageLayout
from .state_transfer import ComponentLayout

if TYPE_CHECKING:
    import torch
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass(frozen=True)
class StateComponentSpec:
    name: str
    projection_sizes: tuple[int, ...]
    shard_axis: int
    shape: tuple[int, ...]
    dtype: str
    wire_shard_axis: int
    replicated: bool = False
    history_axis: int | None = None
    history_rows: int | None = None


@dataclass(frozen=True)
class PagedComponentSpec:
    name: str
    tokens_per_row: int


@dataclass(frozen=True)
class PrivateComponentSpec:
    """Request-private state cannot silently disappear from a checkpoint.

    The accepted-state adapter must supply a boundary-specific export or a
    proof that native initialization suffices. Registration does not omit it.
    """

    name: str
    boundary_multiple: int
    resume_without_history: bool = False


ComponentSpecs = tuple[StateComponentSpec, ...] | PagedComponentSpec | PrivateComponentSpec


@dataclass(frozen=True)
class RegisteredState:
    layer_name: str
    group_id: int
    block_tokens: int
    spec: StateComponentSpec
    tensor: torch.Tensor
    layout: ComponentLayout

    @cached_property
    def checkpoint_layout(self) -> ComponentLayout:
        """Accepted history only; speculative scratch is not persistent state.

        This assumes native accepted-state postprocessing already moved the
        valid history to the front. Runner handoff must prove that operation
        finished before any source pointer is submitted to transport.
        """
        if self.spec.history_axis is None:
            return self.layout
        shape = list(self.layout.tensor.shape)
        shape[self.spec.history_axis] = cast(int, self.spec.history_rows)
        return ComponentLayout(
            TensorLayout(tuple(shape), self.layout.tensor.strides, self.layout.tensor.itemsize),
            self.layout.dtype,
            self.layout.projection_sizes,
            self.layout.shard_axis,
            self.layout.tp_size,
            self.layout.tp_rank,
            self.layout.replicated,
        )

    def slot(self, block_id: int) -> torch.Tensor:
        if type(block_id) is not int or not 0 < block_id < self.tensor.shape[0]:
            raise ValueError("state block must be a non-null registered slot")
        return self.tensor.select(0, block_id)


@dataclass(frozen=True)
class RegisteredPage:
    layer_name: str
    group_id: int
    spec: PagedComponentSpec
    tensor: torch.Tensor
    layout: PageLayout


@dataclass(frozen=True)
class RegisteredPrivate:
    layer_name: str
    group_id: int
    spec: PrivateComponentSpec
    tensors: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class RegisteredCache:
    states: tuple[RegisteredState, ...]
    pages: tuple[RegisteredPage, ...]
    private: tuple[RegisteredPrivate, ...]

    def semantic_schema(self) -> tuple[dict, ...]:
        """Topology-independent identity; excludes physical groups/block IDs.

        Physical layouts and Store shard granularity are negotiated separately.
        Full component membership remains part of this logical schema.
        """
        components = []
        for state in self.states:
            layout = state.checkpoint_layout
            components.append(
                dict(
                    layer=state.layer_name,
                    component=state.spec.name,
                    family="state",
                    projections=layout.projection_sizes,
                    dimensions=layout.logical_shape[1:],
                    dtype=layout.dtype,
                    replicated=layout.replicated,
                )
            )
        for page in self.pages:
            components.append(
                dict(
                    layer=page.layer_name,
                    component=page.spec.name,
                    family="paged",
                    row_bytes=page.layout.row_bytes,
                    tokens_per_row=page.layout.tokens_per_row,
                    dtype=str(page.tensor.dtype).removeprefix("torch."),
                    replicated=True,
                )
            )
        for private in self.private:
            components.append(
                dict(
                    layer=private.layer_name,
                    component=private.spec.name,
                    family="private",
                    boundary_multiple=private.spec.boundary_multiple,
                    resume_without_history=private.spec.resume_without_history,
                )
            )
        return tuple(sorted(components, key=lambda item: (item["layer"], item["component"])))


def register_cache(
    cache_config: KVCacheConfig,
    caches: Mapping[str, torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]],
    model_components: Mapping[str, ComponentSpecs],
    *,
    tp_size: int,
    tp_rank: int,
) -> RegisteredCache:
    """Build once from actual registration; no tensor values are read/copied.

    The framework cache configuration is authoritative for allocated members,
    including speculative layers. Physical caches and adapter semantics must
    cover that configuration exactly. This checks registration consistency,
    not omissions shared by the framework configuration and both mappings.
    """
    names = [name for group in cache_config.kv_cache_groups for name in group.layer_names]
    if len(names) != len(set(names)):
        raise ValueError("a cache layer is registered in multiple physical groups")
    if set(names) != set(caches) or set(names) != set(model_components):
        raise ValueError("model components and registered cache membership differ")
    states, pages, private = [], [], []
    for group_id, group in enumerate(cache_config.kv_cache_groups):
        for name in group.layer_names:
            native_spec = group.kv_cache_spec
            if hasattr(native_spec, "kv_cache_specs"):
                native_spec = native_spec.kv_cache_specs[name]
            raw = caches[name]
            tensors = tuple(raw) if isinstance(raw, (tuple, list)) else (raw,)
            semantics = model_components[name]
            if isinstance(semantics, PrivateComponentSpec):
                if semantics.boundary_multiple <= 0:
                    raise ValueError("private-state boundary unit must be positive")
                private.append(RegisteredPrivate(name, group_id, semantics, tensors))
                continue
            if isinstance(semantics, PagedComponentSpec):
                if len(tensors) != 1:
                    raise ValueError("paged component needs one registered tensor")
                tensor = tensors[0]
                ratio = semantics.tokens_per_row
                blocks = cache_config.num_blocks
                if (
                    ratio <= 0
                    or blocks <= 0
                    or tensor.ndim != 4
                    or tensor.shape[2] != 1
                    or tensor.stride(3) != 1
                    or tensor.shape[0] % blocks
                ):
                    raise ValueError("unsupported replicated page geometry")
                scale = tensor.shape[0] // blocks
                kernel_tokens = tensor.shape[1] * ratio
                if scale * kernel_tokens != native_spec.block_size:
                    raise ValueError("registered pages do not cover native block size")
                itemsize = tensor.element_size()
                page_layout = PageLayout(
                    native_spec.block_size,
                    kernel_tokens,
                    ratio,
                    tensor.shape[3] * itemsize,
                    tensor.stride(1) * itemsize,
                    tensor.stride(0) * itemsize,
                    scale * tensor.stride(0) * itemsize,
                )
                pages.append(RegisteredPage(name, group_id, semantics, tensor, page_layout))
                continue
            if not isinstance(semantics, tuple) or not semantics or len(semantics) != len(tensors):
                raise ValueError("state component count differs from native registration")
            if len({spec.name for spec in semantics}) != len(semantics):
                raise ValueError("duplicate component name within a state layer")
            for spec, tensor in zip(semantics, tensors):
                if tuple(tensor.shape[1:]) != spec.shape or str(tensor.dtype).removeprefix("torch.") != spec.dtype:
                    raise ValueError("registered state shape/dtype differs from model")
                if spec.history_axis is not None and (
                    not 0 <= spec.history_axis < len(spec.shape)
                    or spec.history_rows is None
                    or not 0 < spec.history_rows <= spec.shape[spec.history_axis]
                ):
                    raise ValueError("invalid accepted-history window")
                layout = ComponentLayout.from_tensor(
                    tensor.select(0, 0),
                    projection_sizes=spec.projection_sizes,
                    shard_axis=spec.shard_axis,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    replicated=spec.replicated,
                )
                states.append(RegisteredState(name, group_id, native_spec.block_size, spec, tensor, layout))
    return RegisteredCache(tuple(states), tuple(pages), tuple(private))
