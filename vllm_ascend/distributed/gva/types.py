# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-neutral descriptors for MemFabric GVA resources."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _non_negative(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


@dataclass(frozen=True, slots=True)
class GvaCapabilities:
    """Access modes and graph guarantees provided by one backend adapter."""

    direct_map: bool = False
    entry_gather: bool = False
    graph_safe: bool = False
    max_entry_bytes: int | None = None

    def __post_init__(self) -> None:
        if not (self.direct_map or self.entry_gather):
            raise ValueError("a GVA backend must provide direct_map or entry_gather")
        if self.max_entry_bytes is not None:
            object.__setattr__(self, "max_entry_bytes", _positive(self.max_entry_bytes, "max_entry_bytes"))


@dataclass(frozen=True, slots=True)
class GvaPoolSpec:
    """Static membership and capacity identity for one model-run pool."""

    pool_id: str
    generation: str
    api_family: str
    members: tuple[int, ...]
    capacity_bytes: int
    store_url: str
    schema: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.pool_id, "pool_id"),
            (self.generation, "generation"),
            (self.api_family, "api_family"),
            (self.store_url, "store_url"),
            (self.schema, "schema"),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty")
        members = tuple(self.members)
        if not members or any(isinstance(rank, bool) or not isinstance(rank, Integral) for rank in members):
            raise ValueError("members must contain integer rank IDs")
        if len(set(members)) != len(members):
            raise ValueError("members must contain unique rank IDs")
        if any(rank < 0 for rank in members):
            raise ValueError("members must contain non-negative rank IDs")
        object.__setattr__(self, "members", tuple(int(rank) for rank in members))
        object.__setattr__(self, "capacity_bytes", _positive(self.capacity_bytes, "capacity_bytes"))


@dataclass(frozen=True, slots=True)
class GvaPoolMember:
    """Rank-local device identity, separate from the shared pool descriptor."""

    rank: int
    device_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "rank", _non_negative(self.rank, "rank"))
        object.__setattr__(self, "device_id", _non_negative(self.device_id, "device_id"))


@dataclass(frozen=True, slots=True)
class GvaObjectSpec:
    """Logical object identity; table format remains model-owned."""

    object_id: str
    generation: str
    size_bytes: int
    schema: str
    checkpoint_identity: str
    layout_identity: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.object_id, "object_id"),
            (self.generation, "generation"),
            (self.schema, "schema"),
            (self.checkpoint_identity, "checkpoint_identity"),
            (self.layout_identity, "layout_identity"),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty")
        object.__setattr__(self, "size_bytes", _positive(self.size_bytes, "size_bytes"))


@dataclass(frozen=True, slots=True)
class GvaPoolRef:
    """Opaque pool identity exchanged between members."""

    pool_id: str
    generation: str
    api_family: str
    schema: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.pool_id, "pool_id"),
            (self.generation, "generation"),
            (self.api_family, "api_family"),
            (self.schema, "schema"),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, slots=True)
class GvaObjectRef:
    """Opaque logical object identity; no CPU/NPU pointer is exposed."""

    pool: GvaPoolRef
    object_id: str
    size_bytes: int
    schema: str
    checkpoint_identity: str
    layout_identity: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.object_id, "object_id"),
            (self.schema, "schema"),
            (self.checkpoint_identity, "checkpoint_identity"),
            (self.layout_identity, "layout_identity"),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty")
        object.__setattr__(self, "size_bytes", _positive(self.size_bytes, "size_bytes"))
