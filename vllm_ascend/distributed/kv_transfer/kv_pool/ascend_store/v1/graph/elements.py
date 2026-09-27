"""Typed nodes and edges in the AscendStore KV computation graph."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash

from .coordinates import TokenRange


@dataclass(frozen=True, slots=True)
class KVChunk:
    """One content-identified semantic KV unit on the token axis."""

    group_id: int
    block_index: int
    token_range: TokenRange
    content_hash: BlockHash | str


@dataclass(frozen=True, slots=True)
class PhysicalCoordinate:
    """Axes that distinguish physical representations of one semantic chunk."""

    pp_rank: int | None = None
    dcp_rank: int | None = None
    head_rank: int | None = None
    effective_tp_rank: int | None = None
    consumer_pp_slice: int | None = None


@dataclass(frozen=True, slots=True)
class RemoteKVObject:
    """One concrete Backend representation of a semantic KV chunk."""

    chunk: KVChunk
    key: str
    coordinate: PhysicalCoordinate = PhysicalCoordinate()


@dataclass(frozen=True, slots=True)
class LocalKVSlice:
    """One concrete Worker-local representation of a semantic KV chunk."""

    chunk: KVChunk
    block_id: int
    addresses: tuple[int, ...]
    sizes: tuple[int, ...]
    coordinate: PhysicalCoordinate = PhysicalCoordinate()


@dataclass(frozen=True, slots=True)
class KVBinding:
    """One transfer edge between representations of the same semantic KV chunk."""

    remote_object: RemoteKVObject
    local_slice: LocalKVSlice

    def __post_init__(self) -> None:
        if self.remote_object.chunk != self.local_slice.chunk:
            raise ValueError("A KV binding must connect representations of the same semantic chunk")
        if self.remote_object.coordinate != self.local_slice.coordinate:
            raise ValueError("A KV binding must connect the same physical subrepresentation")


@dataclass(frozen=True, slots=True)
class RemoteObjectBatch:
    """Remote representations selected for one cache group and one invocation."""

    group_id: int
    chunks: tuple[KVChunk, ...]
    remote_objects: tuple[RemoteKVObject, ...]

    def __post_init__(self) -> None:
        chunks = set(self.chunks)
        if any(chunk.group_id != self.group_id for chunk in chunks):
            raise ValueError(f"Remote object batch contains chunks outside cache group {self.group_id}")
        if any(remote_object.chunk not in chunks for remote_object in self.remote_objects):
            raise ValueError("Remote object batch contains a Backend object without its semantic chunk")


@dataclass(frozen=True, slots=True)
class BindingBatch:
    """Object-memory bindings selected for one cache group and one invocation."""

    group_id: int
    bindings: tuple[KVBinding, ...]

    def __post_init__(self) -> None:
        if any(binding.remote_object.chunk.group_id != self.group_id for binding in self.bindings):
            raise ValueError(f"Binding batch contains bindings outside cache group {self.group_id}")
