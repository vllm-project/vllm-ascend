"""Domain representations carried between stages of the KV Pool program."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from vllm.v1.core.kv_cache_utils import BlockHash

from ...coordinates import TokenRange

# ===============================
# Semantic KV
# ===============================


@dataclass(frozen=True, slots=True)
class KVChunk:
    """One content-identified semantic KV unit on the token axis."""

    group_id: int
    block_index: int
    token_range: TokenRange
    content_hash: BlockHash | str


@dataclass(frozen=True, slots=True)
class KVChunkBatch:
    """Semantic KV chunks selected for one cache group and one invocation."""

    group_id: int
    logical_block_count: int
    chunks: tuple[KVChunk, ...]

    def __post_init__(self) -> None:
        if any(chunk.group_id != self.group_id for chunk in self.chunks):
            raise ValueError(f"KV chunk batch contains chunks outside cache group {self.group_id}")


# ===============================
# Remote Object Branch
# ===============================


@dataclass(frozen=True, slots=True)
class RemoteObjectKey:
    """Canonical Backend identity derived for one semantic KV chunk."""

    chunk: KVChunk
    base_key: str


@dataclass(frozen=True, slots=True)
class RemoteObjectKeyBatch:
    """Canonical Backend identities selected for one cache group and one invocation."""

    group_id: int
    keys: tuple[RemoteObjectKey, ...]

    def __post_init__(self) -> None:
        if any(item.chunk.group_id != self.group_id for item in self.keys):
            raise ValueError(f"Remote object key batch contains chunks outside cache group {self.group_id}")
        if len({item.chunk for item in self.keys}) != len(self.keys):
            raise ValueError("Remote object key batch contains duplicate semantic chunks")


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
class RemoteObjectBatch:
    """Remote representations selected for one cache group and one invocation."""

    group_id: int
    remote_objects: tuple[RemoteKVObject, ...]

    def __post_init__(self) -> None:
        if any(remote_object.chunk.group_id != self.group_id for remote_object in self.remote_objects):
            raise ValueError(f"Remote object batch contains objects outside cache group {self.group_id}")


# ===============================
# Local Memory Branch
# ===============================


@dataclass(frozen=True, slots=True)
class KVBlockAssignment:
    """Associate one semantic KV chunk with an existing local Block ID."""

    chunk: KVChunk
    block_id: int
    memory_token_count: int

    def __post_init__(self) -> None:
        if self.memory_token_count <= 0:
            raise ValueError("A KV block assignment must select a positive number of local tokens")


@dataclass(frozen=True, slots=True)
class KVBlockAssignmentBatch:
    """Block assignments selected for one cache group and one invocation."""

    group_id: int
    assignments: tuple[KVBlockAssignment, ...]

    def __post_init__(self) -> None:
        if any(item.chunk.group_id != self.group_id for item in self.assignments):
            raise ValueError(f"KV block assignment batch contains chunks outside cache group {self.group_id}")


@dataclass(frozen=True, slots=True)
class KVMemorySegment:
    """Registered memory geometry for one cache entry in one physical layer."""

    layer_name: str
    physical_layer_id: int
    base_address: int
    block_length: int
    block_stride: int
    bytes_per_token: int


KVMemoryGeometry: TypeAlias = Mapping[int, tuple[KVMemorySegment, ...]]
