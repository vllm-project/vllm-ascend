"""Domain representations carried between stages of the KV Pool program."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from vllm.v1.core.kv_cache_utils import BlockHash

from ..coordinates import TokenRange

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


@dataclass(frozen=True, slots=True)
class KVMemoryView:
    """A non-owning view of one KV region in Worker-local registered memory."""

    block_id: int
    addresses: tuple[int, ...]
    sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.addresses) != len(self.sizes):
            raise ValueError("A KV memory view must align every local address and size")
        if any(size <= 0 for size in self.sizes):
            raise ValueError("A KV memory view must contain positive transfer sizes")


# ===============================
# Executable Binding
# ===============================


@dataclass(frozen=True, slots=True)
class KVRegion:
    """A semantic KV chunk restricted to an explicit set of physical layers."""

    chunk: KVChunk
    physical_layer_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.physical_layer_ids:
            raise ValueError("A KV region must contain at least one physical layer")
        if tuple(sorted(set(self.physical_layer_ids))) != self.physical_layer_ids:
            raise ValueError("KV region physical layers must be unique and ordered")


@dataclass(frozen=True, slots=True)
class KVBinding:
    """Map one semantic region between a remote object and Worker-local memory."""

    region: KVRegion
    remote_object: RemoteKVObject
    remote_object_size: int
    remote_offsets: tuple[int, ...]
    memory: KVMemoryView

    def __post_init__(self) -> None:
        if self.region.chunk != self.remote_object.chunk:
            raise ValueError("A KV binding must belong to its remote object's semantic chunk")
        if self.remote_object_size <= 0:
            raise ValueError("A KV binding must identify a positive remote object size")
        if len(self.remote_offsets) != len(self.memory.addresses):
            raise ValueError("A KV binding must align every remote offset, local address and size")
        if any(
            offset < 0 or size <= 0 or offset + size > self.remote_object_size
            for offset, size in zip(self.remote_offsets, self.memory.sizes, strict=True)
        ):
            raise ValueError("A KV binding range must fit inside its remote object")


@dataclass(frozen=True, slots=True)
class BindingBatch:
    """Object-memory bindings selected for one cache group and one invocation."""

    group_id: int
    bindings: tuple[KVBinding, ...]

    def __post_init__(self) -> None:
        if any(binding.region.chunk.group_id != self.group_id for binding in self.bindings):
            raise ValueError(f"Binding batch contains bindings outside cache group {self.group_id}")
