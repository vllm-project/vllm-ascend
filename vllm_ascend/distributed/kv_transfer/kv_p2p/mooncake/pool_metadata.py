# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Independent pool-PD handoff metadata; no remote HBM block IDs or addresses."""

from dataclasses import asdict, dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.key_codec import PoolKeyIdentity
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.restore_planner import CacheSlice, SourceCacheSlice


def _fields(data: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(data, dict) or data.keys() != expected:
        raise ValueError(f"Invalid {context} fields")
    return data


def _text(value: Any, context: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a nonempty string")


@dataclass(frozen=True)
class PoolObjectSlice:
    cache: CacheSlice
    offset: int


@dataclass(frozen=True)
class PoolObjectRef:
    profile_id: str
    group_id: int
    object_id: str
    head_or_tp_rank: int
    pp_rank: int
    size: int
    slices: tuple[PoolObjectSlice, ...]

    def __post_init__(self) -> None:
        _text(self.profile_id, "source profile ID")
        _text(self.object_id, "source object ID")
        for name in ("group_id", "head_or_tp_rank", "pp_rank"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Object {name} must be a nonnegative integer")
        if type(self.size) is not int or self.size <= 0 or not isinstance(self.slices, tuple) or not self.slices:
            raise ValueError("Pool objects require a positive size and immutable component slices")
        for item in self.slices:
            # Reuse planner bounds validation; a dummy key is local validation,
            # never a transmitted or looked-up object key.
            SourceCacheSlice(item.cache, self.object_id, self.size, item.offset)

    def resolve(self, identity: PoolKeyIdentity) -> tuple[SourceCacheSlice, ...]:
        key = identity.bind().make_full_key(self.group_id, self.object_id, self.head_or_tp_rank, self.pp_rank)
        return tuple(SourceCacheSlice(item.cache, key, self.size, item.offset) for item in self.slices)

    def to_wire(self) -> dict[str, Any]:
        # Avoid recursively copying every cache slice once in asdict(self)
        # only to discard and serialize it again below. Long prompts carry
        # one object per layer/component/page.
        return {
            "profile_id": self.profile_id,
            "group_id": self.group_id,
            "object_id": self.object_id,
            "head_or_tp_rank": self.head_or_tp_rank,
            "pp_rank": self.pp_rank,
            "size": self.size,
            "slices": [{"cache": asdict(item.cache), "offset": item.offset} for item in self.slices],
        }

    @classmethod
    def from_wire(cls, data: Any) -> "PoolObjectRef":
        data = _fields(
            data,
            {
                "profile_id",
                "group_id",
                "object_id",
                "head_or_tp_rank",
                "pp_rank",
                "size",
                "slices",
            },
            "pool object",
        )
        if not isinstance(data["slices"], list):
            raise ValueError("Object slices must be an array")
        slices = []
        for item in data["slices"]:
            item = _fields(item, {"cache", "offset"}, "pool component")
            cache = _fields(item["cache"], set(CacheSlice.__dataclass_fields__), "cache slice")
            slices.append(PoolObjectSlice(CacheSlice(**cache), item["offset"]))
        return cls(**(data | {"slices": tuple(slices)}))


def _objects_to_wire(objects: tuple[PoolObjectRef, ...]) -> dict[str, Any]:
    """Intern repeated layouts without changing object keys or byte ranges.

    Layout rows: profile, group, source rank, PP rank, size, component layouts.
    Object rows: layout index, object ID, [token start, token end, offset] rows.
    Only token intervals and object offsets vary within a component layout.
    """
    layouts: list[Any] = []
    indices: dict[tuple, int] = {}
    rows = []
    for item in objects:
        components = tuple(
            (
                part.cache.layer,
                part.cache.component,
                part.cache.format_id,
                part.cache.head_start,
                part.cache.head_end,
                part.cache.vector_bytes,
                part.cache.token_stride,
                part.cache.head_stride,
            )
            for part in item.slices
        )
        layout = (item.profile_id, item.group_id, item.head_or_tp_rank, item.pp_rank, item.size, components)
        index = indices.get(layout)
        if index is None:
            index = len(layouts)
            indices[layout] = index
            layouts.append([*layout[:5], [list(component) for component in components]])
        rows.append(
            [
                index,
                item.object_id,
                [[part.cache.token_start, part.cache.token_end, part.offset] for part in item.slices],
            ]
        )
    return {"encoding": "layout-table-v1", "layouts": layouts, "rows": rows}


def _objects_from_wire(data: Any) -> tuple[PoolObjectRef, ...]:
    # Accept the initial expanded representation when reading saved handoffs.
    if isinstance(data, list):
        return tuple(PoolObjectRef.from_wire(item) for item in data)
    data = _fields(data, {"encoding", "layouts", "rows"}, "pool object table")
    if data["encoding"] != "layout-table-v1":
        raise ValueError("Unsupported pool object table encoding")
    layouts, rows = data["layouts"], data["rows"]
    if not isinstance(layouts, list) or not isinstance(rows, list):
        raise ValueError("Pool object layouts and rows must be arrays")
    for layout in layouts:
        if not isinstance(layout, list) or len(layout) != 6:
            raise ValueError("Invalid pool object layout row")
        if not isinstance(layout[5], list) or not layout[5]:
            raise ValueError("Pool object layout requires components")
        if any(not isinstance(component, list) or len(component) != 8 for component in layout[5]):
            raise ValueError("Invalid pool component layout row")
    objects = []
    used: set[int] = set()
    for row in rows:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("Invalid pool object row")
        index, object_id, intervals = row
        if type(index) is not int or not 0 <= index < len(layouts):
            raise ValueError("Invalid pool object layout index")
        profile, group, rank, pp_rank, size, components = layouts[index]
        if not isinstance(intervals, list) or len(intervals) != len(components):
            raise ValueError("Pool object component counts differ")
        slices = []
        for component, interval in zip(components, intervals, strict=True):
            if not isinstance(interval, list) or len(interval) != 3:
                raise ValueError("Invalid pool component interval row")
            layer, name, fmt, head_start, head_end, width, token_stride, head_stride = component
            token_start, token_end, offset = interval
            cache = CacheSlice(
                layer, name, fmt, token_start, token_end, head_start, head_end, width, token_stride, head_stride
            )
            slices.append(PoolObjectSlice(cache, offset))
        objects.append(PoolObjectRef(profile, group, object_id, rank, pp_rank, size, tuple(slices)))
        used.add(index)
    if len(used) != len(layouts):
        raise ValueError("Unused pool object layout")
    return tuple(objects)


@dataclass(frozen=True)
class PoolSourceProfile:
    profile_id: str
    engine_incarnation: str
    key_identity: PoolKeyIdentity
    control_endpoint: str = ""
    group_block_sizes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _text(self.profile_id, "profile ID")
        _text(self.engine_incarnation, "engine incarnation")
        if not isinstance(self.key_identity, PoolKeyIdentity):
            raise ValueError("Source profile requires a validated key identity")
        if (
            not isinstance(self.group_block_sizes, tuple)
            or self.group_block_sizes
            and (
                len(self.group_block_sizes) != self.key_identity.num_groups
                or any(type(size) is not int or size <= 0 for size in self.group_block_sizes)
            )
        ):
            raise ValueError("Source profile requires one positive logical block size per cache group")
        if not isinstance(self.control_endpoint, str) or (
            self.control_endpoint and not self.control_endpoint.startswith("tcp://")
        ):
            raise ValueError("Source control endpoint must be TCP")

    def to_wire(self) -> dict[str, Any]:
        result = {
            "profile_id": self.profile_id,
            "engine_incarnation": self.engine_incarnation,
            "key_identity": self.key_identity.to_wire(),
            "control_endpoint": self.control_endpoint,
        }
        if self.group_block_sizes:
            result["group_block_sizes"] = list(self.group_block_sizes)
        return result

    @classmethod
    def from_wire(cls, data: Any) -> "PoolSourceProfile":
        extra = {"group_block_sizes"} if isinstance(data, dict) and "group_block_sizes" in data else set()
        data = _fields(
            data, {"profile_id", "engine_incarnation", "key_identity", "control_endpoint"} | extra, "source profile"
        )
        sizes = data.get("group_block_sizes", [])
        if not isinstance(sizes, list):
            raise ValueError("Source logical block sizes must be an array")
        return cls(
            data["profile_id"],
            data["engine_incarnation"],
            PoolKeyIdentity.from_wire(data["key_identity"]),
            data["control_endpoint"],
            tuple(sizes),
        )


@dataclass(frozen=True)
class PoolTransferMetadata:
    transfer_id: str
    attempt_id: str
    request_digest: str
    token_start: int
    token_end: int
    retention_ref: str
    profiles: tuple[PoolSourceProfile, ...]
    objects: tuple[PoolObjectRef, ...]
    version: int = 1
    status: str = "committed"

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1 or self.status != "committed":
            raise ValueError("Only version-1 committed pool transfers can be handed off")
        for name in ("transfer_id", "attempt_id", "request_digest", "retention_ref"):
            _text(getattr(self, name), name)
        if (
            any(type(value) is not int for value in (self.token_start, self.token_end))
            or not 0 <= self.token_start <= self.token_end
        ):
            raise ValueError("Invalid transfer token interval")
        if not isinstance(self.profiles, tuple) or not isinstance(self.objects, tuple):
            raise ValueError("Transfer profiles and objects must be immutable tuples")
        profiles = {profile.profile_id: profile for profile in self.profiles}
        if len(profiles) != len(self.profiles):
            raise ValueError("Duplicate source profile IDs")
        seen: set[tuple[str, int, str, int, int]] = set()
        for item in self.objects:
            if item.profile_id not in profiles:
                raise ValueError(f"Missing source profile: {item.profile_id}")
            identity = profiles[item.profile_id].key_identity
            if item.group_id >= identity.num_groups or item.pp_rank >= identity.pp_size:
                raise ValueError("Pool object coordinates disagree with its source identity")
            key = (item.profile_id, item.group_id, item.object_id, item.head_or_tp_rank, item.pp_rank)
            if key in seen:
                raise ValueError("Duplicate pool object reference")
            seen.add(key)
        if self.token_end > self.token_start and not self.objects:
            raise ValueError("Nonempty pool transfer requires source objects")

    def sources(self) -> list[SourceCacheSlice]:
        profiles = {profile.profile_id: profile for profile in self.profiles}
        return [cache for item in self.objects for cache in item.resolve(profiles[item.profile_id].key_identity)]

    def to_wire(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status,
            "transfer_id": self.transfer_id,
            "attempt_id": self.attempt_id,
            "request_digest": self.request_digest,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "retention_ref": self.retention_ref,
            "profiles": [profile.to_wire() for profile in self.profiles],
            "objects": _objects_to_wire(self.objects),
        }

    @classmethod
    def from_wire(cls, data: Any) -> "PoolTransferMetadata":
        data = _fields(
            data,
            {
                "version",
                "status",
                "transfer_id",
                "attempt_id",
                "request_digest",
                "token_start",
                "token_end",
                "retention_ref",
                "profiles",
                "objects",
            },
            "pool transfer",
        )
        if not isinstance(data["profiles"], list):
            raise ValueError("Transfer profiles must be an array")
        return cls(
            **(
                data
                | {
                    "profiles": tuple(PoolSourceProfile.from_wire(profile) for profile in data["profiles"]),
                    "objects": _objects_from_wire(data["objects"]),
                }
            )
        )


@dataclass(frozen=True)
class PoolWorkerReport:
    transfer_id: str
    attempt_id: str
    worker_id: str
    committed: bool
    objects: tuple[PoolObjectRef, ...] = ()
    error: str = ""

    def __post_init__(self) -> None:
        for value in (self.transfer_id, self.attempt_id, self.worker_id):
            _text(value, "worker report identity")
        if type(self.committed) is not bool or not isinstance(self.objects, tuple):
            raise ValueError("Invalid worker report status or objects")
        if self.committed and self.error:
            raise ValueError("Committed worker report cannot include an error")
        if not self.committed and not self.error:
            raise ValueError("Failed worker report requires a diagnostic error")

    def to_wire(self) -> dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "attempt_id": self.attempt_id,
            "worker_id": self.worker_id,
            "committed": self.committed,
            "objects": _objects_to_wire(self.objects),
            "error": self.error,
        }

    @classmethod
    def from_wire(cls, data: Any) -> "PoolWorkerReport":
        data = _fields(data, set(cls.__dataclass_fields__), "worker report")
        if not isinstance(data["error"], str):
            raise ValueError("Invalid worker report payload")
        return cls(**(data | {"objects": _objects_from_wire(data["objects"])}))


@dataclass(frozen=True)
class PoolWorkerHandshake(KVConnectorHandshakeMetadata):
    worker_id: str
    profile: PoolSourceProfile
    control_endpoint: str

    def __post_init__(self) -> None:
        _text(self.worker_id, "worker ID")
        _text(self.control_endpoint, "worker control endpoint")
        if not isinstance(self.profile, PoolSourceProfile):
            raise ValueError("Worker handshake requires a validated source profile")


@dataclass
class PoolWorkerMetadata(KVConnectorWorkerMetadata):
    reports: tuple[PoolWorkerReport, ...] = ()

    def aggregate(self, other: KVConnectorWorkerMetadata) -> "PoolWorkerMetadata":
        if not isinstance(other, PoolWorkerMetadata):
            raise TypeError("Cannot aggregate another connector's worker metadata")
        return PoolWorkerMetadata(self.reports + other.reports)


class PoolCommitTracker:
    def __init__(self, transfer_id: str, attempt_id: str, expected_workers: frozenset[str]):
        _text(transfer_id, "transfer ID")
        _text(attempt_id, "attempt ID")
        if not expected_workers or any(not isinstance(worker, str) or not worker for worker in expected_workers):
            raise ValueError("Pool commit requires an explicit expected worker set")
        self.transfer_id = transfer_id
        self.attempt_id = attempt_id
        self.expected_workers = frozenset(expected_workers)
        self.reports: dict[str, PoolWorkerReport] = {}

    def record(self, report: PoolWorkerReport) -> bool:
        if (report.transfer_id, report.attempt_id) != (self.transfer_id, self.attempt_id):
            return False  # Delayed completion from a previous attempt.
        if report.worker_id not in self.expected_workers:
            raise ValueError(f"Unexpected pool worker: {report.worker_id}")
        previous = self.reports.get(report.worker_id)
        if previous is not None and previous != report:
            raise ValueError("Conflicting pool worker completion reports")
        self.reports[report.worker_id] = report
        return True

    @property
    def committed(self) -> bool:
        return self.reports.keys() == self.expected_workers and all(
            report.committed for report in self.reports.values()
        )

    @property
    def failure(self) -> str | None:
        return next((report.error for report in self.reports.values() if not report.committed), None)


@dataclass(frozen=True)
class PoolSaveRequest:
    transfer_id: str
    attempt_id: str
    request_digest: str
    token_start: int
    token_end: int
    block_ids: tuple[tuple[int, ...], ...]
    group_hashes: tuple[tuple[str, ...], ...]
    chunk_start: int = 0
    is_final: bool = True

    def __post_init__(self) -> None:
        for name in ("transfer_id", "attempt_id", "request_digest"):
            _text(getattr(self, name), name)
        if type(self.chunk_start) is not int or not 0 <= self.chunk_start < self.token_end:
            raise ValueError("Save request requires a valid prefill chunk start")
        if type(self.is_final) is not bool:
            raise ValueError("Save request final-chunk flag must be boolean")
        if (
            any(type(value) is not int for value in (self.token_start, self.token_end))
            or not 0 <= self.token_start < self.token_end
        ):
            raise ValueError("Save request requires a nonempty valid token range")
        if not isinstance(self.block_ids, tuple) or not isinstance(self.group_hashes, tuple):
            raise ValueError("Save request group tables must be immutable")
        if len(self.block_ids) != len(self.group_hashes) or not self.block_ids:
            raise ValueError("Save request block/hash group counts differ")
        for blocks, hashes in zip(self.block_ids, self.group_hashes, strict=True):
            if not isinstance(blocks, tuple) or not isinstance(hashes, tuple):
                raise ValueError("Save request per-group tables must be immutable")
            if any(type(block) is not int or block < 0 for block in blocks):
                raise ValueError("Save request contains invalid block IDs")
            if any(not isinstance(value, str) or not value for value in hashes):
                raise ValueError("Save request contains invalid source hashes")


class LayerwisePoolConnectorMetadata(KVConnectorMetadata):
    def __init__(self) -> None:
        self.saves: dict[str, PoolSaveRequest] = {}
        self.loads: dict[str, PoolTransferMetadata] = {}
        self.local_block_ids: dict[str, tuple[list[int], ...]] = {}
        self.cancelled: set[str] = set()
        self.prefix_group_ids: dict[str, tuple[int, ...]] = {}
        self.released_sources: set[str] = set()

    def add_load(self, request_id: str, params: dict[str, Any], block_ids: tuple[list[int], ...]) -> None:
        _text(request_id, "request ID")
        transfer = PoolTransferMetadata.from_wire(params["pool_transfer"])
        if request_id in self.loads:
            raise ValueError("Pool request was added twice in one scheduling step")
        self.loads[request_id] = transfer
        self.local_block_ids[request_id] = tuple(list(group) for group in block_ids)
