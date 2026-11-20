# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 state handoff contract. No torch, NPU, or transport dependency.

The transport owns registration, READ submission and completion aggregation.
This module owns the processed-token boundary, pool identity, byte ranges and
source leases. A lease cannot expire while a receiver may still issue DMA.
"""

from dataclasses import asdict, dataclass, field
from math import prod
from typing import Any

from vllm_ascend.core.kv_cache_pool_plan import RequestStatePoolPlan

STATE_PROTOCOL_VERSION = 1
WINDOW_ROWS = 128
COMPRESSOR_ROWS = 32
STATE_ACQUIRE_MSG = b"dsv41_state_acquire_v1"
STATE_DONE_MSG = b"dsv41_state_done_v1"
STATE_META_MSG = b"dsv41_state_meta_v1"
# Bound newly combined descriptors; this is not a claimed SDK maximum.
MAX_COALESCED_RANGE_BYTES = 16 * 1024 * 1024


def cache_kind(name: str) -> str:
    if name.endswith(".swa_cache"):
        return "window"
    if name.endswith(".compressor.state_cache"):
        return "compressor"
    if name.endswith((".long_kv_cache", ".indexer.k_cache")):
        return "history"
    raise ValueError(f"Unsupported DSV4.1 cache {name!r}")


def is_state_config(config: Any) -> bool:
    """Recognize the explicit three-pool configuration, not native DSV4 KV."""
    return isinstance(getattr(config, "request_state_plan", None), RequestStatePoolPlan)


def validate_state_tp_pair(producer_tp: int, consumer_tp: int) -> None:
    if (
        type(producer_tp) is not int
        or type(consumer_tp) is not int
        or producer_tp < 1
        or consumer_tp < 1
        or (producer_tp != consumer_tp and (producer_tp, consumer_tp) != (4, 2))
    ):
        raise ValueError("State PD requires equal TP or the supported TP4-to-TP2 configuration")


def state_source_rank(producer_tp: int, consumer_tp: int, consumer_rank: int) -> int:
    """Select one full producer replica for each DSV4.1 consumer rank."""
    validate_state_tp_pair(producer_tp, consumer_tp)
    if type(consumer_rank) is not int or not 0 <= consumer_rank < consumer_tp:
        raise ValueError("Invalid state consumer TP rank")
    # MLA/SWA, indexer K/scales and compressor projections are replicated.
    # Keep one source rank across all planes. TP4->TP2 uses producer 0 and 2.
    return consumer_rank * (producer_tp // consumer_tp)


def validate_pd_config(config: Any) -> str:
    """Constrain the initial path; the explicit identity must name the weights."""
    transfer = config.kv_transfer_config
    if transfer is None:
        return ""
    if transfer.kv_connector != "MooncakePullConnector":
        raise ValueError("DSV4.1 state PD requires MooncakePullConnector")
    if transfer.kv_role not in ("kv_producer", "kv_consumer"):
        raise ValueError("Configure separate Mooncake producer and consumer roles for state PD")
    if (config.additional_config or {}).get("enable_kvpp", False):
        raise ValueError("State PD does not support KVPP")
    identity = transfer.get_from_extra_config("state_model_id", "")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("state_model_id must identify the identical P/D weight checkpoint")
    parallel = config.parallel_config
    if (
        parallel.pipeline_parallel_size,
        parallel.prefill_context_parallel_size,
        parallel.decode_context_parallel_size,
    ) != (1, 1, 1):
        raise ValueError("State PD initially requires PP=PCP=DCP=1; DSA internal CP is separate")
    if config.speculative_config is not None or config.cache_config.enable_prefix_caching:
        raise ValueError("State PD requires prefix caching and speculative decoding disabled")
    if transfer.kv_load_failure_policy != "fail":
        raise ValueError("State PD requires kv_load_failure_policy='fail'; no silent D recompute")
    if getattr(config.model_config, "enable_sleep_mode", False):
        raise ValueError("State PD requires persistent registered backing; sleep mode is unsupported")
    return identity


@dataclass(frozen=True)
class StateSnapshot:
    version: int
    transfer_id: str
    source_request_id: str
    model_id: str
    boundary: int
    original_prompt_tokens: int
    tp_size: int
    window_valid: int
    window_next: int
    compressor_valid: int
    compressor_next: int
    compressed_completed: int
    compressed_residual: int

    @classmethod
    def create(
        cls, transfer_id: str, request_id: str, model_id: str, boundary: int, original_prompt_tokens: int, tp_size: int
    ) -> "StateSnapshot":
        value = cls(
            STATE_PROTOCOL_VERSION,
            transfer_id,
            request_id,
            model_id,
            boundary,
            original_prompt_tokens,
            tp_size,
            min(boundary, WINDOW_ROWS),
            boundary % WINDOW_ROWS,
            min(boundary, COMPRESSOR_ROWS),
            boundary % COMPRESSOR_ROWS,
            boundary // 2,
            boundary % 2,
        )
        value.validate()
        return value

    @classmethod
    def from_wire(cls, value: dict[str, Any]) -> "StateSnapshot":
        result = cls(**value)
        result.validate()
        return result

    def validate(self) -> None:
        identities = (self.transfer_id, self.source_request_id, self.model_id)
        if (
            type(self.version) is not int
            or self.version != STATE_PROTOCOL_VERSION
            or any(not isinstance(v, str) or not v for v in identities)
        ):
            raise ValueError("Invalid state snapshot identity or protocol version")
        values = (
            self.boundary,
            self.original_prompt_tokens,
            self.tp_size,
            self.window_valid,
            self.window_next,
            self.compressor_valid,
            self.compressor_next,
            self.compressed_completed,
            self.compressed_residual,
        )
        if any(type(v) is not int for v in values):
            raise ValueError("State snapshot counters must be integers")
        if self.boundary < 1 or self.original_prompt_tokens != self.boundary + 1 or self.tp_size < 1:
            raise ValueError("State PD uses T=N-1 with N>=2; the last prompt token remains on D")
        expected = (
            min(self.boundary, WINDOW_ROWS),
            self.boundary % WINDOW_ROWS,
            min(self.boundary, COMPRESSOR_ROWS),
            self.boundary % COMPRESSOR_ROWS,
            self.boundary // 2,
            self.boundary % 2,
        )
        if values[3:] != expected:
            raise ValueError("State counters do not describe the same processed-token boundary")

    def to_wire(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CachePlane:
    name: str
    cache_index: int
    kind: str
    group: int
    capacity: int
    rows: int
    row_bytes: int
    block_stride: int
    tokens_per_row: int
    dtype: str
    shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))

    def validate(self) -> None:
        expected_group = {"history": 0, "window": 1, "compressor": 2}
        if self.kind != cache_kind(self.name) or self.group != expected_group[self.kind]:
            raise ValueError("Cache plane uses the wrong resource pool")
        if any(
            type(v) is not int or v < 1
            for v in (self.capacity, self.rows, self.row_bytes, self.block_stride, self.tokens_per_row)
        ):
            raise ValueError("Invalid cache geometry")
        if self.cache_index < 0 or self.block_stride < self.rows * self.row_bytes:
            raise ValueError("Cache rows overlap or have an invalid view index")
        if self.kind == "window" and self.rows != WINDOW_ROWS:
            raise ValueError("Unexpected SWA ring size")
        if self.kind == "compressor" and self.rows != COMPRESSOR_ROWS:
            raise ValueError("Unexpected compressor ring size")
        itemsize = {"bfloat16": 2, "float16": 2, "float32": 4, "int8": 1}.get(self.dtype.removeprefix("torch."))
        if itemsize is None:
            raise ValueError("Unsupported state-transfer cache precision")
        if self.shape and (
            self.shape[0] != self.rows
            or any(type(n) is not int or n < 1 for n in self.shape)
            or prod(self.shape) * itemsize != self.rows * self.row_bytes
        ):
            raise ValueError("Cache shape and byte geometry disagree")

    @property
    def key(self) -> tuple[str, int]:
        return self.name, self.cache_index

    @property
    def format(self) -> tuple:
        # Counts can differ between P and D. Layout and dtype cannot.
        return (
            self.name,
            self.cache_index,
            self.kind,
            self.group,
            self.rows,
            self.row_bytes,
            self.block_stride,
            self.tokens_per_row,
            self.dtype,
            self.shape,
        )


def decode_planes(layout: dict[str, Any]) -> tuple[CachePlane, ...]:
    if layout.get("version") != STATE_PROTOCOL_VERSION:
        raise ValueError("Missing or incompatible DSV4.1 state layout")
    planes = tuple(CachePlane(**p) for p in layout["planes"])
    for plane in planes:
        plane.validate()
    if len({p.key for p in planes}) != len(planes) or {p.group for p in planes} != {0, 1, 2}:
        raise ValueError("State layout must uniquely describe all three resource pools")
    return planes


@dataclass(frozen=True)
class CopyRange:
    name: str
    cache_index: int
    source_offset: int
    destination_offset: int
    nbytes: int


def coalesce_copy_ranges(
    ranges: tuple[CopyRange, ...], max_range_bytes: int = MAX_COALESCED_RANGE_BYTES
) -> tuple[CopyRange, ...]:
    """Combine byte-adjacent mappings within one request and registered plane.

    Both source and destination must be contiguous in the same direction.
    Descending block allocation is common, so contiguous reverse runs are
    supported too. Never sort, bridge padding, or enlarge the byte coverage.
    Existing descriptors larger than the cap are retained without expansion.
    """
    if max_range_bytes <= 0:
        raise ValueError("Coalesced range limit must be positive")
    result: list[CopyRange] = []
    for current in ranges:
        if current.nbytes <= 0 or min(current.source_offset, current.destination_offset) < 0:
            raise ValueError("Invalid copy range")
        if result:
            previous = result[-1]
            size = previous.nbytes + current.nbytes
            same_plane = (previous.name, previous.cache_index) == (current.name, current.cache_index)
            if same_plane and size <= max_range_bytes:
                forward = (
                    previous.source_offset + previous.nbytes == current.source_offset
                    and previous.destination_offset + previous.nbytes == current.destination_offset
                )
                reverse = (
                    current.source_offset + current.nbytes == previous.source_offset
                    and current.destination_offset + current.nbytes == previous.destination_offset
                )
                if forward or reverse:
                    first = previous if forward else current
                    result[-1] = CopyRange(
                        first.name, first.cache_index, first.source_offset, first.destination_offset, size
                    )
                    continue
        result.append(current)
    return tuple(result)


def validate_ids(ids: Any, capacity: int, required: int, *, single: bool = False) -> tuple[int, ...]:
    values = tuple(ids)
    if len(values) != 1 if single else len(values) < required:
        raise ValueError("Missing request resources")
    if any(type(v) is not int or not 0 < v < capacity for v in values) or len(set(values)) != len(values):
        raise ValueError("Invalid, duplicate, or null resource IDs")
    return values[:required]


def build_copy_plan(
    snapshot: StateSnapshot,
    local_layout: dict[str, Any],
    remote_layout: dict[str, Any],
    local_ids: Any,
    remote_ids: Any,
) -> tuple[CopyRange, ...]:
    """Validate the complete handoff before submitting READ."""
    snapshot.validate()
    if remote_layout.get("tp_size") != snapshot.tp_size:
        raise ValueError("P metadata disagrees with the published snapshot TP size")
    validate_state_tp_pair(snapshot.tp_size, local_layout.get("tp_size"))
    fingerprint = local_layout.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Missing local model fingerprint")
    for layout in (local_layout, remote_layout):
        if layout.get("model_id") != snapshot.model_id:
            raise ValueError("P/D model identity mismatch")
        if layout.get("fingerprint") != fingerprint:
            raise ValueError("P/D model fingerprint or precision mismatch")
    local, remote = decode_planes(local_layout), decode_planes(remote_layout)
    remote_by_key = {p.key: p for p in remote}
    if {p.key for p in local} != set(remote_by_key) or len(local_ids) != 3 or len(remote_ids) != 3:
        raise ValueError("Incomplete layers or resource groups")
    plan = []
    for dst in local:
        src = remote_by_key[dst.key]
        if dst.format != src.format:
            raise ValueError(f"P/D cache format mismatch for {dst.key}")
        if dst.kind == "history":
            valid_rows = snapshot.boundary // dst.tokens_per_row
            pages = (valid_rows + dst.rows - 1) // dst.rows
            destinations = validate_ids(local_ids[dst.group], dst.capacity, pages)
            sources = validate_ids(remote_ids[src.group], src.capacity, pages)
            for page, (source, destination) in enumerate(zip(sources, destinations)):
                size = min(dst.rows, valid_rows - page * dst.rows) * dst.row_bytes
                plan.append(
                    CopyRange(
                        dst.name, dst.cache_index, source * src.block_stride, destination * dst.block_stride, size
                    )
                )
        else:
            source = validate_ids(remote_ids[src.group], src.capacity, 1, single=True)[0]
            destination = validate_ids(local_ids[dst.group], dst.capacity, 1, single=True)[0]
            # Keep raw ring ordering. For T<ring size the valid rows start at
            # zero; after wrap every row is valid. Absolute T restores position.
            size = min(snapshot.boundary, dst.rows) * dst.row_bytes
            plan.append(
                CopyRange(dst.name, dst.cache_index, source * src.block_stride, destination * dst.block_stride, size)
            )
    if not plan or any(r.nbytes <= 0 for r in plan):
        raise ValueError("State transfer must contain real data")
    return tuple(plan)


@dataclass
class SourceLease:
    """One immutable source snapshot; one consumer, one READ per TP rank.

    Acquiring twice for the same consumer/rank is idempotent for control RPC
    retries. D schedules one data operation per rank and must not retry it
    concurrently. Only the all-worker completion ACK can release acquired data.
    """

    snapshot: StateSnapshot
    expires_at: float
    resource_ids: tuple[tuple[int, ...], ...] = ()
    consumer: str | None = None
    acquired_ranks: set[int] = field(default_factory=set)
    finished: bool = False

    def acquire(self, transfer_id: str, consumer: str, rank: int, now: float) -> bool:
        if (
            self.finished
            or transfer_id != self.snapshot.transfer_id
            or not isinstance(consumer, str)
            or not consumer
            or type(rank) is not int
            or not 0 <= rank < self.snapshot.tp_size
        ):
            return False
        if not self.acquired_ranks and now >= self.expires_at:
            return False
        if self.consumer not in (None, consumer):
            return False
        self.consumer = consumer
        self.acquired_ranks.add(rank)
        return True

    def complete(self, transfer_id: str, consumer: str) -> bool:
        if transfer_id != self.snapshot.transfer_id or not consumer or self.consumer not in (None, consumer):
            return False
        self.finished = True
        return True

    def may_expire(self, now: float) -> bool:
        return not self.acquired_ranks and now >= self.expires_at
