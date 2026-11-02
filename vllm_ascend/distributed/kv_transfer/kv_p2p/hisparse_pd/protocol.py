# SPDX-License-Identifier: Apache-2.0
"""Versioned, JSON-only HiSparse P/D descriptors and transfer leases.

These leases retain a finished producer request through the upstream connector
contract. They do not allocate KV blocks or implement a second cache manager.
"""

from __future__ import annotations

import bisect
import copy
import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

VERSION = 5
BF16_FORMAT = "bf16"
C8_FORMAT = "ascend_sfa_e4m3fn_nope512_rope64_scale4_tile128_v1"
INDEXER_FP8_FORMAT = "ascend_indexer_e4m3fn_key_fp32_scale_v1"
SOURCE_TRANSPORT = "producer_hbm_stage_v1"
STAGE_BYTES = 16 * 1024 * 1024


def validate_wire(message: dict) -> None:
    if not isinstance(message, dict):
        raise ValueError("HiSparse P/D control messages must be objects")
    if type(message.get("version")) is not int or message["version"] != VERSION:
        raise ValueError("HiSparse P/D wire version mismatch")
    if message.get("source_transport") != SOURCE_TRANSPORT:
        raise ValueError("HiSparse P/D requires explicit producer HBM staging")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class LayerLayout:
    name: str
    group_id: int
    role: str
    block_size: int
    row_bytes: int
    base: int
    block_stride: int
    num_blocks: int
    payload_format: str = BF16_FORMAT
    component: str = "kv"
    model_role: str = "target"

    def __post_init__(self):
        if self.role not in ("hisparse_source", "hisparse_indexer"):
            raise ValueError(f"Unsupported transfer role {self.role}")
        if self.model_role not in ("target", "draft"):
            raise ValueError("Unknown target/draft owner")
        if self.payload_format not in (BF16_FORMAT, C8_FORMAT, INDEXER_FP8_FORMAT):
            raise ValueError("Unknown HiSparse P/D payload format")
        if any(
            type(value) is not int
            for value in (self.block_size, self.row_bytes, self.base, self.block_stride, self.num_blocks)
        ):
            raise ValueError("KV geometry must use integer byte addresses")
        if self.payload_format == C8_FORMAT:
            if self.role != "hisparse_source" or self.row_bytes != 656 or self.component != "kv":
                raise ValueError("Ascend C8 SOURCE requires the declared 656-byte format")
        elif self.payload_format == INDEXER_FP8_FORMAT:
            if self.role != "hisparse_indexer" or self.component not in ("key", "scale"):
                raise ValueError("FP8 INDEXER requires explicit key/scale components")
            if self.component == "scale" and self.row_bytes % 4:
                raise ValueError("INDEXER scales must contain complete FP32 values")
        elif self.component != "kv" or self.row_bytes % 2:
            raise ValueError("BF16 KV rows must contain complete two-byte values")
        if type(self.group_id) is not int or self.group_id < 0:
            raise ValueError("Invalid manager group ID")
        if min(self.block_size, self.row_bytes, self.base, self.block_stride, self.num_blocks) <= 0:
            raise ValueError("Invalid or empty KV address geometry")
        if self.block_stride < self.block_size * self.row_bytes:
            raise ValueError("KV pages overlap")
        if self.base + (self.num_blocks - 1) * self.block_stride + self.block_size * self.row_bytes >= 2**64:
            raise ValueError("KV address exceeds uint64")

    def signature(self) -> tuple:
        # Capacities, addresses and HMA block/layer strides may differ on P/D.
        return (
            self.name,
            self.role,
            self.block_size,
            self.row_bytes,
            self.payload_format,
            self.component,
            self.model_role,
        )

    def key(self) -> tuple[str, str, str]:
        return self.name, self.component, self.model_role

    def address(self, block: int, offset: int = 0) -> int:
        if not 0 <= block < self.num_blocks or not 0 <= offset < self.block_size:
            raise ValueError(f"Out-of-range KV page for {self.name}")
        return self.base + block * self.block_stride + offset * self.row_bytes


def validate_layouts(layouts: list[LayerLayout]) -> None:
    names = [layer.key() for layer in layouts]
    if not names or len(names) != len(set(names)):
        raise ValueError("A transfer must contain each layer exactly once")
    roles = {layer.role for layer in layouts}
    if roles != {"hisparse_source", "hisparse_indexer"}:
        raise ValueError("Both SOURCE and INDEXER layers are required")
    for owner in {layer.model_role for layer in layouts}:
        if {layer.role for layer in layouts if layer.model_role == owner} != roles:
            raise ValueError("Each target/draft owner requires SOURCE and INDEXER")
    pairs = {}
    for layer in layouts:
        if layer.payload_format == INDEXER_FP8_FORMAT:
            pairs.setdefault((layer.name, layer.model_role), []).append(layer)
    for components in pairs.values():
        if {layer.component for layer in components} != {"key", "scale"}:
            raise ValueError("FP8 INDEXER publication must include both key and scale")
        if len({(layer.group_id, layer.block_size, layer.num_blocks) for layer in components}) != 1:
            raise ValueError("FP8 INDEXER key/scale manager geometry differs")


def plan_reads(
    source: list[LayerLayout],
    destination: list[LayerLayout],
    source_blocks: dict[int, list[int]],
    destination_blocks: dict[int, list[int]],
    start_token: int,
    end_token: int,
    *,
    include_host: bool = True,
    with_roles: bool = False,
) -> list[tuple[int, int, int]]:
    """Return (destination, source, bytes) without int32 address arithmetic.

    Copy only the requested token interval, including a partial tail. The
    scheduler's group IDs index manager blocks, never expanded indexer pages.
    """
    validate_layouts(source)
    validate_layouts(destination)
    if start_token < 0 or end_token <= start_token:
        raise ValueError("The imported token interval must be nonempty")
    remote = {layer.key(): layer for layer in source}
    if remote.keys() != {layer.key() for layer in destination}:
        raise ValueError("Producer and consumer layer sets differ")
    result = []
    for dst in destination:
        src = remote[dst.key()]
        if src.signature() != dst.signature():
            raise ValueError(f"Incompatible KV layer {dst.name}")
        source_ids = source_blocks[src.group_id]
        destination_ids = destination_blocks[dst.group_id]
        required = (end_token + dst.block_size - 1) // dst.block_size
        if min(len(source_ids), len(destination_ids)) < required:
            raise ValueError(f"Missing block IDs for {dst.name}")
        token = start_token
        while token < end_token:
            page, offset = divmod(token, dst.block_size)
            rows = min(dst.block_size - offset, end_token - token)
            src_ptr = src.address(source_ids[page], offset)
            dst_ptr = dst.address(destination_ids[page], offset)
            if include_host or dst.role != "hisparse_source":
                entry = (dst_ptr, src_ptr, rows * dst.row_bytes)
                result.append((*entry, dst.role, dst.key()) if with_roles else entry)
            token += rows
    if not result:
        raise ValueError("No KV payload was selected for transfer")
    return result


def coalesce_transfers(descriptors):
    """Merge only adjacent source/destination bytes of the same layer/role."""
    result = []
    for destination, source, length, role, layer in descriptors:
        if result:
            dst, src, count, previous_role, previous_layer = result[-1]
            if (
                role == previous_role
                and layer == previous_layer
                and dst + count == destination
                and src + count == source
            ):
                result[-1] = dst, src, count + length, role, layer
                continue
            # Two descending adjacent runs preserve the same byte mapping
            # when copied together from their lower physical addresses.
            # Restrict this reordering to SOURCE; INDEXER keeps its D2D path.
            if (
                role == previous_role == "hisparse_source"
                and layer == previous_layer
                and destination + length == dst
                and source + length == src
            ):
                result[-1] = destination, source, count + length, role, layer
                continue
        result.append((destination, source, length, role, layer))
    return result


def build_source_range_index(manifest: dict) -> dict:
    """Snapshot published SOURCE coverage once for this request lease.

    The immutable interval tuples describe only this publication. They retain
    no KV allocation and disappear with the existing request lease.
    """
    validate_wire(manifest)
    tokens = manifest["ticket"]["num_tokens"]
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("Invalid published source token interval")
    coverage = {}
    for item in manifest["layers"]:
        if item["role"] != "hisparse_source":
            continue
        layer = LayerLayout(**item)
        if layer.name in coverage:
            raise ValueError("Stage requires one declared SOURCE layer")
        blocks = manifest["blocks"].get(layer.group_id, manifest["blocks"].get(str(layer.group_id)))
        required = (tokens + layer.block_size - 1) // layer.block_size
        if not isinstance(blocks, list) or len(blocks) < required:
            raise ValueError("Missing published source blocks")
        intervals = []
        for page, block in enumerate(blocks[:required]):
            if type(block) is not int:
                raise ValueError("Invalid published source block")
            base = layer.address(block)
            extent = min(layer.block_size, tokens - page * layer.block_size) * layer.row_bytes
            intervals.append((base, base + extent))
        merged = []
        for base, end in sorted(intervals):
            if merged and base <= merged[-1][1]:
                merged[-1] = merged[-1][0], max(end, merged[-1][1])
            else:
                merged.append((base, end))
        coverage[layer.name] = tuple(x[0] for x in merged), tuple(x[1] for x in merged)
    return dict(identity=(manifest["identity"], manifest["rank"], manifest["world_size"]), coverage=coverage)


def validate_source_range(manifest: dict, message: dict, *, range_index=None) -> None:
    """Validate only published bytes, using a request-owned interval snapshot."""
    validate_wire(message)
    if any(type(message.get(key)) is not int for key in ("rank", "world_size", "source", "length")):
        raise ValueError("Stage geometry and rank fields must be integers")
    source, length = message["source"], message["length"]
    if source <= 0 or not 0 < length <= STAGE_BYTES or source + length >= 2**64:
        raise ValueError("Invalid bounded stage source range")
    index = build_source_range_index(manifest) if range_index is None else range_index
    if index["identity"] != (message["identity"], message["rank"], message["world_size"]):
        raise ValueError("Stage model, rank, or TP size mismatch")
    coverage = index["coverage"].get(message["layer"])
    if coverage is None:
        raise ValueError("Stage requires one declared SOURCE layer")
    starts, ends = coverage
    position = bisect.bisect_right(starts, source) - 1
    if position < 0 or source + length > ends[position]:
        raise ValueError("Stage range is outside the acquired lease SOURCE segments")


class LeaseRegistry:
    """Retain source ownership until an explicit, drained consumer release.

    A timeout is not proof that a DMA stopped. There is deliberately no TTL
    eviction of acquired leases. A dead peer requires engine/process teardown.
    """

    def __init__(self, unclaimed_timeout: float = 300, ack_retry_window: float = 600, max_tombstones: int = 4096):
        self._lock = threading.RLock()
        # StageSlot shares this lock: a source claim and slot reservation are
        # atomic with release. A failed or uncertain copy/read cannot clear it.
        self._stage_claim: tuple[str, str] | None = None
        self._leases: dict[str, dict] = {}
        self._released: dict[str, tuple[str, float]] = {}
        self._finished: set[str] = set()
        self.unclaimed_timeout = unclaimed_timeout
        self.ack_retry_window = ack_retry_window
        self.max_tombstones = max_tombstones
        self._closing = False

    def _prune_tombstones(self):
        now = time.monotonic()
        for lease_id, (_, expiry) in list(self._released.items()):
            if expiry <= now:
                del self._released[lease_id]
        while len(self._released) > self.max_tombstones:
            del self._released[next(iter(self._released))]

    def publish(self, lease_id: str, request_id: str, manifest: dict) -> None:
        with self._lock:
            if self._closing:
                raise RuntimeError("Producer transport is closing")
            self._prune_tombstones()
            if lease_id in self._leases or lease_id in self._released:
                raise ValueError("Duplicate export lease")
            # Freeze publication metadata before indexing its leased extents.
            # Later caller mutation cannot widen this lease's allowed bytes.
            manifest = copy.deepcopy(manifest)
            range_index = build_source_range_index(manifest)
            self._leases[lease_id] = {
                "request_id": request_id,
                "manifest": manifest,
                "source_range_index": range_index,
                "consumer": None,
                "expires": time.monotonic() + self.unclaimed_timeout,
            }

    def _expire_unclaimed(self):
        for lease_id, lease in list(self._leases.items()):
            if lease["consumer"] is None and time.monotonic() >= lease["expires"]:
                self._finished.add(lease["request_id"])
                del self._leases[lease_id]

    def acquire(self, lease_id: str, consumer: str, identity: str, rank: int, world_size: int) -> dict:
        with self._lock:
            if self._closing:
                raise ValueError("Producer transport is closing")
            self._expire_unclaimed()
            lease = self._leases.get(lease_id)
            if lease is None:
                return {"status": "pending"}
            manifest = lease["manifest"]
            if (manifest["identity"], manifest["rank"], manifest["world_size"]) != (identity, rank, world_size):
                raise ValueError("Model, rank, or TP size mismatch")
            if not consumer or lease["consumer"] not in (None, consumer):
                raise ValueError("Export lease is already claimed by a different consumer")
            lease["consumer"] = consumer
            return {"status": "ready", "manifest": manifest}

    def release(self, lease_id: str, consumer: str, *, drained: bool) -> None:
        with self._lock:
            self._prune_tombstones()
            if self._released.get(lease_id, (None, 0))[0] == consumer:
                return  # A lost reply can be retried without a duplicate completion.
            lease = self._leases.get(lease_id)
            if lease is None or lease["consumer"] != consumer or not drained:
                raise ValueError("Only the owning, drained reader may release a lease")
            if self._stage_claim == (lease_id, consumer):
                raise ValueError("Producer stage is outstanding; request lease remains retained")
            self._finished.add(lease["request_id"])
            self._released[lease_id] = consumer, time.monotonic() + self.ack_retry_window
            self._prune_tombstones()
            del self._leases[lease_id]

    def validate_source_claim(self, message: dict) -> None:
        with self._lock:
            lease = self._leases.get(message["lease_id"])
            if lease is None or not message["consumer"] or lease["consumer"] != message["consumer"]:
                raise ValueError("Stage requires the owning acquired lease")
            validate_source_range(lease["manifest"], message, range_index=lease["source_range_index"])

    def stage_position(self, lease_id, consumer):
        with self._lock:
            lease = self._leases[lease_id]
            if lease["consumer"] != consumer:
                raise ValueError("Stage reader does not own this lease")
            return lease.get("last_stage_chunk")

    def record_stage_release(self, lease_id, consumer, position):
        with self._lock:
            if self._stage_claim != (lease_id, consumer):
                raise ValueError("Only the owning stage may advance its reader")
            self._leases[lease_id]["last_stage_chunk"] = position

    def take_finished(self) -> set[str]:
        with self._lock:
            self._expire_unclaimed()
            result, self._finished = self._finished, set()
            return result

    def active(self) -> int:
        with self._lock:
            return sum(lease["consumer"] is not None for lease in self._leases.values())

    def begin_shutdown(self):
        """Atomically exclude new claims, or leave the ACK server operational."""
        with self._lock:
            if any(lease["consumer"] is not None for lease in self._leases.values()):
                raise RuntimeError("HiSparse P/D still has source leases; drain the peer before cache teardown")
            self._closing = True
            self._leases.clear()  # Unclaimed pointers were never exposed to a reader.


def encode_layouts(layouts: list[LayerLayout]) -> list[dict]:
    validate_layouts(layouts)
    return [asdict(layer) for layer in layouts]


def batch_source_transfers(descriptors):
    """Pack logical SOURCE fragments within one 16 MiB, 256-segment slot.

    Source/destination block allocation and logical ordering are unchanged.
    INDEXER descriptors keep their original D2D path and relative ordering.
    """
    parts, byte_count, owner = [], 0, None
    for destination, source, length, role, layer in descriptors:
        if role != "hisparse_source":
            if parts:
                yield "hisparse_source", owner, parts
                parts, byte_count, owner = [], 0, None
            yield role, layer, [(destination, source, length)]
            continue
        if parts and owner != layer:
            yield "hisparse_source", owner, parts
            parts, byte_count = [], 0
        owner = layer
        while length:
            count = min(length, STAGE_BYTES - byte_count)
            parts.append((destination, source, count))
            byte_count += count
            destination, source, length = destination + count, source + count, length - count
            if byte_count == STAGE_BYTES or len(parts) == 256:
                yield "hisparse_source", owner, parts
                parts, byte_count = [], 0
    if parts:
        yield "hisparse_source", owner, parts
