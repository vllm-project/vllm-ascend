# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 node-local Engram GVA pool and immutable global-row owner maps."""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import json
import socket
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from vllm_ascend import envs

_GIB = 1 << 30
_ENCODE_BLOCK_SIZE = 128
_STORE_PORT_BASE = 8500
_MANIFEST_SCHEMA = 3
_MXFP8_FORMAT = "F8_E4M3/F8_E8M0"


def use_engram_gva(eligible: bool, mode: str | None = None) -> bool:
    """Select the A5 GVA backend, with a host-UVA comparison override.

    ``auto`` preserves capability-based selection. ``uva`` is useful for
    measuring registered host memory with the same shared DP group. ``gva``
    fails closed when the runtime cannot support the GVA path.
    """
    selected = (mode if mode is not None else envs.VLLM_ASCEND_ENGRAM_GVA_MODE).strip().lower()
    if selected not in {"auto", "gva", "uva"}:
        raise ValueError(f"Engram GVA mode must be one of auto, gva, or uva; got {selected!r}")
    if selected == "gva" and not eligible:
        raise RuntimeError("GVA mode requested but A5 GVA prerequisites are not met")
    return eligible and selected != "uva"


@dataclass(frozen=True)
class EngramGvaTable:
    """One table's global-row owner intervals and slot-local block."""

    layer_hash_index: int
    global_rows: int
    owner_ranges: tuple[tuple[int, int], ...]
    block_offset: int

    @property
    def max_rows_per_owner(self) -> int:
        return max(end - start for start, end in self.owner_ranges)

    def owner_range(self, owner: int) -> tuple[int, int]:
        return self.owner_ranges[owner]

    def encode(self, global_row: int, vocab_start: int, vocab_end: int, rows_per_slot: int) -> int:
        """CPU reference for the device-side global-row to owner-row mapping."""
        if global_row < vocab_start or global_row >= vocab_end:
            return 0
        local_row = global_row - vocab_start
        for owner, (start, end) in enumerate(self.owner_ranges):
            if start <= local_row < end:
                return owner * rows_per_slot + self.block_offset + local_row - start
        return 0


def _balanced_owner_ranges(rows: int, world_size: int) -> tuple[tuple[int, int], ...]:
    base, remainder = divmod(rows, world_size)
    ranges = []
    start = 0
    for owner in range(world_size):
        end = start + base + (owner < remainder)
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def make_engram_gva_local_tables(
    table_ids: list[str], head_sizes: list[tuple[int, ...]], tp_rank: int, tp_size: int
) -> tuple[tuple[int, ...], tuple[str, ...], int]:
    """Return each TP rank's head-local row counts and stable table IDs.

    TP/head sharding and GVA row ownership are independent axes. Each TP
    group owns only its head columns; the local DP sharing group row-shards
    that TP-local table.
    """
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError("Engram GVA TP rank must be within a positive TP world size")
    if len(table_ids) != len(head_sizes) or not table_ids:
        raise ValueError("Engram GVA table IDs and head layouts must be non-empty and aligned")
    local_rows = []
    local_table_ids = []
    local_heads = 0
    for table_id, sizes in zip(table_ids, head_sizes):
        if not sizes or len(sizes) % tp_size:
            raise ValueError(f"Engram GVA table {table_id} heads cannot be divided evenly over TP={tp_size}")
        heads_per_rank = len(sizes) // tp_size
        head_start = tp_rank * heads_per_rank
        head_end = head_start + heads_per_rank
        rows = sum(sizes[head_start:head_end])
        if rows <= 0:
            raise ValueError(f"Engram GVA table {table_id} has no rows for TP rank {tp_rank}")
        local_rows.append(rows)
        local_table_ids.append(f"{table_id}/head-cols-{head_start}-{head_end}")
        local_heads = max(local_heads, heads_per_rank)
    return tuple(local_rows), tuple(local_table_ids), local_heads


def make_engram_gva_tables(
    global_rows: list[int],
    world_size: int,
) -> tuple[tuple[EngramGvaTable, ...], int]:
    """Build contiguous, balanced row shards with every local rank as an owner."""
    if world_size < 2:
        raise ValueError("A5 Engram GVA requires at least two local row owners")
    if not global_rows or any(rows <= 0 for rows in global_rows):
        raise ValueError("Engram GVA requires positive table rows")
    tables = []
    offset = 0
    for layer_hash_index, rows in enumerate(global_rows):
        if rows < world_size:
            raise ValueError(f"Engram GVA table {layer_hash_index} has {rows} rows for {world_size} local row owners")
        ranges = _balanced_owner_ranges(rows, world_size)
        table = EngramGvaTable(layer_hash_index, rows, ranges, offset)
        tables.append(table)
        offset += table.max_rows_per_owner
    return tuple(tables), offset


def make_engram_gva_manifest(
    tables: tuple[EngramGvaTable, ...],
    *,
    generation: str,
    checkpoint_identity: str,
    owner_ids: list[str],
    table_ids: list[str],
    row_bytes: int,
    weight_bytes: int,
) -> dict[str, Any]:
    """Describe a packed pool where every local rank owns a balanced row shard."""
    if not generation or not checkpoint_identity:
        raise ValueError("Engram GVA manifest needs a generation and checkpoint identity")
    if len(owner_ids) == 0 or len(set(owner_ids)) != len(owner_ids):
        raise ValueError("Engram GVA manifest owner IDs must be non-empty and unique")
    if len(table_ids) != len(tables) or len(set(table_ids)) != len(table_ids):
        raise ValueError("Engram GVA manifest table IDs must uniquely match the registered tables")
    if row_bytes <= weight_bytes or weight_bytes <= 0:
        raise ValueError("Engram GVA manifest has invalid packed row widths")
    world_size = len(tables[0].owner_ranges) if tables else 0
    if world_size < 2 or len(owner_ids) != world_size:
        raise ValueError("Engram GVA manifest requires one owner for every local rank")
    if any(len(table.owner_ranges) != world_size for table in tables):
        raise ValueError("Engram GVA manifest tables must use the same fixed local owner membership")

    manifest: dict[str, Any] = {
        "schema": _MANIFEST_SCHEMA,
        "generation": generation,
        "checkpoint_identity": checkpoint_identity,
        "members": [
            {
                "member_id": member_id,
                "rank": rank,
                "role": "owner",
            }
            for rank, member_id in enumerate(owner_ids)
        ],
        "owners": [
            {"owner_id": owner_id, "owner_rank": rank, "object_id": rank} for rank, owner_id in enumerate(owner_ids)
        ],
        "row_bytes": row_bytes,
        "weight_bytes": weight_bytes,
        "scale_bytes": row_bytes - weight_bytes,
        "tables": [],
    }
    for table, table_id in zip(tables, table_ids):
        shards = []
        for rank, (owner_id, (row_start, row_end)) in enumerate(zip(owner_ids, table.owner_ranges)):
            object_offset = table.block_offset * row_bytes
            shards.append(
                {
                    "row_start": row_start,
                    "row_end": row_end,
                    "owner_id": owner_id,
                    "owner_rank": rank,
                    "objects": {
                        "weight": {
                            "object_id": rank,
                            "byte_offset": object_offset,
                            "row_stride": row_bytes,
                            "intra_row_offset": 0,
                        },
                        "scale": {
                            "object_id": rank,
                            "byte_offset": object_offset,
                            "row_stride": row_bytes,
                            "intra_row_offset": weight_bytes,
                        },
                    },
                }
            )
        manifest["tables"].append(
            {
                "table_id": table_id,
                "global_rows": table.global_rows,
                "format": _MXFP8_FORMAT,
                "planes": ["weight", "scale"],
                "row_bytes": row_bytes,
                "shards": shards,
            }
        )
    validate_engram_gva_manifest(manifest)
    return manifest


def validate_engram_gva_manifest(manifest: dict, expected_generation: str | None = None) -> None:
    """Reject incompatible or incomplete owner layouts before pool mapping."""
    schema = manifest.get("schema")
    if schema != _MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported Engram GVA manifest schema {schema}; expected {_MANIFEST_SCHEMA}")
    generation = manifest.get("generation")
    if not generation or (expected_generation is not None and generation != expected_generation):
        raise ValueError("Engram GVA manifest generation is missing or does not match the expected generation")
    if not manifest.get("checkpoint_identity"):
        raise ValueError("Engram GVA manifest checkpoint identity is missing")
    owners = manifest.get("owners")
    if not isinstance(owners, list) or not owners:
        raise ValueError("Engram GVA manifest has no owners")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError("A5 Engram GVA manifest requires at least two local owners")
    if [member.get("rank") for member in members] != list(range(len(members))):
        raise ValueError("Engram GVA manifest member ranks must match the fixed membership order")
    member_ids = [member.get("member_id") for member in members]
    if any(not member_id for member_id in member_ids) or len(set(member_ids)) != len(member_ids):
        raise ValueError("Engram GVA manifest member IDs must be present and unique")
    if any(member.get("role") != "owner" for member in members):
        raise ValueError("A5 Engram GVA requires every local pool member to own rows")
    owner_ids = [owner.get("owner_id") for owner in owners]
    owner_ranks = [owner.get("owner_rank") for owner in owners]
    object_ids = [owner.get("object_id") for owner in owners]
    if any(not owner_id for owner_id in owner_ids) or len(set(owner_ids)) != len(owner_ids):
        raise ValueError("Engram GVA manifest owner IDs must be present and unique")
    if any(not isinstance(rank, int) or rank < 0 or rank >= len(members) for rank in owner_ranks):
        raise ValueError("Engram GVA manifest owner rank is outside the pool membership")
    if len(set(owner_ranks)) != len(owner_ranks):
        raise ValueError("Engram GVA manifest owner ranks must be unique")
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("Engram GVA manifest owner object IDs must be unique")
    if owner_ranks != list(range(len(members))) or owner_ids != member_ids:
        raise ValueError("A5 Engram GVA owner membership must match every local pool member")
    row_bytes = manifest.get("row_bytes")
    weight_bytes = manifest.get("weight_bytes")
    scale_bytes = manifest.get("scale_bytes")
    if (
        not isinstance(row_bytes, int)
        or row_bytes <= 0
        or not isinstance(weight_bytes, int)
        or weight_bytes <= 0
        or not isinstance(scale_bytes, int)
        or scale_bytes <= 0
    ):
        raise ValueError("Engram GVA manifest row and plane widths must be positive integers")
    if weight_bytes + scale_bytes != row_bytes:
        raise ValueError("Engram GVA manifest packed plane widths do not match row_bytes")

    tables = manifest.get("tables")
    if not isinstance(tables, list) or not tables:
        raise ValueError("Engram GVA manifest has no tables")
    table_ids = [table.get("table_id") for table in tables]
    if any(not table_id for table_id in table_ids) or len(set(table_ids)) != len(table_ids):
        raise ValueError("Engram GVA manifest table IDs must be present and unique")
    for table in tables:
        global_rows = table.get("global_rows")
        if not isinstance(global_rows, int) or global_rows <= 0:
            raise ValueError("Engram GVA manifest global_rows must be positive")
        if table.get("format") != _MXFP8_FORMAT or table.get("planes") != ["weight", "scale"]:
            raise ValueError("Engram GVA manifest format or data planes do not match native MXFP8")
        if table.get("row_bytes") != row_bytes:
            raise ValueError("Engram GVA manifest table row width differs from the pool row width")
        shards = table.get("shards")
        if not isinstance(shards, list) or len(shards) != len(owners):
            raise ValueError("Engram GVA manifest shard count does not match owner membership")
        cursor = 0
        shard_rows = []
        seen_objects = set()
        for owner_index, shard in enumerate(shards):
            row_start, row_end = shard.get("row_start"), shard.get("row_end")
            if (
                not isinstance(row_start, int)
                or not isinstance(row_end, int)
                or row_start != cursor
                or row_end <= row_start
                or row_end > global_rows
            ):
                raise ValueError("Engram GVA manifest row ranges must be ordered and completely covered")
            if shard.get("owner_id") != owner_ids[owner_index] or shard.get("owner_rank") != owner_ranks[owner_index]:
                raise ValueError("Engram GVA manifest shard owner is missing or out of order")
            objects = shard.get("objects")
            if not isinstance(objects, dict) or set(objects) != {"weight", "scale"}:
                raise ValueError("Engram GVA manifest shard must define weight and scale objects")
            plane_descriptors = [objects[name] for name in ("weight", "scale")]
            if any(
                descriptor.get("object_id") != object_ids[owner_index]
                or not isinstance(descriptor.get("byte_offset"), int)
                or descriptor["byte_offset"] < 0
                or descriptor.get("row_stride") != row_bytes
                for descriptor in plane_descriptors
            ):
                raise ValueError("Engram GVA manifest shard object descriptor is invalid")
            if plane_descriptors[0].get("intra_row_offset") != 0:
                raise ValueError("Engram GVA weight plane must start at byte zero in a packed row")
            if plane_descriptors[1].get("intra_row_offset") != weight_bytes:
                raise ValueError("Engram GVA scale plane offset does not match the packed weight width")
            if object_ids[owner_index] in seen_objects:
                raise ValueError("Engram GVA manifest reuses an object within one table")
            seen_objects.add(object_ids[owner_index])
            shard_rows.append(row_end - row_start)
            cursor = row_end
        if cursor != global_rows:
            raise ValueError("Engram GVA manifest row ranges do not cover global_rows")
        if max(shard_rows) - min(shard_rows) > 1:
            raise ValueError("A5 Engram GVA row shards must be balanced across all local owners")


def close_engram_gva_pools(model) -> None:
    """Synchronize and close GVA pools before a model runner releases its model."""
    if model is None:
        return
    modules = list(model.modules()) if callable(getattr(model, "modules", None)) else [model]
    pools = {}
    for module in modules:
        pool = getattr(module, "engram_gva_pool", None)
        if pool is not None:
            pools[id(pool)] = pool
            module.engram_gva_pool = None
        pool = getattr(module, "_gva_pool", None)
        if pool is None:
            continue
        pools[id(pool)] = pool
        module._gva_pool = None
        for name in ("weight", "weight_scale_inv"):
            param = getattr(module, name, None)
            if param is not None:
                setattr(
                    module,
                    name,
                    torch.nn.Parameter(torch.empty(0, dtype=param.dtype, device=param.device), requires_grad=False),
                )
    for pool in pools.values():
        pool.close()


class EngramGvaPool:
    """Owns the per-process A5 SHARED pool and fixed lookup workspaces."""

    def __init__(
        self,
        global_rows: list[int],
        row_bytes: int,
        weight_bytes: int,
        capacity: int,
        device: torch.device,
        group,
        *,
        checkpoint_identity: str = "runtime-probe",
        table_ids: list[str] | None = None,
    ) -> None:
        if row_bytes <= 0 or weight_bytes <= 0 or weight_bytes >= row_bytes or capacity <= 0:
            raise ValueError("Engram GVA row width and staging capacity must be positive")
        if group is None or group.world_size <= 0:
            raise ValueError("A node-local Engram GVA pool needs a non-empty sharing group")
        self.group = group
        self.world_size = group.world_size
        self.rank = group.rank_in_group
        if not 0 <= self.rank < self.world_size:
            raise ValueError("Engram GVA sharing rank is outside the group")
        self.row_bytes = row_bytes
        self.weight_bytes = weight_bytes
        self.scale_bytes = row_bytes - weight_bytes
        self.capacity = capacity
        self.device = torch.device(device)
        layout_error = None
        try:
            self.tables, self.rows_per_slot = make_engram_gva_tables(global_rows, self.world_size)
            self.used_bytes = self.rows_per_slot * row_bytes
            if row_bytes > 120 * 1024:
                raise ValueError(f"Engram GVA row is too wide for entry_gather: {row_bytes} bytes")
            if self.world_size * self.rows_per_slot > 1 << 32:
                raise ValueError("Engram GVA global row ids exceed entry_gather's uint32 range")
        except Exception as exc:  # noqa: BLE001 - reject an invalid layout on every pool member
            layout_error = f"{type(exc).__name__}: {exc}"
        failures = self._collective_failures("owner-map construction", layout_error)
        if failures:
            raise RuntimeError(f"Engram GVA {failures}")
        table_ids = table_ids or [f"engram-table-{index}" for index in range(len(global_rows))]
        manifest_wire = None
        manifest_error = None
        try:
            pool_members = [
                f"{socket.gethostname()}/global-rank-{dist.get_global_rank(self.group.cpu_group, rank)}"
                for rank in range(self.world_size)
            ]
            owner_ids = pool_members
            generation_content = {
                "checkpoint_identity": checkpoint_identity,
                "table_ids": table_ids,
                "global_rows": global_rows,
                "owner_ids": owner_ids,
                "pool_members": pool_members,
                "owner_ranges": [table.owner_ranges for table in self.tables],
                "row_bytes": row_bytes,
                "weight_bytes": weight_bytes,
            }
            self.generation = hashlib.sha256(
                json.dumps(generation_content, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            self.manifest = make_engram_gva_manifest(
                self.tables,
                generation=self.generation,
                checkpoint_identity=checkpoint_identity,
                owner_ids=owner_ids,
                table_ids=table_ids,
                row_bytes=row_bytes,
                weight_bytes=weight_bytes,
            )
            manifest_wire = json.dumps(self.manifest, sort_keys=True, separators=(",", ":"))
        except Exception as exc:  # noqa: BLE001 - fail together before mapping the pool
            manifest_error = f"{type(exc).__name__}: {exc}"
        failures = self._collective_failures("manifest construction", manifest_error)
        if failures:
            raise RuntimeError(f"Engram GVA {failures}")
        manifests: list[str | None] = [None] * self.world_size
        dist.all_gather_object(manifests, manifest_wire, group=self.group.cpu_group)
        if any(candidate != manifest_wire for candidate in manifests):
            raise RuntimeError("Engram GVA local pool members built different owner-map manifests")
        self.owner_map = self._run_collective_stage(
            "owner-map descriptor allocation",
            lambda: torch.tensor(
                [[(start, end, rank) for rank, (start, end) in enumerate(table.owner_ranges)] for table in self.tables],
                dtype=torch.int64,
                device=self.device,
            ),
        )
        self.slot_bytes = ((self.used_bytes + _GIB - 1) // _GIB) * _GIB
        self._offload = self._initialize_offload()
        self._pool_base = 0
        self.owner_weights: dict[int, torch.Tensor] = {}
        self._backings: list[Any] = []
        self.encoded_ids = None
        self.count = None
        self.staging = None
        try:
            self._allocate_pool()
            self._register_layout()
            self._create_owner_backings()
            self._create_lookup_buffers(capacity)
        except Exception:
            self._release_offload()
            raise
        atexit.register(self.close)

    def _initialize_offload(self) -> Any:
        offload: Any = None
        config: Any = None
        error = None
        try:
            offload = _load_memfabric_offload()
            config = offload.OffloadConfig()
            config.device_id = self.device.index
            config.reserve_size = self.slot_bytes
            config.alloc_size = self.slot_bytes
            config.world_size = self.world_size
            config.rank_id = self.rank
            config.local_world_size = self.world_size
            config.scene = offload.Scene.SHARED
            config.flags = 0
            # MemFabric's default single-node store URL is derived from the
            # device ID and local_world_size. Distinct TP-local DP groups can
            # be non-contiguous, so that default can assign colliding or
            # mismatched URLs. Give each local owner group a deterministic
            # store endpoint based on its first global rank.
            first_global_rank = min(dist.get_global_rank(self.group.cpu_group, rank) for rank in range(self.world_size))
            config.store_url = f"tcp://127.0.0.1:{_STORE_PORT_BASE + first_global_rank}"
        except Exception as exc:  # noqa: BLE001 - propagate API/configuration failures to all members
            error = f"{type(exc).__name__}: {exc}"
        failures = self._collective_failures("offload configuration", error)
        if failures:
            raise RuntimeError(f"Engram GVA {failures}")

        error = None
        try:
            result = offload.initialize(config)
            if result != 0:
                error = f"offload.initialize returned {result}"
        except Exception as exc:  # noqa: BLE001 - report initialization errors to every rank
            error = f"{type(exc).__name__}: {exc}"
        failures = self._collective_failures("offload initialization", error)
        if failures:
            try:
                offload.uninitialize()
            except Exception as exc:  # noqa: BLE001 - include cleanup errors in the original failure
                failures += f"; offload teardown failed: {type(exc).__name__}: {exc}"
            raise RuntimeError(f"Engram GVA {failures}")
        return offload

    def _collective_failures(self, stage: str, error: str | None) -> str:
        errors: list[str | None] = [None] * self.world_size
        dist.all_gather_object(errors, error, group=self.group.cpu_group)
        failures = "; ".join(f"rank {rank}: {failure}" for rank, failure in enumerate(errors) if failure is not None)
        return f"{stage} failed ({failures})" if failures else ""

    def _run_collective_stage(self, stage: str, operation) -> Any:
        result = None
        error = None
        try:
            result = operation()
        except Exception as exc:  # noqa: BLE001 - propagate setup failures to every rank
            error = f"{type(exc).__name__}: {exc}"
        failures = self._collective_failures(stage, error)
        if failures:
            raise RuntimeError(f"Engram GVA {failures}")
        return result

    def _allocate_pool(self) -> None:
        def allocate():
            address = self._offload.malloc(self.slot_bytes, 0)
            if not address:
                raise RuntimeError(f"pool owner allocation returned a null address for {self.slot_bytes} bytes")
            return address

        self._pool_base = self._run_collective_stage("pool allocation", allocate)

    def _register_layout(self) -> None:
        def register():
            result = self._offload.register_entry_table(self.row_bytes, self.rows_per_slot)
            if result != 0:
                required_bytes = self.rows_per_slot * self.row_bytes
                raise RuntimeError(
                    f"layout registration returned {result} for owner: row grid needs "
                    f"{required_bytes} bytes per member, requested local allocation is {self.slot_bytes} bytes"
                )

        self._run_collective_stage("layout registration", register)

    def _create_owner_backings(self) -> None:
        def create():
            backings = []
            owner_weights = {}
            for table in self.tables:
                num_bytes = table.max_rows_per_owner * self.row_bytes
                address = self._pool_base + table.block_offset * self.row_bytes
                backing = (ctypes.c_uint8 * num_bytes).from_address(address)
                tensor = torch.frombuffer(backing, dtype=torch.uint8).reshape(table.max_rows_per_owner, self.row_bytes)
                tensor.zero_()
                backings.append(backing)
                owner_start, owner_end = table.owner_range(self.rank)
                owner_weights[table.layer_hash_index] = tensor[: owner_end - owner_start]
            return backings, owner_weights

        self._backings, self.owner_weights = self._run_collective_stage("owner backing creation", create)

    def _create_lookup_buffers(self, capacity: int) -> None:
        def create():
            return (
                torch.empty((capacity,), dtype=torch.int64, device=self.device),
                torch.zeros((1,), dtype=torch.int32, device=self.device),
                torch.empty((capacity, self.row_bytes), dtype=torch.uint8, device=self.device),
            )

        self.encoded_ids, self.count, self.staging = self._run_collective_stage("lookup buffer allocation", create)

    def _release_offload(self) -> None:
        offload = getattr(self, "_offload", None)
        if offload is None:
            return
        offload.uninitialize()
        self._offload = None

    def table(self, layer_hash_index: int) -> EngramGvaTable:
        table = self.tables[layer_hash_index]
        if table.layer_hash_index != layer_hash_index:
            raise ValueError(f"Invalid Engram GVA table index: {layer_hash_index}")
        return table

    def raw_weight(self, layer_hash_index: int) -> torch.Tensor:
        self.table(layer_hash_index)
        return self.owner_weights[layer_hash_index]

    def weight(self, layer_hash_index: int, dtype: torch.dtype) -> torch.Tensor:
        return self.raw_weight(layer_hash_index)[:, : self.weight_bytes].view(dtype)

    def scale(self, layer_hash_index: int) -> torch.Tensor:
        return self.raw_weight(layer_hash_index)[:, self.weight_bytes :]

    def owner_range(self, layer_hash_index: int) -> tuple[int, int]:
        return self.table(layer_hash_index).owner_range(self.rank)

    def gather(
        self,
        layer_hash_index: int,
        ids: torch.Tensor,
        vocab_start: int,
        vocab_end: int,
        head_start: int,
        local_heads: int,
    ):
        """Gather valid table rows into fixed HBM staging and mask invalid IDs later."""
        table = self.table(layer_hash_index)
        if vocab_start < 0 or vocab_end - vocab_start != table.global_rows:
            raise ValueError("Engram GVA vocabulary range exceeds the registered row layout")
        count = ids.shape[0] * local_heads
        if count > self.capacity:
            raise ValueError(f"Engram GVA lookup needs {count} rows, capacity is {self.capacity}")
        assert self.staging is not None
        if count == 0:
            return self.staging[:0, :0], self.staging[:0, :0]
        assert self.owner_map is not None
        assert self.encoded_ids is not None
        assert self.count is not None
        assert self._offload is not None

        from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton, triton

        init_device_properties_triton()
        from vllm_ascend.ops.triton.engram_lookup import _engram_gva_encode_ids_kernel

        _engram_gva_encode_ids_kernel[(triton.cdiv(count, _ENCODE_BLOCK_SIZE),)](
            ids,
            self.owner_map[layer_hash_index],
            self.encoded_ids,
            self.count,
            count,
            vocab_start,
            vocab_end,
            ids.stride(0),
            ROWS_PER_SLOT=self.rows_per_slot,
            BLOCK_OFFSET=table.block_offset,
            HEAD_START=head_start,
            LOCAL_HEADS=local_heads,
            NUM_OWNERS=len(table.owner_ranges),
            BLOCK=_ENCODE_BLOCK_SIZE,
            num_warps=4,
        )
        result = self._offload.entry_gather(self.staging, self.encoded_ids[:count], self.count, self.device)
        if result not in (None, 0):
            raise RuntimeError(f"Engram GVA entry_gather failed: {result}")
        codes = self.staging[:count, : self.weight_bytes].view(torch.float8_e4m3fn)
        scales = self.staging[:count, self.weight_bytes :].view(torch.uint8)
        return codes, scales

    def close(self) -> None:
        offload = getattr(self, "_offload", None)
        if offload is None:
            return
        if self.device.type == "npu":
            torch.npu.synchronize()
        self._release_offload()
        self._pool_base = 0
        self.owner_weights.clear()
        self._backings.clear()
        self.owner_map = None
        self.encoded_ids = None
        self.count = None
        self.staging = None
        atexit.unregister(self.close)

    def uninitialize(self) -> None:
        """Compatibility alias for explicit offload teardown."""
        self.close()


def _load_memfabric_offload() -> Any:
    from memfabric_hybrid import offload  # type: ignore[import-untyped]

    return offload
