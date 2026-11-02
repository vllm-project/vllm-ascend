# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 node-local Engram GVA pool and immutable global-row owner maps."""

from __future__ import annotations

import atexit
import ctypes
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from vllm_ascend import envs

_GIB = 1 << 30
_ENCODE_BLOCK_SIZE = 128
_STORE_PORT_BASE = 8500


def use_engram_gva(
    eligible: bool,
    mode: str | None = None,
    *,
    sparse_kv_offload: bool = False,
) -> bool:
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
    use_gva = eligible and selected != "uva"
    if use_gva and sparse_kv_offload:
        raise RuntimeError(
            "A5 Engram GVA and sparse KV offload cannot share the MemFabric Offload runtime; "
            "disable sparse KV offload or set VLLM_ASCEND_ENGRAM_GVA_MODE=uva"
        )
    return use_gva


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
    """Return each TP rank's head-local row counts, IDs, and maximum heads.

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
        self._validate_shared_layout(checkpoint_identity, table_ids, global_rows)
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
        self._owner_storage = None
        self.encoded_ids = None
        self.count = None
        self.staging = None
        try:
            self._allocate_pool()
            self._register_layout()
            self._create_owner_storage()
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

    def _validate_shared_layout(self, checkpoint_identity: str, table_ids: list[str], global_rows: list[int]) -> None:
        layout = (checkpoint_identity, tuple(table_ids), tuple(global_rows), self.row_bytes, self.weight_bytes)
        layouts: list[Any] = [None] * self.world_size
        dist.all_gather_object(layouts, layout, group=self.group.cpu_group)
        if any(candidate != layout for candidate in layouts):
            raise RuntimeError("Engram GVA ranks must use the same checkpoint and table layout")

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

    def _create_owner_storage(self) -> None:
        def create():
            backing = (ctypes.c_uint8 * self.used_bytes).from_address(self._pool_base)
            storage = torch.frombuffer(backing, dtype=torch.uint8).reshape(self.rows_per_slot, self.row_bytes)
            storage.zero_()
            return storage

        self._owner_storage = self._run_collective_stage("owner backing creation", create)

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
        table = self.table(layer_hash_index)
        assert self._owner_storage is not None
        owner_start, owner_end = table.owner_range(self.rank)
        start = table.block_offset
        return self._owner_storage[start : start + owner_end - owner_start]

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
        self._owner_storage = None
        self.owner_map = None
        self.encoded_ids = None
        self.count = None
        self.staging = None
        atexit.unregister(self.close)


def _load_memfabric_offload() -> Any:
    from memfabric_hybrid import offload  # type: ignore[import-untyped]

    return offload
