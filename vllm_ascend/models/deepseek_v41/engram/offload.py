# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Global Engram rows shared by P/D workers through MemFabric entry_gather."""

import atexit
import ctypes
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from urllib.parse import urlparse

import torch
import torch.distributed as dist
from vllm.logger import logger

from vllm_ascend.ascend_config import EngramOffloadConfig
from vllm_ascend.ops.triton.engram_lookup import (
    _engram_offload_decode_kernel,
    _engram_offload_encode_kernel,
    mapped_engram_gather_dequant,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton, triton
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

_GIB = 1 << 30
_MAX_GATHER_ROWS = 65536
_SCALE_GROUP = 32
_ENCODE_BLOCK = 128


@dataclass(frozen=True)
class EngramOffloadTable:
    rows: int
    rows_per_owner: int
    offset: int

    def owner_range(self, owner: int) -> tuple[int, int]:
        start = min(owner * self.rows_per_owner, self.rows)
        return start, min(start + self.rows_per_owner, self.rows)


def make_offload_tables(rows: list[int], owners: int) -> tuple[list[EngramOffloadTable], int]:
    if owners <= 0 or not rows or any(count <= 0 for count in rows):
        raise ValueError("Engram offload requires positive table sizes and at least one owner")
    tables = []
    offset = 1  # Every owner reserves a zero row for invalid/padded IDs.
    for count in rows:
        per_owner = (count + owners - 1) // owners
        tables.append(EngramOffloadTable(count, per_owner, offset))
        offset += per_owner
    return tables, offset


class EngramOffloadPool:
    def __init__(
        self,
        config: EngramOffloadConfig,
        rank: int,
        rows: list[int],
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
        checkpoint_identity: str,
    ) -> None:
        # Import only in the NPU worker; native MemFabric owns process/device state.
        from memfabric_hybrid import offload

        self.config = config
        self.rank = rank
        self.device = device
        self.dtype = dtype
        self.dim = dim
        if not 0 <= rank < config.world_size or dim <= 0 or dim % _SCALE_GROUP:
            raise ValueError("Invalid Engram pool rank or group-32 embedding width")
        if dtype not in (torch.int8, torch.bfloat16, torch.float8_e4m3fn):
            raise ValueError(f"Unsupported Engram offload storage dtype: {dtype}")
        self.scale_dtype = torch.uint8 if dtype == torch.float8_e4m3fn else torch.float32
        self.weight_bytes = dim * dtype.itemsize
        scale_bytes = 0 if dtype == torch.bfloat16 else dim // _SCALE_GROUP * self.scale_dtype.itemsize
        self.row_bytes = self.weight_bytes + scale_bytes
        owners = config.owner_ranks
        self.tables, self.rows_per_slot = make_offload_tables(rows, len(owners))
        if self.row_bytes > 65536 or config.world_size * self.rows_per_slot > 1 << 32:
            raise ValueError("Engram pool exceeds entry_gather's row width or uint32 ID limit")
        self.owner_index = owners.index(rank) if rank in owners else None
        self.slot_bytes = ((self.rows_per_slot * self.row_bytes + _GIB - 1) // _GIB) * _GIB
        endpoint = urlparse(config.store_url)
        port = endpoint.port
        if port is None:
            raise ValueError("engram_offload.store_url must include a TCP port")
        self.store = dist.TCPStore(
            endpoint.hostname,
            port + 1,
            config.world_size,
            rank == 0,
            timeout=timedelta(seconds=config.timeout_seconds),
        )
        layout = json.dumps(
            (
                checkpoint_identity,
                rows,
                dim,
                str(dtype),
                config.owner_ranks,
                config.local_world_size,
            )
        )
        self.store.set(f"layout/{rank}", layout)
        for peer in range(config.world_size):
            if self.store.get(f"layout/{peer}").decode() != layout:
                raise ValueError(f"Engram offload layout/checkpoint differs between ranks {rank} and {peer}")
        native = offload.OffloadConfig()
        native.device_id = device.index
        native.world_size = config.world_size
        native.local_world_size = config.local_world_size
        native.rank_id = rank
        native.reserve_size = self.slot_bytes
        native.alloc_size = self.slot_bytes if self.owner_index is not None else 0
        native.store_url = config.store_url
        native.scene = offload.Scene.SHARED
        if get_ascend_device_type() in (AscendDeviceType.A2, AscendDeviceType.A5):
            native.executor = offload.Executor.AICPU
        self.pool = offload.Offload()
        self.storage = None
        self.buffers = []
        self.mapped_bases = []
        self.load_epochs = [0] * len(rows)
        self.lookup: Callable[[int, torch.Tensor, torch.Tensor, int, int, int, int], None] = self._lookup_native
        try:
            self._check(self.pool.initialize(native), "initialize")
            if self.owner_index is not None:
                address = self.pool.malloc(self.slot_bytes, 0)
                if not address:
                    raise RuntimeError(f"Engram rank {rank}: allocating {self.slot_bytes} bytes of DRAM failed")
                backing = (ctypes.c_uint8 * (self.rows_per_slot * self.row_bytes)).from_address(address)
                self.storage = torch.frombuffer(backing, dtype=torch.uint8).reshape(self.rows_per_slot, self.row_bytes)
                self.storage.zero_()
            self._check(self.pool.register_entry_table(self.row_bytes, self.rows_per_slot), "register_entry_table")
            self.owners = torch.tensor(owners, dtype=torch.int64, device=device)
            get_mapped_pool = (
                getattr(self.pool, "get_mapped_pool", None) if get_ascend_device_type() == AscendDeviceType.A3 else None
            )
            base, rank_stride = get_mapped_pool() if get_mapped_pool is not None else (0, 0)
            if base:
                self.mapped_bases = [
                    torch.tensor(
                        [base + owner * rank_stride + table.offset * self.row_bytes for owner in owners],
                        dtype=torch.int64,
                        device=device,
                    )
                    for table in self.tables
                ]
                self.lookup = self._lookup_mapped
            else:
                self.stream = torch.npu.Stream(device=device)
                for _ in rows:
                    ids = torch.empty(_MAX_GATHER_ROWS, dtype=torch.int64, device=device)
                    count = torch.zeros(1, dtype=torch.int32, device=device)
                    staging = torch.empty((_MAX_GATHER_ROWS, self.row_bytes), dtype=torch.uint8, device=device)
                    self._check(self.pool.register_memory(staging.data_ptr(), staging.numel()), "register_memory")
                    self.buffers.append((ids, count, staging))
        except BaseException:
            try:
                self.close()
            except BaseException:
                logger.exception("Failed to close MemFabric pool after initialization failure")
            raise
        atexit.register(self.close)
        logger.info(
            "Engram MemFabric pool: rank=%d/%d owner=%s local DRAM=%.2f GiB row_bytes=%d mapped=%s",
            rank,
            config.world_size,
            self.owner_index,
            native.alloc_size / _GIB,
            self.row_bytes,
            bool(self.mapped_bases),
        )

    def _check(self, result, operation: str) -> None:
        if result not in (None, 0):
            raise RuntimeError(f"Engram rank {self.rank}: MemFabric {operation} failed: {result}")

    def owner_range(self, slot: int) -> tuple[int, int]:
        return (0, 0) if self.owner_index is None else self.tables[slot].owner_range(self.owner_index)

    def weights(self, slot: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        start, end = self.owner_range(slot)
        table = self.tables[slot]
        raw = (
            torch.empty((0, self.row_bytes), dtype=torch.uint8, device="cpu")
            if self.storage is None
            else self.storage[table.offset : table.offset + end - start]
        )
        weight = raw[:, : self.weight_bytes].view(self.dtype)
        scale = raw[:, self.weight_bytes :].view(self.scale_dtype) if self.dtype != torch.bfloat16 else None
        return weight, scale

    def finish_load(self, slot: int, error: str | None) -> None:
        epoch = self.load_epochs[slot]
        self.load_epochs[slot] += 1
        self.store.set(f"loaded/{slot}/{epoch}/{self.rank}", error or "")
        failures = []
        for peer in range(self.config.world_size):
            failure = self.store.get(f"loaded/{slot}/{epoch}/{peer}").decode()
            if failure:
                failures.append(f"rank {peer}: {failure}")
        if failures:
            raise RuntimeError(f"Engram table {slot} failed to load: {'; '.join(failures)}")

    def _lookup_mapped(
        self,
        slot: int,
        indices: torch.Tensor,
        out: torch.Tensor,
        head_start: int,
        local_heads: int,
        vocab_start: int,
        vocab_end: int,
    ) -> None:
        mapped_engram_gather_dequant(
            self.mapped_bases[slot],
            indices,
            out,
            self.tables[slot].rows_per_owner,
            self.row_bytes,
            self.weight_bytes,
            head_start,
            local_heads,
            vocab_start,
            vocab_end,
            self.dtype,
        )

    def _lookup_native(
        self,
        slot: int,
        indices: torch.Tensor,
        out: torch.Tensor,
        head_start: int,
        local_heads: int,
        vocab_start: int,
        vocab_end: int,
    ) -> None:
        init_device_properties_triton()
        table = self.tables[slot]
        encoded, count, staging = self.buffers[slot]
        codes = staging[:, : self.weight_bytes].view(self.dtype)
        scales = staging[:, self.weight_bytes :].view(self.scale_dtype) if self.dtype != torch.bfloat16 else codes
        # Native RDMA binds its queue/notify state to one stream across eager and graphs.
        caller = torch.npu.current_stream(self.device)
        self.stream.wait_stream(caller)
        try:
            with torch.npu.stream(self.stream):
                for offset in range(0, indices.shape[0] * local_heads, _MAX_GATHER_ROWS):
                    size = min(_MAX_GATHER_ROWS, indices.shape[0] * local_heads - offset)
                    _engram_offload_encode_kernel[(triton.cdiv(size, _ENCODE_BLOCK),)](
                        indices,
                        self.owners,
                        encoded,
                        count,
                        size,
                        offset,
                        indices.stride(0),
                        vocab_start,
                        vocab_end,
                        ROWS_PER_OWNER=table.rows_per_owner,
                        ROWS_PER_SLOT=self.rows_per_slot,
                        TABLE_OFFSET=table.offset,
                        HEAD_START=head_start,
                        LOCAL_HEADS=local_heads,
                        BLOCK=_ENCODE_BLOCK,
                    )
                    # The registered range's remaining bytes bound the native descriptor capacity.
                    self._check(
                        self.pool.entry_gather(staging[-size:], encoded, count, self.device),
                        "entry_gather",
                    )
                    _engram_offload_decode_kernel[(size,)](
                        codes[-size:],
                        scales[-size:],
                        indices,
                        out,
                        offset,
                        indices.stride(0),
                        vocab_start,
                        vocab_end,
                        CODE_STRIDE=codes.stride(0),
                        SCALE_STRIDE=scales.stride(0),
                        WIDTH=self.dim,
                        GROUP=_SCALE_GROUP,
                        HEAD_START=head_start,
                        LOCAL_HEADS=local_heads,
                        PAD_HEADS=out.shape[1],
                        QUANTIZED=self.dtype != torch.bfloat16,
                        MXFP8=self.dtype == torch.float8_e4m3fn,
                    )
        finally:
            caller.wait_stream(self.stream)

    def close(self) -> None:
        if self.pool is None:
            return
        torch.npu.synchronize(self.device)
        self._check(self.pool.uninitialize(), "uninitialize")
        self.pool = None
        self.storage = None
        self.buffers.clear()
        self.mapped_bases.clear()
        self.store = None
        atexit.unregister(self.close)
