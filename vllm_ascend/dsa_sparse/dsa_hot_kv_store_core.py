"""DSA worker-local DRAM hot store 的设备无关逻辑块管理核心。

``DSAHotKVStore`` 维护一套跨 layer/NOPE/ROPE 共用的物理 DRAM block id，
以及 request resident-row -> logical full-block -> physical DRAM block 的固定容量
块表。它负责 model-forward 级目的块预留、hash/refcount 复用、同 forward 重复
hash 隔离、请求批量释放和 block-table 版本管理；block 0 保留为空逻辑映射。

每层 NOPE/ROPE payload arena 由 Ascend 子类注入，实际 HBM -> DRAM 数据搬运由
独立满块复制算子完成。本模块不分配 Ascend swapped memory、不执行 payload
``copy_``，也不参与 scheduler 侧 HBM admission。初始化冻结后禁止运行期扩容，
以保证 LIDU/KSC 和图 replay 观察到的 arena 地址稳定。
"""

import collections
import enum
import math
from dataclasses import dataclass, field

import torch
from vllm.logger import init_logger

from vllm_ascend.dsa_sparse.dsa_types import (
    KV_CACHE_FULL_BLOCK_DUMP_NOOP_DST_BLOCK_ID,
)

logger = init_logger(__name__)

# DSA hot DRAM 当前由各 worker 本地维护：scheduler 只管理 HBM block 分配，
# worker 负责 logical full-block 到 DRAM pool row 的映射和复制计划。各 rank
# 依赖同一请求/hash 顺序得到一致结果，但 DRAM 物理块尚未提升为 scheduler 真源。


class BlockType(enum.Enum):
    NOPE_K = enum.auto()
    ROPE_K = enum.auto()


@dataclass
class _ArenaState:
    """One layer/plane's DRAM payload arena.

    DRAM block ids are shared by every layer and both cache planes because the
    device-side block table has no layer/plane dimension.  Keep only payload
    storage here; allocator metadata belongs to ``_SharedBlockPoolState``.
    """

    arena: torch.Tensor | None = None


@dataclass
class _SharedBlockPoolState:
    """Store-wide DRAM block-id allocator and ownership metadata."""

    hash_to_pool_idx: dict = field(default_factory=dict)
    pool_idx_to_hash: dict[int, object] = field(default_factory=dict)
    pool_ref_counts: dict[int, int] = field(default_factory=dict)
    free_block_ids: list[int] = field(default_factory=list)
    free_block_id_set: set[int] = field(default_factory=set)
    capacity: int = 0


@dataclass(frozen=True)
class _DumpRequestRow:
    request_id: object
    request_table_row_idx: int
    block_hashes: list | None
    logical_block_indices: list[int]


_DRAM_NULL_BLOCK_ID = 0


def _calculate_hot_num_blocks(indexer_num_blocks: int,
                              block_multiple: float) -> int:
    """Return usable DRAM blocks without truncating a fractional multiplier."""
    indexer_num_blocks = int(indexer_num_blocks)
    block_multiple = float(block_multiple)
    if indexer_num_blocks <= 0:
        raise ValueError(
            f"indexer_num_blocks must be positive, got {indexer_num_blocks}")
    if not math.isfinite(block_multiple) or block_multiple <= 0:
        raise ValueError(f"hot_cpu_block_multiple must be a positive finite number, got {block_multiple}")
    return int(math.ceil(indexer_num_blocks * block_multiple))


class DSAHotKVStore:
    def __init__(self):
        # Cache payloads mirror the shared DSA MLA physical split:
        # layer -> NOPE_K arena and layer -> ROPE_K arena. Request-local
        # logical block tables are stored separately as dense tensors.
        self.block_pools = {
            BlockType.NOPE_K: collections.defaultdict(_ArenaState),
            BlockType.ROPE_K: collections.defaultdict(_ArenaState),
        }
        # A DRAM block-table entry addresses the same pool index in every
        # layer's NOPE/ROPE arena.  Allocator/hash/refcount state must therefore
        # be unique at store scope rather than duplicated 2 * num_layers times.
        self._shared_block_pool = _SharedBlockPoolState()
        self._request_to_pool_idx: dict = {}
        self._pool_idx_to_request: dict[int, object] = {}
        self._dram_block_table: torch.Tensor | None = None
        self._dram_block_table_version = 0
        self._dram_block_table_device_cache: dict[tuple[str, str], tuple[int, torch.Tensor]] = {}
        # request_id -> unique DRAM block ids referenced by that request.
        self._request_owned_blocks = collections.defaultdict(set)
        # Ascend graph replay captures arena addresses. Initialization may
        # build the per-layer arenas incrementally, but runtime allocation must
        # never replace them after the worker has finished KV-cache setup.
        self._capacity_frozen = False

    @property
    def capacity_frozen(self) -> bool:
        return self._capacity_frozen

    def freeze_capacity(self) -> None:
        """Freeze the shared block-id range and every payload arena address.

        A frozen store may recycle existing block ids, but it must not grow the
        pool or replace a layer/plane arena. This is required by FULL graph
        capture and also turns DRAM exhaustion into an explicit admission error
        instead of a large hidden reallocation in the model-forward hot path.
        """
        pool_capacity = int(self._shared_block_pool.capacity)
        if pool_capacity <= _DRAM_NULL_BLOCK_ID + 1:
            raise RuntimeError(
                "Cannot freeze DSA hot DRAM cache before it is preallocated")
        for block_type, layer_arenas in self.block_pools.items():
            for layer_id, arena_state in layer_arenas.items():
                arena = arena_state.arena
                if arena is None or int(arena.shape[0]) < pool_capacity:
                    raise RuntimeError(
                        "Cannot freeze DSA hot DRAM cache with an undersized "
                        f"arena: layer={layer_id}, block_type={block_type.name}, "
                        "arena_capacity="
                        f"{0 if arena is None else int(arena.shape[0])}, "
                        f"pool_capacity={pool_capacity}"
                    )
        self._capacity_frozen = True

    @staticmethod
    def _maybe_pin_memory(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu" or tensor.is_pinned():
            return tensor
        try:
            return tensor.pin_memory()
        except RuntimeError:
            return tensor

    @classmethod
    def _allocate_host_arena(cls, block_shape: tuple[int, ...],
                             dtype: torch.dtype,
                             capacity: int) -> torch.Tensor:
        arena = torch.empty((capacity, *block_shape),
                            dtype=dtype,
                            device="cpu")
        return cls._maybe_pin_memory(arena.contiguous())

    def _ensure_arena_capacity(
        self, arena_state: _ArenaState, block_shape: tuple[int, ...], dtype: torch.dtype, min_capacity: int
    ) -> None:
        current_capacity = 0 if arena_state.arena is None else int(arena_state.arena.shape[0])
        if current_capacity >= min_capacity:
            return

        if self._capacity_frozen:
            raise RuntimeError(
                "DSA hot DRAM cache capacity is fixed after initialization: "
                f"arena_capacity={current_capacity}, required={min_capacity}. "
                "Increase hot_cpu_block_multiple instead of growing arenas "
                "during model execution."
            )

        new_capacity = max(min_capacity, current_capacity * 2, 1)
        new_arena = self._allocate_host_arena(
            block_shape=block_shape,
            dtype=dtype,
            capacity=new_capacity,
        )
        if arena_state.arena is not None and current_capacity > 0:
            new_arena[:current_capacity].copy_(arena_state.arena)
        arena_state.arena = new_arena

    def _ensure_shared_pool_capacity(self, pool_state: _SharedBlockPoolState,
                                     min_capacity: int) -> None:
        current_capacity = int(pool_state.capacity)
        if current_capacity >= min_capacity:
            return

        if self._capacity_frozen:
            raise RuntimeError(
                "DSA hot DRAM block pool capacity is fixed after "
                f"initialization: pool_capacity={current_capacity}, "
                f"required={min_capacity}. Increase "
                "hot_cpu_block_multiple before starting the engine."
            )

        new_capacity = max(int(min_capacity), current_capacity * 2, 1)
        first_new_block = max(current_capacity, _DRAM_NULL_BLOCK_ID + 1)
        if first_new_block < new_capacity:
            new_free_blocks = list(range(first_new_block, new_capacity))
            # free_block_ids is a LIFO stack. Store ranges in reverse order so
            # consecutive allocations return ascending block ids. The dump op
            # accepts arbitrary destinations, while locality still helps its
            # DRAM writes and keeps logical tables easier to inspect.
            pool_state.free_block_ids.extend(reversed(new_free_blocks))
            pool_state.free_block_id_set.update(new_free_blocks)
        pool_state.capacity = new_capacity

    def preallocate_layer_cache(
        self,
        layer_id: int,
        blk_type: BlockType,
        block_shape: tuple[int, ...],
        dtype: torch.dtype,
        num_blocks: int,
        *,
        max_request_rows: int | None = None,
        max_logical_blocks: int | None = None,
    ) -> None:
        """Preallocate a worker-local DRAM hot cache arena for one layer/type."""
        if num_blocks <= 0:
            return
        arena_state = self.block_pools[blk_type][layer_id]
        current_capacity = 0 if arena_state.arena is None else int(arena_state.arena.shape[0])
        # Capacity includes block 0, which is reserved as the IO null block.
        # Treat num_blocks as usable cache blocks to match HBM block-table
        # semantics and to keep block id 0 available for padding/invalid.
        required_capacity = int(num_blocks) + 1
        if current_capacity < required_capacity:
            self._ensure_arena_capacity(
                arena_state=arena_state,
                block_shape=tuple(block_shape),
                dtype=dtype,
                min_capacity=required_capacity,
            )
        self._ensure_shared_pool_capacity(self._shared_block_pool,
                                          required_capacity)
        if max_request_rows is not None and max_logical_blocks is not None:
            self._ensure_dram_block_table_capacity(
                min_rows=int(max_request_rows),
                min_logical_blocks=int(max_logical_blocks),
            )

    def _bump_dram_block_table_version(self) -> None:
        self._dram_block_table_version += 1
        self._dram_block_table_device_cache.clear()

    @property
    def dram_block_table_version(self) -> int:
        """Monotonic version used by graph replay table refresh caching."""
        return int(self._dram_block_table_version)

    def _ensure_dram_block_table_capacity(
        self, min_rows: int, min_logical_blocks: int, *, dtype: torch.dtype = torch.int32
    ) -> torch.Tensor:
        min_rows = max(0, int(min_rows))
        min_logical_blocks = max(0, int(min_logical_blocks))
        current = self._dram_block_table
        if (current is not None and current.dtype == dtype
                and int(current.shape[0]) >= min_rows
                and int(current.shape[1]) >= min_logical_blocks):
            return current

        if self._capacity_frozen:
            current_shape = (
                (0, 0) if current is None else
                (int(current.shape[0]), int(current.shape[1])))
            raise RuntimeError(
                "DSA hot DRAM logical block-table capacity is fixed after "
                "initialization: table_shape="
                f"{current_shape}, required=({min_rows}, "
                f"{min_logical_blocks}), dtype={dtype}. Increase "
                "max_active_reqs/max_model_len before starting the engine."
            )

        new_rows = max(min_rows,
                       0 if current is None else int(current.shape[0]) * 2,
                       1)
        new_width = max(
            min_logical_blocks,
            0 if current is None else int(current.shape[1]) * 2,
            1)
        new_table = torch.full((new_rows, new_width),
                               _DRAM_NULL_BLOCK_ID,
                               dtype=dtype,
                               device=torch.device("cpu"))
        if current is not None and int(current.numel()) > 0:
            rows = int(current.shape[0])
            cols = int(current.shape[1])
            new_table[:rows, :cols] = current.to(dtype=dtype)
        self._dram_block_table = new_table
        self._bump_dram_block_table_version()
        return new_table

    def _clear_pool_idx_tables(self, pool_idx: int) -> None:
        pool_idx = int(pool_idx)
        table = self._dram_block_table
        if table is not None and 0 <= pool_idx < int(table.shape[0]):
            table[pool_idx].fill_(_DRAM_NULL_BLOCK_ID)
            self._bump_dram_block_table_version()

    def bind_request_pool_index(self, request_id, pool_idx: int) -> None:
        pool_idx = int(pool_idx)
        old_pool_idx = self._request_to_pool_idx.get(request_id)
        if old_pool_idx == pool_idx:
            return

        old_request = self._pool_idx_to_request.get(pool_idx)
        if old_request is not None and old_request != request_id:
            self.release_request(old_request)

        if old_pool_idx is not None and int(old_pool_idx) != pool_idx:
            self._pool_idx_to_request.pop(int(old_pool_idx), None)
            self._clear_pool_idx_tables(int(old_pool_idx))

        self._request_to_pool_idx[request_id] = pool_idx
        self._pool_idx_to_request[pool_idx] = request_id
        self._clear_pool_idx_tables(pool_idx)

    def release_request(self, request_id) -> None:
        pool_idx = self._request_to_pool_idx.pop(request_id, None)
        if pool_idx is None:
            return
        pool_idx = int(pool_idx)
        self._pool_idx_to_request.pop(pool_idx, None)
        self._clear_pool_idx_tables(pool_idx)
        owned_blocks = self._request_owned_blocks.pop(request_id, set())
        self._release_pool_block_refs(owned_blocks)

    def _get_request_pool_idx(self, request_id) -> int | None:
        pool_idx = self._request_to_pool_idx.get(request_id)
        return None if pool_idx is None else int(pool_idx)

    def _bind_pool_hash(self, pool_idx: int, blk_hash) -> None:
        if blk_hash is None:
            return
        pool_state = self._shared_block_pool
        pool_idx = int(pool_idx)
        old_hash = pool_state.pool_idx_to_hash.get(pool_idx)
        if old_hash is not None and old_hash != blk_hash:
            if pool_state.hash_to_pool_idx.get(old_hash) == pool_idx:
                pool_state.hash_to_pool_idx.pop(old_hash, None)
        pool_state.hash_to_pool_idx[blk_hash] = pool_idx
        pool_state.pool_idx_to_hash[pool_idx] = blk_hash

    def _add_request_block_ref(self, request_id, pool_idx: int,
                               blk_hash) -> None:
        pool_idx = int(pool_idx)
        owned_blocks = self._request_owned_blocks[request_id]
        if pool_idx in owned_blocks:
            return

        pool_state = self._shared_block_pool
        owned_blocks.add(pool_idx)
        pool_state.pool_ref_counts[pool_idx] = int(pool_state.pool_ref_counts.get(pool_idx, 0)) + 1
        self._bind_pool_hash(pool_idx, blk_hash)

    @staticmethod
    def _mark_pool_blocks_free(pool_state: _SharedBlockPoolState,
                               pool_indices) -> None:
        if not pool_indices:
            return
        new_free_blocks = [
            int(pool_idx) for pool_idx in pool_indices
            if int(pool_idx) > _DRAM_NULL_BLOCK_ID
            and int(pool_idx) not in pool_state.free_block_id_set
        ]
        if not new_free_blocks:
            return
        pool_state.free_block_id_set.update(new_free_blocks)
        pool_state.free_block_ids.extend(sorted(new_free_blocks, reverse=True))

    def _release_pool_block_refs(self, pool_indices) -> None:
        pool_state = self._shared_block_pool
        blocks_to_free: list[int] = []
        for pool_idx in pool_indices:
            pool_idx = int(pool_idx)
            old_ref_count = int(pool_state.pool_ref_counts.pop(pool_idx, 0))
            if old_ref_count > 1:
                pool_state.pool_ref_counts[pool_idx] = old_ref_count - 1
                continue

            blk_hash = pool_state.pool_idx_to_hash.pop(pool_idx, None)
            if blk_hash is not None:
                current_pool_idx = pool_state.hash_to_pool_idx.get(blk_hash)
                if current_pool_idx == pool_idx:
                    pool_state.hash_to_pool_idx.pop(blk_hash, None)
            blocks_to_free.append(pool_idx)
        self._mark_pool_blocks_free(pool_state, blocks_to_free)

    @staticmethod
    def _pop_free_pool_block(pool_state: _SharedBlockPoolState) -> int | None:
        while pool_state.free_block_ids:
            pool_idx = int(pool_state.free_block_ids.pop())
            if pool_idx in pool_state.free_block_id_set:
                pool_state.free_block_id_set.remove(pool_idx)
                return pool_idx
        return None

    def _allocate_shared_pool_block(self) -> int:
        """Allocate one store-wide block id without touching layer payloads.

        Model-forward setup resolves logical DRAM ownership before any
        attention layer runs, so it cannot depend on a current layer's source
        tensor shape. Every Ascend arena is preallocated to the shared pool
        capacity before runtime capacity is frozen.
        """
        pool_state = self._shared_block_pool
        if not pool_state.free_block_id_set:
            self._ensure_shared_pool_capacity(
                pool_state,
                max(int(pool_state.capacity) + 1,
                    _DRAM_NULL_BLOCK_ID + 2),
            )
        pool_idx = self._pop_free_pool_block(pool_state)
        if pool_idx is None:
            raise RuntimeError("DSA hot DRAM cache has no free block")
        return int(pool_idx)

    def get_arena(self, layer_id: int, blk_type: BlockType) -> torch.Tensor:
        arena_state = self.block_pools[blk_type][layer_id]
        if arena_state.arena is None:
            raise ValueError("DRAM arena is not initialized.")
        return arena_state.arena

    def get_dram_block_table_tensor(
        self,
        num_logical_blocks: int | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        target_device = torch.device("cpu") if device is None else torch.device(device)
        table = self._dram_block_table
        if table is None:
            width = 0 if num_logical_blocks is None else max(
                0, int(num_logical_blocks))
            table = self._ensure_dram_block_table_capacity(
                min_rows=1,
                min_logical_blocks=width,
                dtype=dtype,
            )
        elif num_logical_blocks is not None and int(table.shape[1]) < int(num_logical_blocks):
            table = self._ensure_dram_block_table_capacity(
                min_rows=int(table.shape[0]),
                min_logical_blocks=int(num_logical_blocks),
                dtype=dtype,
            )
        if table.dtype != dtype:
            table = table.to(dtype=dtype)
        if target_device.type == "cpu":
            result = table
        else:
            cache_key = (str(target_device), str(dtype))
            cached = self._dram_block_table_device_cache.get(cache_key)
            if cached is not None and int(cached[0]) == self._dram_block_table_version:
                result = cached[1]
            else:
                result = table.to(device=target_device, non_blocking=True)
                self._dram_block_table_device_cache[cache_key] = (
                    self._dram_block_table_version, result)
        if num_logical_blocks is None:
            return result
        return result[:, :max(0, int(num_logical_blocks))]

    @staticmethod
    def _contiguous_bounds(indices: list[int]) -> tuple[int, int] | None:
        if not indices:
            return None
        start = int(indices[0])
        for offset, value in enumerate(indices[1:], start=1):
            if int(value) != start + offset:
                return None
        return start, start + len(indices)

    @classmethod
    def _read_table_row(cls, table: torch.Tensor, row_idx: int,
                        column_indices: list[int]) -> list[int]:
        bounds = cls._contiguous_bounds(column_indices)
        if bounds is not None:
            start, end = bounds
            return table[int(row_idx), start:end].tolist()
        return table[int(row_idx), column_indices].tolist()

    @classmethod
    def _write_table_row(cls, table: torch.Tensor, row_idx: int,
                         column_indices: list[int], values: list[int]) -> None:
        value_tensor = torch.as_tensor(values,
                                       dtype=table.dtype,
                                       device=table.device)
        bounds = cls._contiguous_bounds(column_indices)
        if bounds is not None:
            start, end = bounds
            table[int(row_idx), start:end].copy_(value_tensor)
            return
        table[int(row_idx), column_indices] = value_tensor

    def reserve_blocks_for_requests(
        self,
        *,
        request_ids: list,
        request_pool_indices: list[int],
        block_hash_rows: list[list] | list[None],
        logical_block_index_rows: list[list[int]],
    ) -> list[list[int]]:
        """Resolve DRAM block ids once at model-forward granularity.

        The returned rows contain *physical-copy destinations*. A -1 entry
        means that the request's logical table still references a valid DRAM
        block, but its payload was completed by an earlier forward. New
        same-forward duplicate hashes deliberately receive independent blocks
        and copies so every dump row has a self-contained producer/destination
        pair and the dump kernel needs no cross-row producer protocol.
        Reusing these rows for every layer keeps allocation, hash lookup,
        refcount updates and request ownership out of layer hooks.
        """
        row_count = len(request_ids)
        if not (
            row_count == len(request_pool_indices)
            == len(block_hash_rows)
            == len(logical_block_index_rows)
        ):
            raise ValueError(
                "DSA DRAM reservation rows must have matching request, pool, hash, and logical-index lengths"
            )
        if row_count == 0:
            return []
        if len({int(value) for value in request_pool_indices}) != row_count:
            raise RuntimeError(
                "DSA active requests cannot share a resident/request pool row")

        dump_rows: list[_DumpRequestRow] = []
        for (request_id, request_pool_idx, block_hashes,
             logical_block_indices) in zip(
                 request_ids,
                 request_pool_indices,
                 block_hash_rows,
                 logical_block_index_rows,
             ):
            if block_hashes is None:
                block_hashes = [None] * len(logical_block_indices)
            if len(block_hashes) != len(logical_block_indices):
                raise ValueError("DSA DRAM reservation hash and logical-index counts must match")
            if not logical_block_indices:
                dump_rows.append(
                    _DumpRequestRow(
                        request_id=request_id,
                        request_table_row_idx=int(request_pool_idx),
                        block_hashes=[],
                        logical_block_indices=[],
                    )
                )
                continue
            if min(int(value) for value in logical_block_indices) < 0:
                raise ValueError("DSA logical block indices must be non-negative")

            self.bind_request_pool_index(request_id, int(request_pool_idx))
            request_table_row_idx = self._get_request_pool_idx(request_id)
            if request_table_row_idx is None:
                raise RuntimeError(
                    f"DSA request {request_id!r} has no DRAM table row")
            dump_rows.append(
                _DumpRequestRow(
                    request_id=request_id,
                    request_table_row_idx=int(request_table_row_idx),
                    block_hashes=list(block_hashes),
                    logical_block_indices=[
                        int(value) for value in logical_block_indices
                    ],
                ))

        non_empty_rows = [
            row for row in dump_rows if row.logical_block_indices
        ]
        if not non_empty_rows:
            return [[] for _ in dump_rows]

        max_request_row = max(row.request_table_row_idx
                              for row in non_empty_rows) + 1
        max_logical_blocks = max(
            max(row.logical_block_indices) for row in non_empty_rows) + 1
        logical_table = self._ensure_dram_block_table_capacity(
            min_rows=max_request_row,
            min_logical_blocks=max_logical_blocks,
        )
        old_pool_id_rows = [
            self._read_table_row(
                logical_table,
                row.request_table_row_idx,
                row.logical_block_indices,
            )
            for row in dump_rows
        ]

        # Capacity errors must be reported before block-table/refcount writes.
        # bind_request_pool_index above may legitimately recycle an old row,
        # so preflight after binding sees the final free-list state.
        pool_state = self._shared_block_pool
        # Only hashes published by an earlier model forward are safe to reuse
        # without a copy. New same-forward duplicates deliberately keep
        # independent destinations: this preserves a one-row/one-copy contract
        # for the standalone dump kernel and avoids introducing a producer-row
        # dependency into its otherwise embarrassingly parallel metadata.
        required_new_blocks = 0
        for row, old_pool_ids in zip(dump_rows, old_pool_id_rows):
            for old_pool_id, blk_hash in zip(old_pool_ids, row.block_hashes):
                if int(old_pool_id) > _DRAM_NULL_BLOCK_ID:
                    continue
                if blk_hash is not None and blk_hash in pool_state.hash_to_pool_idx:
                    continue
                required_new_blocks += 1
        if self._capacity_frozen and required_new_blocks > len(self._shared_block_pool.free_block_id_set):
            raise RuntimeError(
                "DSA hot DRAM cache has insufficient fixed capacity for "
                f"this forward: required_new_blocks={required_new_blocks}, "
                "free_blocks="
                f"{len(self._shared_block_pool.free_block_id_set)}. Increase "
                "hot_cpu_block_multiple before starting the engine."
            )

        table_changed = False
        copy_destination_rows: list[list[int]] = []
        published_new_hashes: set = set()
        for row, old_pool_ids in zip(dump_rows, old_pool_id_rows):
            resolved_pool_ids: list[int] = []
            copy_destinations: list[int] = []
            for old_pool_id, blk_hash in zip(old_pool_ids, row.block_hashes):
                old_pool_id = int(old_pool_id)
                needs_physical_copy = False
                ref_hash = blk_hash
                if old_pool_id > _DRAM_NULL_BLOCK_ID:
                    pool_idx = old_pool_id
                    old_hash = pool_state.pool_idx_to_hash.get(pool_idx)
                    if blk_hash is not None and old_hash is not None and old_hash != blk_hash:
                        raise RuntimeError(
                            "DSA logical DRAM block hash changed in place: "
                            f"request={row.request_id!r}, pool_idx={pool_idx}"
                        )
                else:
                    # The global map contains reusable blocks published by a
                    # previous model forward. Hashes first published in this
                    # forward are tracked separately, so a later duplicate
                    # still receives its own producer/destination row without
                    # cloning the full, request-lifetime hash dictionary.
                    pool_idx = None
                    if blk_hash is not None and blk_hash not in published_new_hashes:
                        pool_idx = pool_state.hash_to_pool_idx.get(blk_hash)
                    if pool_idx is None:
                        pool_idx = self._allocate_shared_pool_block()
                        needs_physical_copy = True
                        if blk_hash is not None and blk_hash in published_new_hashes:
                            # Keep the first new block as the canonical future
                            # hash hit.  Later same-forward duplicates remain
                            # request-owned but deliberately unpublished.
                            ref_hash = None
                    pool_idx = int(pool_idx)
                    table_changed = True

                if pool_idx <= _DRAM_NULL_BLOCK_ID:
                    raise RuntimeError("DSA DRAM block 0 is reserved as null")
                pool_state.free_block_id_set.discard(pool_idx)
                hash_was_published = blk_hash is not None and blk_hash in pool_state.hash_to_pool_idx
                self._add_request_block_ref(
                    request_id=row.request_id,
                    pool_idx=pool_idx,
                    blk_hash=ref_hash,
                )
                if (
                    blk_hash is not None
                    and not hash_was_published
                    and pool_state.hash_to_pool_idx.get(blk_hash) == pool_idx
                ):
                    published_new_hashes.add(blk_hash)
                resolved_pool_ids.append(pool_idx)
                copy_destinations.append(
                    pool_idx if needs_physical_copy else KV_CACHE_FULL_BLOCK_DUMP_NOOP_DST_BLOCK_ID
                )

            if old_pool_ids != resolved_pool_ids:
                self._write_table_row(
                    logical_table,
                    row.request_table_row_idx,
                    row.logical_block_indices,
                    resolved_pool_ids,
                )
            copy_destination_rows.append(copy_destinations)

        if table_changed:
            self._bump_dram_block_table_version()
        return copy_destination_rows
