# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-Ascend metadata and byte-copy operations for HiSparse."""

from typing import Protocol

import torch
from vllm.triton_utils import tl, triton

_INVALIDATE_BLOCK_SIZE = 256
_INT32_MAX = 2**31 - 1
_MAX_INDEX_GROUPS = 65535


class HiSparseHostMapping(Protocol):
    """Backend-owned mapping, retained until all enqueued gathers complete.

    ``address_of`` must validate the entire tensor view against a live mapping
    and return its device-visible *view start* address. It must reject stale,
    unregistered or out-of-range views. Registration and capability discovery
    belong to the backend, not to these kernels.
    """

    device: torch.device

    def address_of(self, host_cache: torch.Tensor) -> int: ...


class HiSparseResolverWorkspace:
    """One index group's scratch, reused on one ordered stream.

    Capacities are host metadata. No allocation or device scalar read occurs
    during resolution. Concurrent streams must use different workspaces.
    """

    def __init__(self, device, max_rows, num_states, top_k, hot_size, max_union_rows):
        if min(max_rows, num_states, top_k, hot_size, max_union_rows) < 1:
            raise ValueError("workspace capacities must be positive")
        if top_k > hot_size or max_union_rows > hot_size or hot_size > 32768:
            raise ValueError("workspace requires top_k, union <= hot_size <= 32768")
        self.device = torch.device(device)
        self.max_rows = max_rows
        self.num_states = num_states
        self.top_k = top_k
        self.hot_size = hot_size
        self.max_union_rows = max_union_rows
        self.hash_size = triton.next_power_of_2(2 * max_union_rows)
        self.keys = torch.empty((num_states, self.hash_size), dtype=torch.int32, device=device)
        self.slots = torch.empty_like(self.keys)
        self.first = torch.empty_like(self.keys)
        self.loaded = torch.empty_like(self.keys)
        self.order = torch.empty((num_states, max_union_rows), dtype=torch.int32, device=device)
        self.protected = torch.empty((num_states, hot_size), dtype=torch.int32, device=device)
        self.evict = torch.empty_like(self.protected)
        self.hits = torch.empty_like(self.protected)
        self.refs = torch.empty((max_rows, top_k), dtype=torch.int32, device=device)
        self.counts = torch.empty((max_rows, 2), dtype=torch.uint64, device=device)


@triton.jit
def _hisparse_hash_find(keys, value, HASH_SIZE: tl.constexpr):
    # Integer hashing also preserves row IDs above the float32 exact range.
    h = ((value.to(tl.uint64) * 2654435761) & (HASH_SIZE - 1)).to(tl.int32)
    key = tl.load(keys + h)
    probes = 0
    while (key != -1) & (key != value) & (probes < HASH_SIZE):
        h = (h + 1) & (HASH_SIZE - 1)
        key = tl.load(keys + h)
        probes += 1
    return h, key


@triton.jit
def hisparse_resolve_residency_kernel(
    hot_table,
    global_indices,
    hot_indices,
    ownership,
    lru,
    states,
    miss_mask,
    attention,
    request_ids,
    source_table,
    resolved,
    valid_counts,
    swap_host,
    swap_hot,
    swap_counts,
    resident_table,
    num_valid_rows,
    keys,
    slots,
    first,
    loaded,
    order,
    protected,
    evict,
    hits,
    refs,
    counts,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    U: tl.constexpr,
    HASH: tl.constexpr,
    STATES: tl.constexpr,
    STATE_COUNT: tl.constexpr,
    HOST_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    HOT_STRIDE: tl.constexpr,
    INPUT_STRIDE: tl.constexpr,
    ATT_STRIDE: tl.constexpr,
    ATT_BLOCK: tl.constexpr,
    VALID_STRIDE: tl.constexpr,
    SOURCE_ROWS: tl.constexpr,
    SOURCE_COLS: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    SOURCE_BLOCK: tl.constexpr,
    RES_ROWS: tl.constexpr,
    RES_COLS: tl.constexpr,
    RES_STRIDE: tl.constexpr,
    RES_BLOCK: tl.constexpr,
    RES_NULL: tl.constexpr,
    WORK_K: tl.constexpr,
    WORK_H: tl.constexpr,
    WORK_U: tl.constexpr,
    TILE: tl.constexpr,
):
    """Five CUDA phases, with one writer for each active request's state.

    Serial union insertion makes first-reference ordering explicit. All large
    structures live in bounded global scratch; vector operations use fixed
    tiles. This correctness-first implementation makes no throughput promise.
    """
    row = tl.program_id(0).to(tl.int64)
    tl.store(counts + row * 2, tl.full((), 0, tl.uint64))
    tl.store(counts + row * 2 + 1, tl.full((), 0, tl.uint64))
    valid_rows = ROWS
    if num_valid_rows is not None:
        valid_rows = tl.minimum(tl.load(num_valid_rows), ROWS)
    request = row
    if request_ids is not None:
        request = tl.load(request_ids + row)
        request = tl.where(row < valid_rows, request, -1)
    state = request
    if states is not None:
        state = -1
        if (request >= 0) & (request < STATE_COUNT):
            state = tl.load(states + request)
    if state < 0:
        for base in range(tl.cdiv(K, TILE)):
            col = base * TILE + tl.arange(0, TILE)
            tl.store(hot_indices + row * K + col, -1, col < K)
            if attention is not None:
                tl.store(attention + row * ATT_STRIDE + col, -1, col < K)
            if resolved is not None:
                tl.store(resolved + row * K + col, -1, col < K)
        if valid_counts is not None:
            tl.store(valid_counts + row * VALID_STRIDE, 0)
        if swap_counts is not None:
            tl.store(swap_counts + row, 0)
        return
    if request_ids is not None:
        if row > 0:
            previous = tl.load(request_ids + row - 1)
            if previous == request:
                return
    tl.device_assert(state < STATES, "HiSparse state exceeds ownership rows")
    state = state.to(tl.int64)
    end = row + 1
    if request_ids is not None:
        same_request = True
        while (end < tl.minimum(ROWS, valid_rows)) & same_request:
            following = tl.load(request_ids + end)
            same_request = following == request
            end += same_request.to(tl.int64)

    key_row = keys + state * HASH
    slot_row = slots + state * HASH
    first_row = first + state * HASH
    loaded_row = loaded + state * HASH
    order_row = order + state * WORK_U
    protect_row = protected + state * WORK_H
    evict_row = evict + state * WORK_H
    hit_row = hits + state * WORK_H
    for base in range(tl.cdiv(HASH, TILE)):
        i = base * TILE + tl.arange(0, TILE)
        tl.store(key_row + i, -1, i < HASH)
        tl.store(slot_row + i, -1, i < HASH)
        tl.store(loaded_row + i, 0, i < HASH)
    union = 0
    # Phases 1/1b: translation, resident bypass, stable host union.
    for query in range(row, end):
        valid = 0
        for col in range(K):
            token = tl.load(global_indices + query * INPUT_STRIDE + col)
            g = token
            resident = -1
            if source_table is not None:
                g = -1
                if (token >= 0) & (request >= 0) & (request < SOURCE_ROWS):
                    block = token // SOURCE_BLOCK
                    if block < SOURCE_COLS:
                        physical = tl.load(source_table + request * SOURCE_STRIDE + block)
                        if physical > 0:
                            g = physical * SOURCE_BLOCK + token % SOURCE_BLOCK
                if resident_table is not None:
                    if (token >= 0) & (state < RES_ROWS):
                        block = token // RES_BLOCK
                        if block < RES_COLS:
                            physical = tl.load(resident_table + state * RES_STRIDE + block)
                            if (physical >= 0) & (physical != RES_NULL):
                                resident = physical * RES_BLOCK + token % RES_BLOCK
            g = tl.where(g >= HOST_ROWS, -1, g)
            if resolved is not None:
                tl.store(resolved + query * K + col, g)
            if miss_mask is not None:
                tl.store(miss_mask + query * K + col, 0)
            valid += ((resident >= 0) | (g >= 0)).to(tl.int32)
            out = resident
            host = (resident < 0) & (g >= 0)
            tl.store(refs + query * WORK_K + col, tl.where(host, g, -1))
            tl.store(hot_indices + query * K + col, out)
            if attention is not None:
                encoded = (out // BLOCK) * ATT_BLOCK + out % BLOCK
                tl.store(attention + query * ATT_STRIDE + col, tl.where(out >= 0, encoded, -1))
            if host:
                h, old = _hisparse_hash_find(key_row, g, HASH)
                if old == -1:
                    # Assert BEFORE either bounded order insertion or hash write.
                    tl.device_assert(union < U, "HiSparse host union exceeds max_union_rows")
                    if union < U:
                        tl.store(key_row + h, g)
                        tl.store(first_row + h, query * K + col)
                        tl.store(order_row + union, h)
                        union += 1
        if valid_counts is not None:
            tl.store(valid_counts + query * VALID_STRIDE, valid)
        if swap_counts is not None:
            tl.store(swap_counts + query, 0)
    if union == 0:
        return

    # Phase 2: protect ALL existing hits before selecting any eviction slot.
    for slot in range(H):
        g = tl.load(ownership + state * H + slot)
        is_hit = False
        if g >= 0:
            h, key = _hisparse_hash_find(key_row, g, HASH)
            is_hit = key == g
            if is_hit:
                tl.store(slot_row + h, slot)
        tl.store(protect_row + slot, is_hit.to(tl.int32))
    nhit = 0
    nevict = 0
    for pos in range(H):
        slot = tl.load(lru + state * H + pos).to(tl.int32)
        is_hit = False
        if (slot >= 0) & (slot < H):
            is_hit = tl.load(protect_row + slot) != 0
        if is_hit:
            tl.store(hit_row + nhit, slot)
            nhit += 1
        else:
            tl.store(evict_row + nevict, slot)
            nevict += 1

    # Phase 3: ordered misses, compact records and ownership publication.
    nmiss = 0
    for entry in range(union):
        h = tl.load(order_row + entry)
        slot = tl.load(slot_row + h)
        if slot == -1:
            slot = tl.load(evict_row + nmiss)
            g = tl.load(key_row + h)
            physical = -1
            if (slot >= 0) & (slot < H):
                block = tl.load(hot_table + request * HOT_STRIDE + slot // BLOCK)
                physical = block * BLOCK + slot % BLOCK
                tl.store(ownership + state * H + slot, g)
                tl.store(slot_row + h, slot)
            else:
                tl.store(slot_row + h, -2)
            tl.store(loaded_row + h, 1)
            if swap_host is not None:
                tl.store(swap_host + row * K + nmiss, g)
                tl.store(swap_hot + row * K + nmiss, physical)
            nmiss += 1
    # Phase 4: every reference; shared loads count once, resident counts never.
    hit_count = tl.full((), 0, tl.uint64)
    miss_count = tl.full((), 0, tl.uint64)
    for query in range(row, end):
        if swap_counts is not None:
            tl.store(swap_counts + query, tl.minimum(K, tl.maximum(0, nmiss - (query - row) * K)))
        for col in range(K):
            g = tl.load(refs + query * WORK_K + col)
            if g >= 0:
                h, key = _hisparse_hash_find(key_row, g, HASH)
                slot = tl.load(slot_row + h)
                physical = -1
                if slot >= 0:
                    block = tl.load(hot_table + request * HOT_STRIDE + slot // BLOCK)
                    physical = block * BLOCK + slot % BLOCK
                    is_miss = (tl.load(loaded_row + h) != 0) & (tl.load(first_row + h) == query * K + col)
                    hit_count += (~is_miss).to(tl.uint64)
                    miss_count += is_miss.to(tl.uint64)
                    if miss_mask is not None:
                        tl.store(miss_mask + query * K + col, is_miss.to(tl.int32))
                tl.store(hot_indices + query * K + col, physical)
                if attention is not None:
                    encoded = (physical // BLOCK) * ATT_BLOCK + physical % BLOCK
                    tl.store(attention + query * ATT_STRIDE + col, tl.where(physical >= 0, encoded, -1))
    tl.store(counts + row * 2, hit_count)
    tl.store(counts + row * 2 + 1, miss_count)
    # Phase 5: stale evictables, new loads, then old hits (all stable).
    for pos in range(H):
        slot = 0
        remaining = nevict - nmiss
        if pos < remaining:
            slot = tl.load(evict_row + pos + nmiss)
        elif pos < nevict:
            slot = tl.load(evict_row + pos - remaining)
        else:
            slot = tl.load(hit_row + pos - nevict)
        tl.store(lru + state * H + pos, slot)


@triton.jit
def _hisparse_accumulate_stats(counts, stats, ROWS: tl.constexpr):
    # Separate ordered launch avoids requiring a device uint64 atomic add.
    hits = tl.load(stats)
    misses = tl.load(stats + 1)
    for row in range(ROWS):
        hits = tl.add(hits, tl.load(counts + row * 2), sanitize_overflow=False)
        misses = tl.add(misses, tl.load(counts + row * 2 + 1), sanitize_overflow=False)
    tl.store(stats, hits)
    tl.store(stats + 1, misses)


@triton.jit
def hisparse_gather_bytes_kernel(
    host_address,
    hot_cache,
    source,
    target,
    mask,
    counts,
    states,
    attention,
    K: tl.constexpr,
    HOST_ROWS: tl.constexpr,
    HOT_ROWS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    ATT_BLOCK: tl.constexpr,
    COMPACT: tl.constexpr,
    TILE: tl.constexpr,
):
    """Opaque bytes: page gaps and unaligned view starts remain untouched."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1)
    active = True
    if states is not None:
        active = tl.load(states + row) >= 0
    dst = tl.load(target + row * K + col)
    if attention is not None:
        encoded = (dst // BLOCK) * ATT_BLOCK + dst % BLOCK
        tl.store(attention + row * K + col, tl.where(active & (dst >= 0), encoded, -1))
    if COMPACT:
        count = tl.minimum(K, tl.maximum(0, tl.load(counts + row)))
        active = active & (col < count)
    else:
        active = active & (tl.load(mask + row * K + col) != 0)
    if active:
        src = tl.load(source + row * K + col)
        if (src >= 0) & (dst >= 0) & (dst < HOT_ROWS):
            host = host_address.to(tl.pointer_type(tl.uint8))
            hot = hot_cache.to(tl.pointer_type(tl.uint8))
            out = hot + (dst // BLOCK).to(tl.int64) * PAGE_BYTES + (dst % BLOCK) * ROW_BYTES
            for base in range(tl.cdiv(ROW_BYTES, TILE)):
                byte = base * TILE + tl.arange(0, TILE)
                value = tl.load(
                    host + src.to(tl.int64) * ROW_BYTES + byte, (byte < ROW_BYTES) & (src < HOST_ROWS), other=0
                )
                tl.store(out + byte, value, byte < ROW_BYTES)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _indices(name, tensor, device, shape=None, ndim=None, contiguous=True, dtype=torch.int32):
    _require(isinstance(tensor, torch.Tensor), f"{name} must be a tensor")
    _require(tensor.device == device and tensor.dtype == dtype, f"{name}: wrong device or dtype")
    if shape is not None:
        _require(tuple(tensor.shape) == tuple(shape), f"{name}: wrong shape")
    if ndim is not None:
        _require(tensor.ndim == ndim, f"{name}: wrong rank")
    if contiguous:
        _require(tensor.is_contiguous(), f"{name} must be contiguous")
    elif tensor.ndim == 2:
        _require(tensor.stride(1) == 1, f"{name} must have dense rows")


def _cache_layout(host, hot, allow_staging=False):
    _require(
        host.device.type == "cpu" and host.ndim == 2 and host.is_contiguous(),
        "host_cache must be a contiguous CPU matrix",
    )
    _require(hot.ndim == 3 or (allow_staging and hot.ndim == 2), "unsupported hot cache rank")
    width = host.shape[1] * host.element_size()
    _require(
        width == hot.shape[-1] * hot.element_size() and width % 16 == 0,
        "cache row byte widths must match and be divisible by 16",
    )
    _require(hot.stride(-1) == 1, "hot cache columns must be dense")
    block = hot.shape[1] if hot.ndim == 3 else 1
    if hot.ndim == 3:
        _require(hot.stride(1) == hot.shape[2], "hot cache rows must be dense within pages")
    page_bytes = hot.stride(0) * hot.element_size()
    _require(page_bytes >= block * width, "hot cache pages must not overlap")
    return width, block, page_bytes, hot.shape[0] * block


class HiSparseTritonCacheOps:
    """CUDA-compatible four-method adapter; backend owns mapping and scratch.

    This object is intentionally not registered as an Ascend backend yet.
    Public methods require NPU tensors. The private launch paths are also used
    by CPU interpreter tests, which explicitly supply a CPU-only test mapping.
    """

    def __init__(self, workspace: HiSparseResolverWorkspace, host_mapping: HiSparseHostMapping | None = None):
        self.workspace = workspace
        self.host_mapping = host_mapping

    @staticmethod
    def _npu(tensor):
        if tensor.device.type != "npu":
            raise RuntimeError("HiSparse public cache ops require an NPU device")

    def hisparse_invalidate_written_slots(
        self,
        device_global_indices_ptrs,
        request_state_indices,
        req_id_per_token,
        written_slots,
        num_state_rows,
        region_stride,
    ):
        return hisparse_invalidate_written_slots(
            device_global_indices_ptrs,
            request_state_indices,
            req_id_per_token,
            written_slots,
            num_state_rows,
            region_stride,
        )

    def hisparse_resolve_residency(
        self,
        host_cache,
        hot_cache,
        hot_block_table,
        global_indices,
        hot_indices,
        device_global_indices,
        lru_slots,
        request_state_indices,
        region_stride,
        max_union_rows,
        miss_mask=None,
        stats=None,
        attention_indices=None,
        attention_block_stride=0,
        request_ids=None,
        source_block_table=None,
        source_block_size=0,
        resolved_global_indices=None,
        valid_counts=None,
        swap_host_physical_rows=None,
        swap_device_physical_rows=None,
        swap_counts=None,
        resident_block_table=None,
        resident_block_size=0,
        resident_null_block=0,
        num_valid_rows=None,
    ):
        self._npu(hot_cache)
        with torch.npu.device(hot_cache.device):
            self._resolve(
                host_cache,
                hot_cache,
                hot_block_table,
                global_indices,
                hot_indices,
                device_global_indices,
                lru_slots,
                request_state_indices,
                region_stride,
                max_union_rows,
                miss_mask,
                stats,
                attention_indices,
                attention_block_stride,
                request_ids,
                source_block_table,
                source_block_size,
                resolved_global_indices,
                valid_counts,
                swap_host_physical_rows,
                swap_device_physical_rows,
                swap_counts,
                resident_block_table,
                resident_block_size,
                resident_null_block,
                num_valid_rows,
            )

    def _resolve(
        self,
        host_cache,
        hot_cache,
        hot_block_table,
        global_indices,
        hot_indices,
        device_global_indices,
        lru_slots,
        request_state_indices,
        region_stride,
        max_union_rows,
        miss_mask=None,
        stats=None,
        attention_indices=None,
        attention_block_stride=0,
        request_ids=None,
        source_block_table=None,
        source_block_size=0,
        resolved_global_indices=None,
        valid_counts=None,
        swap_host_physical_rows=None,
        swap_device_physical_rows=None,
        swap_counts=None,
        resident_block_table=None,
        resident_block_size=0,
        resident_null_block=0,
        num_valid_rows=None,
    ):
        """Validate CUDA-compatible metadata and launch one resolver per request.

        Optional outputs retain independent strides; preallocated workspace and
        request leaders keep state ownership ordered on the caller's stream.
        """
        _, block, _, hot_rows = _cache_layout(host_cache, hot_cache)
        device = hot_cache.device
        _indices("global_indices", global_indices, device, ndim=2, contiguous=False)
        rows, k = global_indices.shape
        _indices("hot_indices", hot_indices, device, (rows, k))
        _indices("ownership", device_global_indices, device, ndim=2)
        _indices("lru_slots", lru_slots, device, ndim=2, dtype=torch.int16)
        nstates, h = lru_slots.shape
        _require(
            device_global_indices.shape == (nstates, h) and region_stride == h,
            "ownership, LRU and region_stride must agree",
        )
        _require(k <= h <= 32768 and hot_rows < _INT32_MAX, "invalid hot region capacity")
        _indices("hot_block_table", hot_block_table, device, ndim=2, contiguous=False)
        _require(hot_block_table.shape[1] * block >= h, "hot block table is too short")
        _require((request_ids is None) == (source_block_table is None), "request IDs and source table are paired")
        source_rows = source_cols = source_stride = 0
        if request_ids is not None:
            _indices("request_ids", request_ids, device, ndim=1)
            _require(request_ids.numel() >= rows, "request_ids is too short")
            _indices("source_block_table", source_block_table, device, ndim=2, contiguous=False)
            source_rows, source_cols = source_block_table.shape
            source_stride = source_block_table.stride(0)
            _require(source_block_size > 0, "source_block_size must be positive")
        _require(
            hot_block_table.shape[0] >= (source_rows if request_ids is not None else rows),
            "hot block table has too few request rows",
        )
        if request_state_indices is not None:
            _indices("request_state_indices", request_state_indices, device, ndim=1)
        else:
            _require(nstates >= rows, "ownership has too few rows without state mapping")
        if num_valid_rows is not None:
            _indices("num_valid_rows", num_valid_rows, device, (1,))
            _require(request_ids is not None, "num_valid_rows requires request_ids")
        res_rows = res_cols = res_stride = 0
        if resident_block_table is not None:
            _require(source_block_table is not None, "resident table requires request-relative inputs")
            _indices("resident_block_table", resident_block_table, device, ndim=2, contiguous=False)
            res_rows, res_cols = resident_block_table.shape
            res_stride = resident_block_table.stride(0)
            _require(resident_block_size == block and resident_null_block >= 0, "invalid resident layout")
        for name, tensor in (("miss_mask", miss_mask), ("resolved_global_indices", resolved_global_indices)):
            if tensor is not None:
                _indices(name, tensor, device, (rows, k))
        if attention_indices is not None:
            _indices("attention_indices", attention_indices, device, (rows, k), contiguous=False)
            _require(attention_block_stride >= block, "attention_block_stride is shorter than a block")
        if valid_counts is not None:
            _indices("valid_counts", valid_counts, device, (rows,), contiguous=False)
        compact = (swap_host_physical_rows, swap_device_physical_rows, swap_counts)
        _require(
            all(t is None for t in compact) or all(t is not None for t in compact),
            "all compact plan outputs must be provided together",
        )
        if swap_counts is not None:
            _indices("swap_host_physical_rows", swap_host_physical_rows, device, (rows, k))
            _indices("swap_device_physical_rows", swap_device_physical_rows, device, (rows, k))
            _indices("swap_counts", swap_counts, device, ndim=1)
            _require(swap_counts.numel() >= rows, "swap_counts is too short")
        if stats is not None:
            _indices("stats", stats, device, (2,), dtype=torch.uint64)
        if rows == 0 or k == 0:
            return
        _require(0 < max_union_rows <= h, "max_union_rows must be in [1, hot_size]")
        w = self.workspace
        _require(
            w.device == device
            and rows <= w.max_rows
            and nstates <= w.num_states
            and k <= w.top_k
            and h <= w.hot_size
            and max_union_rows <= w.max_union_rows,
            "resolver workspace capacity or device mismatch",
        )
        hisparse_resolve_residency_kernel[(rows,)](
            hot_block_table,
            global_indices,
            hot_indices,
            device_global_indices,
            lru_slots,
            request_state_indices,
            miss_mask,
            attention_indices,
            request_ids,
            source_block_table,
            resolved_global_indices,
            valid_counts,
            swap_host_physical_rows,
            swap_device_physical_rows,
            swap_counts,
            resident_block_table,
            num_valid_rows,
            w.keys,
            w.slots,
            w.first,
            w.loaded,
            w.order,
            w.protected,
            w.evict,
            w.hits,
            w.refs,
            w.counts,
            ROWS=rows,
            K=k,
            H=h,
            U=max_union_rows,
            HASH=w.hash_size,
            STATES=nstates,
            STATE_COUNT=0 if request_state_indices is None else request_state_indices.numel(),
            HOST_ROWS=host_cache.shape[0],
            BLOCK=block,
            HOT_STRIDE=hot_block_table.stride(0),
            INPUT_STRIDE=global_indices.stride(0),
            ATT_STRIDE=0 if attention_indices is None else attention_indices.stride(0),
            ATT_BLOCK=attention_block_stride,
            VALID_STRIDE=0 if valid_counts is None else valid_counts.stride(0),
            SOURCE_ROWS=source_rows,
            SOURCE_COLS=source_cols,
            SOURCE_STRIDE=source_stride,
            SOURCE_BLOCK=source_block_size,
            RES_ROWS=res_rows,
            RES_COLS=res_cols,
            RES_STRIDE=res_stride,
            RES_BLOCK=resident_block_size,
            RES_NULL=resident_null_block,
            WORK_K=w.top_k,
            WORK_H=w.hot_size,
            WORK_U=w.max_union_rows,
            TILE=256,
            debug=True,
        )
        if stats is not None:
            _hisparse_accumulate_stats[(1,)](w.counts, stats, ROWS=rows)

    def hisparse_gather_plan(
        self,
        host_cache,
        hot_cache,
        global_indices,
        hot_indices,
        miss_mask,
        request_state_indices,
        attention_indices=None,
        attention_block_stride=0,
    ):
        self._npu(hot_cache)
        with torch.npu.device(hot_cache.device):
            self._gather(
                host_cache,
                hot_cache,
                global_indices,
                hot_indices,
                mask=miss_mask,
                states=request_state_indices,
                attention=attention_indices,
                attention_block_stride=attention_block_stride,
            )

    def hisparse_gather_compact(self, host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts):
        self._npu(hot_cache)
        with torch.npu.device(hot_cache.device):
            self._gather(host_cache, hot_cache, miss_global_indices, miss_hot_indices, counts=miss_counts)

    def _gather(
        self, host, hot, source, target, mask=None, counts=None, states=None, attention=None, attention_block_stride=0
    ):
        compact = counts is not None
        width, block, page_bytes, hot_rows = _cache_layout(host, hot, allow_staging=not compact)
        device = hot.device
        _indices("source", source, device, ndim=2)
        rows, k = source.shape
        _indices("target", target, device, (rows, k))
        if compact:
            _indices("counts", counts, device, ndim=1)
            _require(counts.numel() >= rows, "miss_counts is too short")
        else:
            _indices("miss_mask", mask, device, (rows, k))
        if states is not None:
            _indices("states", states, device, ndim=1)
            _require(states.numel() >= rows, "request_state_indices is too short")
        if attention is not None:
            _indices("attention", attention, device, (rows, k))
            _require(attention_block_stride >= block, "attention_block_stride is shorter than a block")
        if rows == 0 or k == 0:
            return
        mapping = self.host_mapping
        _require(
            mapping is not None and torch.device(mapping.device) == device,
            "gather requires a live Host mapping for the target device",
        )
        address = mapping.address_of(host)
        _require(isinstance(address, int) and 0 < address < 2**64, "invalid mapped Host view address")
        hisparse_gather_bytes_kernel[(rows, k)](
            address,
            hot,
            source,
            target,
            mask,
            counts,
            states,
            attention,
            K=k,
            HOST_ROWS=host.shape[0],
            HOT_ROWS=hot_rows,
            ROW_BYTES=width,
            BLOCK=block,
            PAGE_BYTES=page_bytes,
            ATT_BLOCK=attention_block_stride,
            COMPACT=compact,
            TILE=256,
        )


@triton.jit(do_not_specialize=["num_tokens", "num_request_ids"])
def hisparse_invalidate_written_slots_kernel(
    device_global_indices_ptrs,
    request_state_indices,
    req_id_per_token,
    written_slots,
    num_tokens,
    num_request_ids,
    REGION_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Give each (group, state, row tile) one writer for all rewritten slots.

    This implements csrc/hisparse_kernels.cu's invalidation predicate. Owning
    the row tile avoids overlapping vector stores when several tokens rewrite
    different slots in the same state row on Ascend.
    """
    state_idx = tl.program_id(1)
    table_address = tl.load(device_global_indices_ptrs + tl.program_id(2))
    table = table_address.to(tl.pointer_type(tl.int32))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row = table + state_idx.to(tl.int64) * REGION_STRIDE + offsets
    values = tl.load(row, offsets < REGION_STRIDE, other=-1)

    for token_idx in range(num_tokens):
        req_idx = tl.load(req_id_per_token + token_idx)
        # A scalar branch guards the indirect load itself, not just its value.
        if (req_idx >= 0) & (req_idx < num_request_ids):
            token_state = tl.load(request_state_indices + req_idx)
            slot = tl.load(written_slots + token_idx)
            if (token_state == state_idx) & (slot >= 0) & (slot <= 2147483647):
                values = tl.where(values == slot.to(tl.int32), -1, values)

    tl.store(row, values, offsets < REGION_STRIDE)


def hisparse_invalidate_written_slots(
    device_global_indices_ptrs: torch.Tensor,
    request_state_indices: torch.Tensor,
    req_id_per_token: torch.Tensor,
    written_slots: torch.Tensor,
    num_state_rows: int,
    region_stride: int,
) -> None:
    """Invalidate rewritten host slots in every listed group's hot index table.

    The six arguments match the upstream CUDA operator. ``ptrs`` is a contiguous
    int64 NPU vector of live, contiguous int32 [num_state_rows, region_stride]
    table addresses, retained by the caller through stream completion. Request
    states and request IDs are contiguous int32 vectors; written slots are
    contiguous int64. Invalid IDs, states and slots are ignored. Tables mutate
    in place, with no payload copy, host scalar reads or synchronization.

    Work is enqueued on the selected device's current stream. Address validity
    and table geometry are caller contracts, as in the CUDA implementation.
    """
    tensors = (
        ("device_global_indices_ptrs", device_global_indices_ptrs, torch.int64),
        ("request_state_indices", request_state_indices, torch.int32),
        ("req_id_per_token", req_id_per_token, torch.int32),
        ("written_slots", written_slots, torch.int64),
    )
    device = device_global_indices_ptrs.device
    for name, tensor, dtype in tensors:
        if tensor.device.type != "npu" or tensor.device != device:
            raise RuntimeError("HiSparse invalidation tensors must be on the same NPU device")
        if tensor.dtype != dtype or tensor.ndim != 1 or not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be a contiguous 1D {dtype} NPU tensor")
    if req_id_per_token.numel() != written_slots.numel():
        raise RuntimeError("req_id_per_token and written_slots must have equal length")
    if num_state_rows < 0 or region_stride < 0:
        raise RuntimeError("HiSparse invalidation table dimensions must be nonnegative")
    num_tokens = written_slots.numel()
    num_groups = device_global_indices_ptrs.numel()
    if num_tokens == 0 or num_groups == 0 or num_state_rows == 0 or region_stride == 0:
        return
    if num_tokens > _INT32_MAX or num_groups > _MAX_INDEX_GROUPS:
        raise RuntimeError("HiSparse invalidation grid exceeds supported limits")
    with torch.npu.device(device):
        hisparse_invalidate_written_slots_kernel[
            (triton.cdiv(region_stride, _INVALIDATE_BLOCK_SIZE), num_state_rows, num_groups)
        ](
            device_global_indices_ptrs,
            request_state_indices,
            req_id_per_token,
            written_slots,
            num_tokens,
            request_state_indices.numel(),
            REGION_STRIDE=region_stride,
            BLOCK_SIZE=_INVALIDATE_BLOCK_SIZE,
        )
