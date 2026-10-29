# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend HiSparse operators and explicit mapped-host memory ownership.

Mapping and scratch allocation happen at cache initialization. Decode planning
and payload copies stay on the NPU, including during ACL graph replay.
"""

from __future__ import annotations

import ctypes
import math
import mmap

import torch
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.hisparse import (
    assign_residency_kernel,
    classify_residency_kernel,
    finish_residency_kernel,
    gather_rows_kernel,
    invalidate_kernel,
    pack_attention_rows_kernel,
    partition_residency_kernel,
    prepare_residency_kernel,
    protect_residency_kernel,
    residency_stats_kernel,
    scatter_packed_kv_kernel,
    update_lru_kernel,
)

_ACL_HOST_REGISTER_MAPPED = 0
_ACL_HAC_AIV = 3


class MappedHostTensor:
    """Own an ACL mapping and retain the CPU allocation until explicit close.

    Close only after all streams and graphs using this mapping have completed
    and been destroyed. No destructor unmaps memory behind asynchronous work.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("HiSparse host mapping requires a contiguous CPU tensor")
        self.tensor = tensor
        self._lib = ctypes.CDLL("libascendcl.so")
        self._lib.aclrtHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.aclrtHostRegister.restype = ctypes.c_int
        self._lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
        self._lib.aclrtHostUnregister.restype = ctypes.c_int
        self._lib.aclrtHostMemMapCapabilities.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self._lib.aclrtHostMemMapCapabilities.restype = ctypes.c_int
        device_index = torch.npu.current_device()
        self.device = torch.device("npu", device_index)
        capability = ctypes.c_int()
        status = self._lib.aclrtHostMemMapCapabilities(device_index, _ACL_HAC_AIV, ctypes.byref(capability))
        if status or not capability.value:
            raise RuntimeError(
                f"HiSparse requires AIV host-memory mapping: status={status}, capability={capability.value}"
            )
        # The upstream host-pool allocator supplies an aligned subview of a
        # larger backing allocation. Register that range, not its unaligned
        # storage base; retained tensor ownership still retains the backing.
        self.host_address = tensor.data_ptr()
        self.nbytes = tensor.numel() * tensor.element_size()
        if not self.nbytes:
            raise ValueError("Cannot register an empty HiSparse host allocation")
        mapped = ctypes.c_void_p()
        status = self._lib.aclrtHostRegister(
            self.host_address, self.nbytes, _ACL_HOST_REGISTER_MAPPED, ctypes.byref(mapped)
        )
        if status:
            raise RuntimeError(f"aclrtHostRegister for HiSparse failed: {status}")
        self.device_ptr = int(mapped.value)
        self.closed = False

    def address_of(self, tensor: torch.Tensor) -> int:
        offset = tensor.data_ptr() - self.host_address
        size = tensor.numel() * tensor.element_size()
        if self.closed or tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("HiSparse source must be a contiguous view of a live mapped allocation")
        if offset < 0 or offset + size > self.nbytes:
            raise ValueError("HiSparse source is outside its registered host allocation")
        return self.device_ptr + offset

    def close(self) -> None:
        if not self.closed:
            status = self._lib.aclrtHostUnregister(self.host_address)
            if status:
                raise RuntimeError(f"aclrtHostUnregister for HiSparse failed: {status}")
            self.closed = True


def allocate_mapped_host(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Allocate page-aligned CPU storage and attach its explicit mapping owner."""
    count = math.prod(shape)
    nbytes = count * torch.empty((), dtype=dtype).element_size()
    if nbytes <= 0:
        raise ValueError("HiSparse host allocation must be nonempty")
    allocated_bytes = triton.cdiv(nbytes, mmap.PAGESIZE) * mmap.PAGESIZE
    owner = mmap.mmap(-1, allocated_bytes)
    raw = torch.frombuffer(owner, dtype=torch.uint8)
    mapping = MappedHostTensor(raw)
    tensor = raw[:nbytes].view(dtype).view(shape)
    # The mapping owns raw; frombuffer keeps the mmap alive. Runtime owns the
    # registration through this base tensor until its explicit shutdown.
    tensor._hisparse_registration = mapping
    return tensor


class ResolverWorkspace:
    """Fixed scratch shared by one index group on its ordered copy stream."""

    def __init__(self, max_rows: int, hot_size: int, max_union_rows: int, device: torch.device) -> None:
        if not 0 < max_union_rows <= hot_size <= 32768 or max_rows <= 0:
            raise ValueError("HiSparse requires 0 < union <= hot_size <= 32768 and positive max_rows")
        self.max_rows = max_rows
        self.hot_size = hot_size
        self.max_union_rows = max_union_rows
        # Packed int64 keys retain exact int32 host IDs and deterministic first
        # reference positions. torch.sort's out= ABI reuses these NPU buffers.
        self.row_keys = torch.empty(max_rows * hot_size, dtype=torch.int64, device=device)
        self.sorted_rows = torch.empty_like(self.row_keys)
        self.row_sort_indices = torch.empty_like(self.row_keys)
        self.cache_keys = torch.empty((max_rows, hot_size), dtype=torch.int64, device=device)
        self.sorted_cache = torch.empty_like(self.cache_keys)
        self.cache_sort_indices = torch.empty_like(self.cache_keys)
        self.first_refs = torch.empty(max_rows * hot_size, dtype=torch.int32, device=device)
        self.first_misses = torch.empty_like(self.first_refs)
        self.metadata = torch.empty((max_rows, 8), dtype=torch.int32, device=device)
        self.row_stats = torch.empty((max_rows, 4), dtype=torch.int64, device=device)
        self.lru_temp = torch.empty((max_rows, hot_size), dtype=torch.int32, device=device)
        self.protected = torch.empty_like(self.lru_temp)
        self.hit_prefix = torch.empty_like(self.lru_temp)
        self.errors = torch.zeros((max_rows, 8), dtype=torch.int32, device=device)


def _cache_layout(host_cache: torch.Tensor, hot_cache: torch.Tensor) -> tuple[int, int, int, int]:
    if host_cache.ndim != 2 or not host_cache.is_contiguous() or host_cache.device.type != "cpu":
        raise ValueError("HiSparse host cache must be a contiguous CPU [rows, width] view")
    _check_packed_cache(hot_cache)
    row_bytes = hot_cache.size(2) * hot_cache.element_size()
    if hot_cache.stride(1) != hot_cache.size(2) or host_cache.size(1) * host_cache.element_size() != row_bytes:
        raise ValueError("HiSparse cache rows must be dense and have matching byte widths")
    return (
        row_bytes,
        hot_cache.size(1),
        hot_cache.size(0) * hot_cache.size(1),
        hot_cache.stride(0) * hot_cache.element_size(),
    )


def _check_tensor(tensor, name, device, dtypes, *, ndim=None, shape=None, contiguous=True) -> None:
    """Validate metadata without copying values or synchronizing the device."""
    if tensor.device != device or tensor.dtype not in dtypes:
        raise ValueError(f"{name} must have dtype {dtypes} on {device}")
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_packed_cache(cache) -> None:
    if cache.device.type != "npu" or cache.ndim != 3 or min(cache.shape) <= 0:
        raise ValueError("HiSparse device cache must be nonempty NPU [blocks, block_size, width]")
    if cache.stride(2) != 1 or cache.stride(1) != cache.size(2):
        raise ValueError("HiSparse device cache must have dense token rows")
    if cache.stride(0) < cache.size(1) * cache.size(2):
        raise ValueError("HiSparse cache blocks must not overlap")
    if cache.size(0) * cache.size(1) >= 2**31:
        raise ValueError("HiSparse physical row IDs must fit int32")


def _check_payload_rows(tensor, name, device, dtype, rows) -> None:
    _check_tensor(tensor, name, device, (dtype,), contiguous=False)
    if tensor.ndim < 2 or tensor.size(0) < rows:
        raise ValueError(f"{name} must contain at least {rows} token rows")
    stride = 1
    for dim in range(tensor.ndim - 1, 0, -1):
        if tensor.size(dim) > 1 and tensor.stride(dim) != stride:
            raise ValueError(f"{name} must have dense token rows")
        stride *= tensor.size(dim)
    if tensor.stride(0) < stride:
        raise ValueError(f"{name} token rows must not overlap")


class HiSparseNPUOps:
    """Runtime-owned implementations of the upstream HiSparse operator ABI."""

    def __init__(self, host_mapping: MappedHostTensor, workspace: ResolverWorkspace | None = None) -> None:
        self.host_mapping = host_mapping
        self.workspace = workspace

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
    ) -> None:
        """Resolve logical selections and prepare resident/hot-cache refill plans.

        Protect every selected token in a request's query-row union before
        evicting LRU slots. Update residency and output plans in place using the
        persistent workspace; host KV payloads are copied by the gather stage.
        """
        workspace = self.workspace
        if workspace is None:
            raise RuntimeError("HiSparse resolver workspace must be prepared before forward")
        _cache_layout(host_cache, hot_cache)
        device = hot_cache.device
        _check_tensor(global_indices, "global_indices", device, (torch.int32,), ndim=2, contiguous=False)
        if global_indices.stride(1) != 1:
            raise ValueError("HiSparse top-k columns must be contiguous")
        _check_tensor(device_global_indices, "device_global_indices", device, (torch.int32,), ndim=2)
        _check_tensor(lru_slots, "lru_slots", device, (torch.int16,), shape=device_global_indices.shape)
        rows, topk = global_indices.shape
        hot_size = lru_slots.size(1)
        if rows == 0:
            return
        if rows > workspace.max_rows or hot_size != workspace.hot_size or max_union_rows != workspace.max_union_rows:
            raise ValueError("HiSparse resolver input exceeds its prepared workspace")
        if region_stride != device_global_indices.size(1) or region_stride != hot_size or hot_size < topk:
            raise ValueError("HiSparse region and LRU widths must match and cover top-k")
        if topk <= 0 or topk % 8 or hot_size % 16:
            raise ValueError("Ascend HiSparse requires top-k aligned to 8 and hot capacity aligned to 16")
        if workspace.row_keys.device != device:
            raise ValueError("HiSparse resolver workspace must be on the cache device")
        if host_cache.size(0) >= 2**31:
            raise ValueError("HiSparse host row IDs must fit int32")
        for name, tensor in (
            ("hot_indices", hot_indices),
            ("resolved_global_indices", resolved_global_indices),
            ("swap_host_physical_rows", swap_host_physical_rows),
            ("swap_device_physical_rows", swap_device_physical_rows),
        ):
            if tensor is not None:
                _check_tensor(tensor, name, device, (torch.int32,), shape=(rows, topk))
        if request_ids is not None:
            _check_tensor(request_ids, "request_ids", device, (torch.int32,), shape=(rows,))
        if request_state_indices is not None:
            _check_tensor(request_state_indices, "request_state_indices", device, (torch.int32,), ndim=1)
        _check_tensor(hot_block_table, "hot_block_table", device, (torch.int32,), ndim=2, contiguous=False)
        request_capacity = (
            device_global_indices.size(0) if request_state_indices is None else request_state_indices.numel()
        )
        if (
            hot_block_table.stride(1) != 1
            or hot_block_table.size(0) < request_capacity
            or hot_block_table.size(1) < triton.cdiv(hot_size, hot_cache.size(1))
        ):
            raise ValueError("HiSparse hot block table does not cover all state slots")
        for name, table, block_size in (
            ("source_block_table", source_block_table, source_block_size),
            ("resident_block_table", resident_block_table, resident_block_size),
        ):
            if table is not None:
                _check_tensor(table, name, device, (torch.int32,), ndim=2, contiguous=False)
                if table.stride(1) != 1 or block_size <= 0:
                    raise ValueError(f"{name} requires contiguous columns and a positive block size")
        if attention_indices is not None:
            _check_tensor(
                attention_indices, "attention_indices", device, (torch.int32,), shape=(rows, topk), contiguous=False
            )
            if attention_indices.stride(1) != 1 or attention_block_stride < hot_cache.size(1):
                raise ValueError("HiSparse attention indices require dense columns and a valid block stride")
        if miss_mask is not None:
            _check_tensor(miss_mask, "miss_mask", device, (torch.uint8, torch.bool, torch.int32), shape=(rows, topk))
        if valid_counts is not None:
            _check_tensor(valid_counts, "valid_counts", device, (torch.int32,), contiguous=False)
            if tuple(valid_counts.shape) not in ((rows,), (rows, 1)):
                raise ValueError("valid_counts must have one entry per query row")
        if swap_counts is not None:
            _check_tensor(swap_counts, "swap_counts", device, (torch.int32,), shape=(rows,))
        if stats is not None:
            _check_tensor(stats, "stats", device, (torch.int64,), shape=(2,))
        dummy = hot_indices
        source = source_block_table
        resident = resident_block_table
        has_swaps = swap_counts is not None
        if (
            len({tensor is not None for tensor in (swap_counts, swap_host_physical_rows, swap_device_physical_rows)})
            != 1
        ):
            raise ValueError("All three compact HiSparse swap outputs are required together")
        row_keys = workspace.row_keys[: rows * topk].view(rows, topk)
        sorted_rows = workspace.sorted_rows[: rows * topk].view(rows, topk)
        row_sort_indices = workspace.row_sort_indices[: rows * topk].view(rows, topk)
        cache_keys = workspace.cache_keys[:rows]
        sorted_cache = workspace.sorted_cache[:rows]
        prepare_residency_kernel[(rows,)](
            global_indices,
            request_ids if request_ids is not None else dummy,
            request_state_indices if request_state_indices is not None else dummy,
            source if source is not None else dummy,
            resident if resident is not None else dummy,
            device_global_indices,
            hot_indices,
            resolved_global_indices if resolved_global_indices is not None else dummy,
            valid_counts if valid_counts is not None else dummy,
            row_keys,
            cache_keys,
            workspace.metadata,
            ROWS=rows,
            TOP_K=topk,
            HOT_SIZE=hot_size,
            INPUT_STRIDE=global_indices.stride(0),
            COUNT_STRIDE=valid_counts.stride(0) if valid_counts is not None else 1,
            STATE_ROWS=device_global_indices.size(0),
            STATE_COUNT=request_state_indices.numel() if request_state_indices is not None else 0,
            HOST_ROWS=host_cache.size(0),
            SOURCE_STRIDE=source.stride(0) if source is not None else 0,
            SOURCE_ROWS=source.size(0) if source is not None else 0,
            SOURCE_COLS=source.size(1) if source is not None else 0,
            SOURCE_BLOCK=source_block_size if source is not None else 1,
            RESIDENT_STRIDE=resident.stride(0) if resident is not None else 0,
            RESIDENT_ROWS=resident.size(0) if resident is not None else 0,
            RESIDENT_COLS=resident.size(1) if resident is not None else 0,
            RESIDENT_BLOCK=resident_block_size if resident is not None else 1,
            RESIDENT_NULL=resident_null_block,
            HAS_REQUESTS=request_ids is not None,
            HAS_STATE=request_state_indices is not None,
            HAS_SOURCE=source is not None,
            HAS_RESIDENT=resident is not None,
            HAS_RESOLVED=resolved_global_indices is not None,
            HAS_COUNTS=valid_counts is not None,
            BK=triton.next_power_of_2(topk),
            multibuffer=False,
        )
        torch.sort(row_keys, dim=-1, out=(sorted_rows, row_sort_indices))
        torch.sort(cache_keys, dim=-1, out=(sorted_cache, workspace.cache_sort_indices[:rows]))
        classify_residency_kernel[(rows,)](
            row_keys,
            sorted_rows,
            sorted_cache,
            workspace.metadata,
            hot_block_table,
            hot_indices,
            workspace.first_refs,
            workspace.first_misses,
            workspace.row_stats,
            TOP_K=topk,
            HOT_SIZE=hot_size,
            HOT_BLOCK=hot_cache.size(1),
            TABLE_STRIDE=hot_block_table.stride(0),
            BK=triton.next_power_of_2(topk),
            multibuffer=False,
        )
        protect_residency_kernel[(rows, triton.cdiv(hot_size, 256))](
            sorted_rows,
            workspace.metadata,
            device_global_indices,
            lru_slots,
            workspace.protected,
            TOP_K=topk,
            HOT_SIZE=hot_size,
            STATE_ROWS=device_global_indices.size(0),
            BLOCK=256,
            multibuffer=False,
        )
        torch.cumsum(workspace.protected[:rows], dim=-1, dtype=torch.int32, out=workspace.hit_prefix[:rows])
        partition_residency_kernel[(rows, triton.cdiv(hot_size, 256))](
            workspace.metadata,
            workspace.protected,
            workspace.hit_prefix,
            workspace.row_stats,
            lru_slots,
            workspace.lru_temp,
            workspace.errors,
            HOT_SIZE=hot_size,
            MAX_UNION=max_union_rows,
            STATE_ROWS=device_global_indices.size(0),
            BLOCK=256,
            multibuffer=False,
        )
        update_lru_kernel[(rows, triton.cdiv(hot_size, 256))](
            workspace.metadata,
            workspace.lru_temp,
            lru_slots,
            workspace.errors,
            HOT_SIZE=hot_size,
            BLOCK=256,
            multibuffer=False,
        )
        assign_residency_kernel[(rows,)](
            row_keys,
            workspace.metadata,
            workspace.first_misses,
            hot_block_table,
            hot_indices,
            device_global_indices,
            workspace.lru_temp,
            workspace.errors,
            swap_host_physical_rows if has_swaps else dummy,
            swap_device_physical_rows if has_swaps else dummy,
            swap_counts if has_swaps else dummy,
            TOP_K=topk,
            HOT_SIZE=hot_size,
            HOT_BLOCK=hot_cache.size(1),
            TABLE_STRIDE=hot_block_table.stride(0),
            HAS_SWAPS=has_swaps,
            BK=triton.next_power_of_2(topk),
            multibuffer=False,
        )
        finish_residency_kernel[(rows,)](
            workspace.metadata,
            workspace.first_refs,
            workspace.first_misses,
            hot_indices,
            attention_indices if attention_indices is not None else dummy,
            miss_mask if miss_mask is not None else dummy,
            valid_counts if valid_counts is not None else dummy,
            workspace.errors,
            TOP_K=topk,
            HOT_BLOCK=hot_cache.size(1),
            ATTENTION_BLOCK=attention_block_stride,
            ATTENTION_STRIDE=attention_indices.stride(0) if attention_indices is not None else topk,
            COUNT_STRIDE=valid_counts.stride(0) if valid_counts is not None else 1,
            HAS_ATTENTION=attention_indices is not None,
            HAS_MISS_MASK=miss_mask is not None,
            HAS_COUNTS=valid_counts is not None,
            BK=triton.next_power_of_2(topk),
            multibuffer=False,
        )
        if stats is not None:
            residency_stats_kernel[(1,)](
                workspace.row_stats,
                stats,
                ROWS=rows,
                BLOCK=triton.next_power_of_2(rows),
                multibuffer=False,
            )

    def hisparse_gather_compact(
        self, host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts
    ) -> None:
        self._gather(host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts, None, None, None, 0)

    def hisparse_gather_plan(
        self,
        host_cache,
        hot_cache,
        global_indices,
        hot_indices,
        miss_mask,
        request_state_indices,
        attention_indices,
        attention_block_stride,
    ) -> None:
        self._gather(
            host_cache,
            hot_cache,
            global_indices,
            hot_indices,
            None,
            miss_mask,
            request_state_indices,
            attention_indices,
            attention_block_stride,
        )

    def _gather(
        self, host_cache, hot_cache, source, destination, counts, mask, state, attention, attention_block
    ) -> None:
        row_bytes, block_size, hot_rows, hot_stride = _cache_layout(host_cache, hot_cache)
        device = hot_cache.device
        if self.host_mapping.device != device:
            raise ValueError("HiSparse host registration must belong to the cache device")
        _check_tensor(source, "source", device, (torch.int32,), ndim=2)
        _check_tensor(destination, "destination", device, (torch.int32,), shape=source.shape)
        rows, topk = source.shape
        if counts is not None:
            _check_tensor(counts, "counts", device, (torch.int32,), shape=(rows,))
        elif mask is not None:
            _check_tensor(mask, "mask", device, (torch.uint8, torch.bool, torch.int32), shape=source.shape)
        else:
            raise ValueError("HiSparse gather requires compact counts or a miss mask")
        if state is not None:
            _check_tensor(state, "state", device, (torch.int32,), shape=(rows,))
        if attention is not None:
            _check_tensor(attention, "attention", device, (torch.int32,), shape=source.shape)
            if attention_block < block_size:
                raise ValueError("HiSparse attention block stride must cover a physical block")
        if rows == 0:
            return
        gather_rows_kernel[(rows, topk)](
            self.host_mapping.address_of(host_cache),
            hot_cache.view(torch.uint8),
            source,
            destination,
            counts if counts is not None else source,
            mask if mask is not None else source,
            attention if attention is not None else source,
            state if state is not None else source,
            HOST_ROWS=host_cache.size(0),
            HOT_ROWS=hot_rows,
            ROW_BYTES=row_bytes,
            HOT_BLOCK=block_size,
            HOT_STRIDE_BYTES=hot_stride,
            TOP_K=topk,
            ATTENTION_BLOCK=attention_block,
            COMPACT=counts is not None,
            HAS_ATTENTION=attention is not None,
            HAS_STATE=state is not None,
            BLOCK=min(1024, triton.next_power_of_2(row_bytes)),
            multibuffer=False,
        )

    @staticmethod
    def hisparse_invalidate_written_slots(
        device_global_indices, request_state_indices, req_id_per_token, written_slots
    ) -> None:
        device = device_global_indices.device
        if device.type != "npu":
            raise ValueError("HiSparse invalidation requires NPU state")
        _check_tensor(device_global_indices, "device_global_indices", device, (torch.int32,), ndim=2)
        _check_tensor(request_state_indices, "request_state_indices", device, (torch.int32,), ndim=1)
        _check_tensor(written_slots, "written_slots", device, (torch.int32, torch.int64), ndim=1)
        _check_tensor(req_id_per_token, "req_id_per_token", device, (torch.int32,), shape=written_slots.shape)
        if written_slots.numel() == 0:
            return
        region = device_global_indices.size(1)
        if region == 0:
            return
        invalidate_kernel[(written_slots.numel(), triton.cdiv(region, 256))](
            device_global_indices,
            request_state_indices,
            req_id_per_token,
            written_slots,
            REQUESTS=request_state_indices.numel(),
            STATE_ROWS=device_global_indices.size(0),
            REGION=region,
            BLOCK=256,
            multibuffer=False,
        )


def pack_attention_rows(packed_cache, physical_topk, selected_latent_out, selected_rope_out, latent_dim) -> None:
    """Gather packed physical KV rows into the split, dense SFA input layout."""
    _check_packed_cache(packed_cache)
    device = packed_cache.device
    _check_tensor(physical_topk, "physical_topk", device, (torch.int32,), ndim=2, contiguous=False)
    if physical_topk.stride(1) != 1:
        raise ValueError("HiSparse attention top-k columns must be contiguous")
    rows, topk = physical_topk.shape
    width = packed_cache.size(-1)
    if not 0 < latent_dim < width:
        raise ValueError("HiSparse attention requires nonempty latent and RoPE rows")
    _check_tensor(selected_latent_out, "selected_latent_out", device, (packed_cache.dtype,))
    _check_tensor(selected_rope_out, "selected_rope_out", device, (packed_cache.dtype,))
    if selected_latent_out.numel() != rows * topk * latent_dim or selected_rope_out.numel() != rows * topk * (
        width - latent_dim
    ):
        raise ValueError("HiSparse attention staging shapes do not match the selected rows")
    if rows:
        pack_attention_rows_kernel[(rows, topk)](
            packed_cache,
            physical_topk,
            selected_latent_out,
            selected_rope_out,
            K=topk,
            CACHE_ROWS=packed_cache.size(0) * packed_cache.size(1),
            CACHE_BLOCK=packed_cache.size(1),
            CACHE_STRIDE=packed_cache.stride(0),
            WIDTH=width,
            LATENT=latent_dim,
            INDEX_STRIDE=physical_topk.stride(0),
            BLOCK=triton.next_power_of_2(width),
            multibuffer=False,
        )


def scatter_packed_kv(k_nope, k_rope, packed_cache, slot_mapping, num_valid_rows=None) -> None:
    """Write current KV with explicit page strides; ignore padding slots."""
    _check_packed_cache(packed_cache)
    device = packed_cache.device
    _check_tensor(slot_mapping, "slot_mapping", device, (torch.int32, torch.int64), ndim=1)
    rows = slot_mapping.numel()
    _check_payload_rows(k_nope, "k_nope", device, packed_cache.dtype, rows)
    _check_payload_rows(k_rope, "k_rope", device, packed_cache.dtype, rows)
    if num_valid_rows is not None:
        _check_tensor(num_valid_rows, "num_valid_rows", device, (torch.int32, torch.int64))
        if num_valid_rows.numel() != 1:
            raise ValueError("num_valid_rows must contain one device scalar")
    latent_dim = math.prod(k_nope.shape[1:])
    rope_dim = math.prod(k_rope.shape[1:])
    if latent_dim + rope_dim != packed_cache.size(-1):
        raise ValueError("HiSparse packed row width must equal latent + RoPE widths")
    if rows:
        scatter_packed_kv_kernel[(rows,)](
            k_nope,
            k_rope,
            packed_cache,
            slot_mapping,
            num_valid_rows if num_valid_rows is not None else slot_mapping,
            CACHE_ROWS=packed_cache.size(0) * packed_cache.size(1),
            CACHE_BLOCK=packed_cache.size(1),
            CACHE_STRIDE=packed_cache.stride(0),
            WIDTH=latent_dim + rope_dim,
            LATENT=latent_dim,
            LATENT_STRIDE=k_nope.stride(0),
            ROPE_STRIDE=k_rope.stride(0),
            HAS_VALID=num_valid_rows is not None,
            BLOCK=triton.next_power_of_2(latent_dim + rope_dim),
            multibuffer=False,
        )
