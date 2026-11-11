# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiSparse NPU cache operations and fixed resolver workspaces.

Decode planning and payload copies stay on the NPU, including during ACL graph
replay. Workspaces are owned by the bound cache operations.
"""

from __future__ import annotations

import math

import torch
from vllm.triton_utils import triton

from vllm_ascend.device.device_config import is_950
from vllm_ascend.hisparse.host_pool import MappedHostTensor
from vllm_ascend.ops.triton.hisparse import (
    gather_resident_rows_kernel,
    gather_rows_kernel,
    invalidate_kernel,
    invalidate_written_slots_sorted,
    pack_attention_rows_kernel,
    pack_written_slot_keys,
    scatter_packed_kv_kernel,
)

_SORT_MIN_DIRECT_TILES = 64 * 1024
_SORT_BUFFER_LIMIT_BYTES = 768 * 1024


def _load_resolver():
    namespace = torch.ops._C_ascend
    if not all(
        hasattr(namespace, name) for name in ("hisparse_resolver_validate_scratch", "hisparse_resolve_residency")
    ):
        raise RuntimeError("HiSparse requires its compiled AscendC resolver. Rebuild vllm-ascend for this target.")
    return namespace


class ResolverWorkspace:
    """Fixed scratch shared by one index group on its ordered copy stream."""

    def __init__(
        self, max_rows: int, hot_size: int, max_union_rows: int, device: torch.device, *, max_topk: int | None = None
    ) -> None:
        if not 0 < max_union_rows <= hot_size <= 32768 or max_rows <= 0:
            raise ValueError("HiSparse requires 0 < union <= hot_size <= 32768 and positive max_rows")
        self.max_rows = max_rows
        self.hot_size = hot_size
        self.max_union_rows = max_union_rows
        self.max_topk = hot_size if max_topk is None else max_topk
        if not 0 < self.max_topk <= hot_size:
            raise ValueError("HiSparse resolver TopK capacity must be positive and fit the Hot cache")
        self.native = _load_resolver()
        self.native.hisparse_resolver_validate_scratch(hot_size, max_union_rows)
        # One resolved Host row per selected token, not per Hot-cache slot.
        self.first_refs = torch.empty(max_rows * self.max_topk, dtype=torch.int32, device=device)
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

    def __init__(
        self,
        host_mapping: MappedHostTensor,
        workspace: ResolverWorkspace | None = None,
        *,
        attention_capacity: int = 0,
        attention_block_stride: int | None = None,
        attention_lengths: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        self.host_mapping = host_mapping
        self.workspace = workspace
        self.attention_capacity = attention_capacity
        self.attention_block_stride = attention_block_stride
        self.attention_lengths = attention_lengths

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
        resident_null_block=0,
        num_valid_rows=None,
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
        if num_valid_rows is not None:
            _check_tensor(num_valid_rows, "num_valid_rows", device, (torch.int32,))
            if num_valid_rows.numel() != 1:
                raise ValueError("num_valid_rows must contain one device scalar")
        hot_size = lru_slots.size(1)
        if rows == 0:
            return
        if (
            rows > workspace.max_rows
            or topk > workspace.max_topk
            or hot_size != workspace.hot_size
            or max_union_rows != workspace.max_union_rows
        ):
            raise ValueError("HiSparse resolver input exceeds its prepared workspace")
        if region_stride != device_global_indices.size(1) or region_stride != hot_size or hot_size < topk:
            raise ValueError("HiSparse region and LRU widths must match and cover top-k")
        if topk <= 0:
            raise ValueError("HiSparse top-k must be positive")
        if workspace.first_refs.device != device:
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
        if (
            len({tensor is not None for tensor in (swap_counts, swap_host_physical_rows, swap_device_physical_rows)})
            != 1
        ):
            raise ValueError("All three compact HiSparse swap outputs are required together")
        self._resolve(workspace, locals())

    def _resolve(self, workspace: ResolverWorkspace, arguments) -> None:
        """Dispatch live tensors to the native request-union resolver.

        During replay, external tensors remain owned by the runtime/workspace.
        Callers must destroy graphs before releasing or rebinding those owners.
        """
        rows, topk = arguments["global_indices"].shape
        resolved = arguments["resolved_global_indices"]
        if resolved is None:
            resolved = workspace.first_refs[: rows * topk].view(rows, topk)
        workspace.native.hisparse_resolve_residency(
            arguments["global_indices"],
            arguments["request_ids"],
            arguments["request_state_indices"],
            arguments["source_block_table"],
            arguments["resident_block_table"],
            arguments["hot_block_table"],
            arguments["device_global_indices"],
            arguments["lru_slots"],
            arguments["hot_indices"],
            arguments["attention_indices"],
            resolved,
            arguments["valid_counts"],
            arguments["swap_host_physical_rows"],
            arguments["swap_device_physical_rows"],
            arguments["swap_counts"],
            arguments["miss_mask"],
            arguments["stats"],
            workspace.errors[:rows],
            arguments["num_valid_rows"],
            arguments["host_cache"].shape[0],
            arguments["hot_cache"].shape[1],
            arguments["source_block_size"],
            arguments["resident_block_size"],
            arguments["resident_null_block"],
            arguments["attention_block_stride"] if self.attention_block_stride is None else self.attention_block_stride,
            arguments["max_union_rows"],
            self.attention_capacity,
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
        request_state_indices=None,
        attention_indices=None,
        attention_block_stride=0,
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
        if rows == 0 or topk == 0:
            return
        parallel_columns = min(topk, 128) if counts is not None else 0
        gather_rows_kernel[(parallel_columns, rows) if parallel_columns else (rows, topk)](
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
            HAS_ATTENTION=attention is not None,
            HAS_STATE=state is not None,
            BLOCK=min(1024, triton.next_power_of_2(row_bytes)),
            PARALLEL_COLUMNS=parallel_columns,
            multibuffer=False,
        )

    @staticmethod
    def hisparse_invalidate_written_slots(
        group_table,
        request_state_indices,
        req_id_per_token,
        written_slots,
        num_state_rows,
        region_stride,
    ) -> None:
        device = group_table.device
        if device.type != "npu":
            raise ValueError("HiSparse invalidation requires NPU state")
        if num_state_rows < 0 or region_stride < 0:
            raise ValueError("HiSparse invalidation requires nonnegative state and region sizes")
        _check_tensor(group_table, "group_table", device, (torch.int64,), ndim=1)
        groups = group_table.numel()
        _check_tensor(request_state_indices, "request_state_indices", device, (torch.int32,), ndim=1)
        _check_tensor(written_slots, "written_slots", device, (torch.int32, torch.int64), ndim=1)
        _check_tensor(req_id_per_token, "req_id_per_token", device, (torch.int32,), shape=written_slots.shape)
        num_tokens = written_slots.numel()
        if num_tokens == 0:
            return
        if region_stride == 0 or num_state_rows == 0 or groups == 0:
            return
        tile = 256
        hot_tiles = triton.cdiv(region_stride, tile)
        table_tiles = groups * hot_tiles
        search_steps = num_tokens.bit_length()
        # Direct invalidation scans the Hot tiles for each write. Sorting
        # instead searches once per state and Hot tile. Keep a margin for
        # sorting and cap the three int64 buffers before launching it.
        direct_tiles = num_tokens * table_tiles
        search_probes = num_state_rows * table_tiles * search_steps
        sort_buffer_bytes = 3 * num_tokens * torch.int64.itemsize
        if (
            direct_tiles >= 2 * search_probes + _SORT_MIN_DIRECT_TILES
            and sort_buffer_bytes <= _SORT_BUFFER_LIMIT_BYTES
            and is_950()
        ):
            keys = torch.empty((num_tokens,), dtype=torch.int64, device=device)
            sorted_keys = torch.empty_like(keys)
            order = torch.empty_like(keys)
            pack_written_slot_keys[(triton.cdiv(num_tokens, tile),)](
                request_state_indices,
                req_id_per_token,
                written_slots,
                keys,
                ROWS=num_tokens,
                REQUESTS=request_state_indices.numel(),
                STATES=num_state_rows,
                TILE=tile,
                multibuffer=False,
            )
            torch.sort(keys, out=(sorted_keys, order))
            invalidate_written_slots_sorted[(groups, num_state_rows, hot_tiles)](
                group_table,
                sorted_keys,
                ROWS=num_tokens,
                HOT=region_stride,
                TILE=tile,
                STEPS=search_steps,
                multibuffer=False,
            )
            return
        invalidate_kernel[(num_tokens, hot_tiles, groups)](
            group_table,
            request_state_indices,
            req_id_per_token,
            written_slots,
            REQUESTS=request_state_indices.numel(),
            STATE_ROWS=num_state_rows,
            REGION=region_stride,
            BLOCK=tile,
            multibuffer=False,
        )


def gather_resident_rows(source: torch.Tensor, destination: torch.Tensor, row_ids: torch.Tensor) -> None:
    """Copy packed Resident rows using Core's negative-ID-to-row0 convention.

    Source and destination may use different dtypes with the same byte width.
    """
    _check_packed_cache(source)
    if destination.ndim == 2:
        destination = destination.unsqueeze(1)
    if destination.ndim != 3:
        raise ValueError("HiSparse Resident staging must contain flat or paged token rows")
    rows = destination.size(0) * destination.size(1)
    _check_tensor(row_ids, "row_ids", source.device, (torch.int32, torch.int64), shape=(rows,))
    row_bytes = source.size(2) * source.element_size()
    if (
        source.device != destination.device
        or row_bytes != destination.size(2) * destination.element_size()
        or source.untyped_storage().data_ptr() == destination.untyped_storage().data_ptr()
    ):
        raise ValueError("HiSparse Resident gather requires matching, separate source and staging rows")
    if rows == 0:
        return
    _check_packed_cache(destination)
    block_rows = 8
    block_bytes = min(1024, triton.next_power_of_2(row_bytes))
    grid = (triton.cdiv(rows, block_rows), triton.cdiv(row_bytes, block_bytes))
    gather_resident_rows_kernel[grid](
        source.view(torch.uint8),
        destination.view(torch.uint8),
        row_ids,
        ROWS=rows,
        SOURCE_ROWS=source.size(0) * source.size(1),
        SOURCE_BLOCK=source.size(1),
        SOURCE_STRIDE=source.stride(0) * source.element_size(),
        DEST_BLOCK=destination.size(1),
        DEST_STRIDE=destination.stride(0) * destination.element_size(),
        ROW_BYTES=row_bytes,
        BLOCK_ROWS=block_rows,
        BLOCK_BYTES=block_bytes,
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
