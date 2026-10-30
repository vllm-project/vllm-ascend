# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Device exact reads for ordinary dense, parallel eager proposals.

Capacity controls allocation only. Gather only valid KV tokens, repeating the
last valid token into masked scratch padding: masking uninitialized cache tail
directly is unsafe because zero attention weights can still multiply NaNs.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_read import DirectPrefixSource

DENSE_KV_ALIGNMENT = 32


@dataclass(frozen=True)
class DenseDraftRead:
    slots: torch.Tensor
    seq_lens: torch.Tensor
    starts: torch.Tensor
    mask: torch.Tensor
    capacity: int
    query_width: int
    retain: Callable[..., None]
    may_have_empty_rows: bool = False
    query_indices: torch.Tensor | None = None
    output_indices: torch.Tensor | None = None
    _attention_workspaces: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )

    @property
    def num_query_tokens(self) -> int:
        return self.output_indices.numel() if self.output_indices is not None else self.num_reqs * self.query_width

    @property
    def num_reqs(self) -> int:
        return self.slots.shape[0]

    def apply(self, common: Any) -> None:
        _apply_read_metadata(self, common)

    def capacity_cpu(self, common: Any) -> torch.Tensor:
        if getattr(common, "_draft_window_device_read", None) is not self:
            raise ValueError("Dense Draft read belongs to another common metadata object")
        return torch.full((common.num_reqs,), self.capacity, dtype=torch.int32, device="cpu")


@dataclass(frozen=True)
class DenseDraftReadPart:
    request_indices: tuple[int, ...]
    query_indices: torch.Tensor
    read: DenseDraftRead


@dataclass(frozen=True)
class DenseDraftReadBatch:
    """Private reads partitioned by phase, queries and Full allocation geometry.

    An unfinished prompt can need Full KV while concurrent decodes need W.
    Separate native operands avoid padding every decode to that Full capacity
    or every decode query to the prompt's query width. This never partitions
    by device E, S, rejection, or any other value requiring a Host read.
    """

    parts: tuple[DenseDraftReadPart, ...]
    seq_lens: torch.Tensor
    starts: torch.Tensor
    capacity: int
    num_query_tokens: int
    retain: Callable[..., None]

    @property
    def num_reqs(self) -> int:
        return self.seq_lens.numel()

    def apply(self, common: Any) -> None:
        _apply_read_metadata(self, common)

    def capacity_cpu(self, common: Any) -> torch.Tensor:
        if getattr(common, "_draft_window_device_read", None) is not self:
            raise ValueError("Dense Draft read belongs to another common metadata object")
        return torch.full((common.num_reqs,), self.capacity, dtype=torch.int32, device="cpu")


def _apply_read_metadata(read: DenseDraftRead | DenseDraftReadBatch, common: Any) -> None:
    n = common.num_reqs
    lengths = read.seq_lens.new_empty(n)
    read.retain(lengths)
    lengths.zero_()
    lengths[: read.num_reqs] = read.seq_lens
    common.seq_lens = lengths
    # These are explicitly capacity/workspace bounds, never CPU exact.
    common.seq_lens_cpu_upper_bound = torch.full((n,), read.capacity, dtype=torch.int32, device="cpu")
    common.seq_lens_cpu = None
    common._seq_lens_cpu = None
    common.max_seq_len = read.capacity
    common._draft_window_device_read = read


def _copy_owned_read_tensor(
    source: torch.Tensor, retain: Callable[..., None], *, device: torch.device | None = None
) -> torch.Tensor:
    """Own both operands before a copy can partially submit and raise."""
    destination = torch.empty_like(source, device=source.device if device is None else device)
    retain(source, destination)
    destination.copy_(source, non_blocking=True)
    return destination


def _prepare_partitioned_reads(
    source: DirectPrefixSource,
    *,
    groups: dict[tuple[bool, int, int], list[int]],
    exact_device: torch.Tensor,
    row_indices: torch.Tensor,
    upper_cpu: torch.Tensor,
    window_size: int,
    anchor_span: int,
    query_width: int,
    causal: bool,
    retain: Callable[..., None],
    frozen_starts: torch.Tensor | None,
    query_ends: torch.Tensor | None,
    query_start_loc_device: torch.Tensor | None,
) -> DenseDraftReadBatch:
    parts = []
    visible = torch.empty_like(exact_device[: len(source.request_rows)])
    starts = torch.empty_like(visible)
    retain(visible, starts)
    for (full, width, _capacity), rows in groups.items():
        host_rows = torch.tensor(rows, dtype=torch.int64, device="cpu")
        selected = _copy_owned_read_tensor(host_rows, retain, device=exact_device.device)
        bounds = (
            selected * query_width
            if query_start_loc_device is None
            else query_start_loc_device.index_select(0, selected)
        )
        queries = (bounds[:, None] + torch.arange(width, device=selected.device)[None, :]).flatten()
        retain(bounds, queries)
        subset = DirectPrefixSource(
            source.table,
            tuple(source.request_rows[row] for row in rows),
            tuple(source.valid_columns[row] for row in rows),
            source.block_span,
        )
        read = prepare_dense_draft_read(
            subset,
            exact_device=exact_device.index_select(0, selected),
            row_indices=row_indices.index_select(0, selected),
            upper_cpu=upper_cpu.index_select(0, host_rows),
            window_size=window_size,
            anchor_span=anchor_span,
            query_width=width,
            causal=causal,
            retain=retain,
            frozen_starts=None if frozen_starts is None else frozen_starts.index_select(0, selected),
            query_ends=None if query_ends is None else query_ends.index_select(0, selected),
            full_rows_cpu=torch.full((len(rows),), full, dtype=torch.bool, device="cpu"),
        )
        assert isinstance(read, DenseDraftRead)
        visible.index_copy_(0, selected, read.seq_lens)
        starts.index_copy_(0, selected, read.starts)
        parts.append(DenseDraftReadPart(tuple(rows), queries, read))
    result = DenseDraftReadBatch(
        tuple(parts),
        visible,
        starts,
        max(part.read.capacity for part in parts),
        sum(part.query_indices.numel() for part in parts),
        retain,
    )
    retain(result)
    return result


def prepare_dense_draft_read(
    source: DirectPrefixSource,
    *,
    exact_device: torch.Tensor,
    row_indices: torch.Tensor,
    upper_cpu: torch.Tensor,
    window_size: int,
    anchor_span: int,
    query_width: int,
    causal: bool,
    retain: Callable[..., None],
    frozen_starts: torch.Tensor | None = None,
    query_ends: torch.Tensor | None = None,
    full_rows_cpu: torch.Tensor | None = None,
    query_start_loc_cpu: torch.Tensor | None = None,
    query_start_loc_device: torch.Tensor | None = None,
) -> DenseDraftRead | DenseDraftReadBatch:
    """Snapshot current device E and canonical addresses without Host exact.

    Caller supplies current preparation's 0<E<=U and matching device request
    order. FullAttentionSpec residency retains all active pages. The Host
    upper must fit initialized columns, proving address capacity for every E
    in that preparation contract; it never chooses S or readable token count.
    An Eagle AR caller may supply the current proposal's validated frozen
    anchor. Its producer guarantees S<=E<=H; it must not follow later E-W.
    A first-pass Eagle caller may preserve its padded native query end while
    E remains the last valid KV end. Padding never authorizes a KV read.
    No device assertion is used: this torch_npu version implements even
    _assert_async through a synchronizing scalar copy.
    """
    n = len(source.request_rows)
    if (
        n < 1
        or window_size < 1
        or query_width < 1
        or anchor_span < 1
        or source.block_span < 1
        or anchor_span % source.block_span
    ):
        raise ValueError("Invalid dense Draft read geometry")
    if upper_cpu.device.type != "cpu" or upper_cpu.shape != (n,) or upper_cpu.dtype not in (torch.int32, torch.int64):
        raise ValueError("Missing current preparation allocation upper bounds")
    available = torch.tensor(source.valid_columns, dtype=torch.int64, device="cpu") * source.block_span
    if source.table.ndim != 2 or any(count < 1 or count > source.table.shape[1] for count in source.valid_columns):
        raise ValueError("Invalid initialized canonical column extent")
    if any(row < 0 or row >= source.table.shape[0] for row in source.request_rows):
        raise ValueError("Invalid canonical request row")
    if bool(((upper_cpu <= 0) | (upper_cpu > available)).any()):
        raise ValueError("Allocation upper exceeds initialized canonical columns")
    if exact_device.ndim != 1 or exact_device.numel() < n or row_indices.shape != (n,):
        raise ValueError("Missing current device lengths or request mapping")
    if exact_device.dtype not in (torch.int32, torch.int64) or row_indices.dtype != torch.int64:
        raise ValueError("Dense read lengths and row mapping must be integer tensors")
    if exact_device.device != source.table.device or row_indices.device != source.table.device:
        raise ValueError("Dense read operands must share their source device")
    exact = _copy_owned_read_tensor(exact_device[:n], retain)
    if query_ends is not None and (
        query_ends.shape != (n,)
        or query_ends.device != exact.device
        or query_ends.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Native query ends must match current request rows and device")
    query_ends = exact if query_ends is None else _copy_owned_read_tensor(query_ends, retain)
    if frozen_starts is not None and (
        frozen_starts.shape != (n,)
        or frozen_starts.device != exact.device
        or frozen_starts.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Frozen Draft anchors must match current request rows and device")
    full_rows = None
    if full_rows_cpu is not None:
        if full_rows_cpu.device.type != "cpu" or full_rows_cpu.dtype != torch.bool or full_rows_cpu.shape != (n,):
            raise ValueError("Full-read phase rows require one current CPU boolean per request")
        if full_rows_cpu.any():
            # This copies phase facts, never device exact lengths. Retain an
            # independent Host snapshot until the asynchronous H2D completes.
            phase_snapshot = full_rows_cpu.clone()
            full_rows = _copy_owned_read_tensor(phase_snapshot, retain, device=exact.device)
    query_offsets = query_indices = output_indices = None
    if (query_start_loc_cpu is None) != (query_start_loc_device is None):
        raise ValueError("Packed queries require current CPU and device boundaries")
    widths = None
    if query_start_loc_cpu is not None:
        assert query_start_loc_device is not None
        if (
            query_start_loc_cpu.device.type != "cpu"
            or query_start_loc_cpu.dtype not in (torch.int32, torch.int64)
            or query_start_loc_cpu.shape != (n + 1,)
            or query_start_loc_device.shape != (n + 1,)
            or query_start_loc_device.device != exact.device
            or query_start_loc_device.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Packed query boundaries must match the active preparation")
        widths = query_start_loc_cpu.diff()
        if int(query_start_loc_cpu[0]) != 0 or bool(((widths <= 0) | (widths > query_width)).any()):
            raise ValueError("Packed query widths exceed their current allocation bound")
    if full_rows_cpu is not None and full_rows_cpu.any():
        groups: dict[tuple[bool, int, int], list[int]] = {}
        for row in range(n):
            full = bool(full_rows_cpu[row])
            # U selects allocation geometry only. Padding every unfinished
            # prompt to the longest peer can multiply private Full KV memory.
            # Rows with equal native alignment still share one batched read;
            # the device E continues to determine every actual read/mask.
            full_capacity = (
                (int(upper_cpu[row]) + DENSE_KV_ALIGNMENT - 1) // DENSE_KV_ALIGNMENT * DENSE_KV_ALIGNMENT if full else 0
            )
            key = (full, query_width if widths is None else int(widths[row]), full_capacity)
            groups.setdefault(key, []).append(row)
        if len(groups) > 1:
            return _prepare_partitioned_reads(
                source,
                groups=groups,
                exact_device=exact,
                row_indices=row_indices,
                upper_cpu=upper_cpu,
                window_size=window_size,
                anchor_span=anchor_span,
                query_width=query_width,
                causal=causal,
                retain=retain,
                frozen_starts=frozen_starts,
                query_ends=query_ends,
                query_start_loc_device=query_start_loc_device,
            )
    if widths is not None:
        if not bool((widths == query_width).all()):
            query_offsets = _copy_owned_read_tensor(query_start_loc_device, retain)
            total = int(query_start_loc_cpu[-1])
            if exact.device.type == "npu":
                from vllm_ascend.ops.triton.v2.spec_decode.draft_window import pack_draft_query_indices

                query_indices, output_indices = pack_draft_query_indices(
                    query_offsets, total, query_width, retain=retain
                )
            else:
                local = torch.arange(query_width, device=exact.device)[None, :]
                row_widths = query_offsets.diff()
                padding = query_width - row_widths
                query_indices = query_offsets[:-1, None] + (local - padding[:, None]).clamp(min=0)
                output_indices = torch.cat(
                    [row * query_width + padding[row] + torch.arange(int(width)) for row, width in enumerate(widths)]
                )
            retain(query_offsets, query_indices, output_indices)
    # Both are allocation bounds: V <= E <= U and V <= W+B-1. Avoid
    # window-sized padding on short contexts without using U to choose S/V.
    capacity_bound = min(window_size + anchor_span - 1, int(upper_cpu.max()))
    if full_rows is not None:
        capacity_bound = max(capacity_bound, int(upper_cpu[full_rows_cpu].max()))
    capacity = (capacity_bound + DENSE_KV_ALIGNMENT - 1) // DENSE_KV_ALIGNMENT * DENSE_KV_ALIGNMENT
    if exact.device.type == "npu":
        from vllm_ascend.ops.triton.v2.spec_decode.draft_window import plan_draft_window

        slots, starts, visible, mask = plan_draft_window(
            source.table,
            row_indices,
            exact,
            frozen_starts,
            block_span=source.block_span,
            anchor_span=anchor_span,
            window=window_size,
            capacity=capacity,
            query_width=query_width,
            causal=causal,
            query_ends=query_ends,
            full_rows=full_rows,
            query_offsets=query_offsets,
            retain=retain,
        )
    else:
        # Keep a CPU execution path for geometry/ownership contract tests.
        starts = (
            torch.div(torch.clamp(exact - window_size, min=0), anchor_span, rounding_mode="floor") * anchor_span
            if frozen_starts is None
            else frozen_starts.clone()
        )
        visible = exact - starts
        if full_rows is not None:
            starts = torch.where(full_rows, 0, starts)
            visible = exact - starts
        columns = torch.arange(capacity, device=exact.device, dtype=torch.int64)
        absolute = starts[:, None] + torch.minimum(columns[None, :], visible[:, None] - 1)
        block_columns = torch.div(absolute, source.block_span, rounding_mode="floor")
        table_indices = row_indices[:, None] * source.table.shape[1] + block_columns
        blocks = source.table.reshape(-1).index_select(0, table_indices.reshape(-1)).reshape(n, capacity)
        slots = blocks.to(torch.int64) * source.block_span + absolute % source.block_span
        mask = (columns[None, None, :] >= visible[:, None, None]).expand(n, query_width, capacity)
        if causal:
            queries = (
                query_ends[:, None]
                - starts[:, None]
                - query_width
                + torch.arange(query_width, device=exact.device)[None, :]
            )
            mask = mask | (columns[None, None, :] > queries[:, :, None])
        if query_offsets is not None:
            padding = query_width - query_offsets.diff()
            mask = mask | (torch.arange(query_width)[None, :, None] < padding[:, None, None])
        mask = mask[:, None].contiguous()
    read = DenseDraftRead(
        slots,
        visible,
        starts,
        mask,
        capacity,
        query_width,
        retain,
        causal and window_size < query_width,
        query_indices,
        output_indices,
    )
    retain(read, exact)
    if query_ends is not exact:
        retain(query_ends)
    return read


def gather_dense_draft_kv(
    read: DenseDraftRead,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
    output_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather ND token rows, including interleaved/padded physical KV pages."""
    if key_cache.ndim != 4 or key_cache.shape != value_cache.shape:
        raise ValueError("Dense Draft KV requires matching four-dimensional caches")
    if key_cache.dtype not in (torch.float16, torch.bfloat16) or value_cache.dtype != key_cache.dtype:
        raise ValueError("Dense Draft KV requires unquantized float16/bfloat16 storage")
    if output_dtype not in (None, torch.float32):
        raise ValueError("Dense Draft gather conversion supports only FP32 native operands")
    if key_cache.device != value_cache.device or read.slots.device != key_cache.device:
        raise ValueError("Dense Draft slots and KV caches must share their device")
    shape = (read.num_reqs, read.capacity, *key_cache.shape[2:])
    layouts = []
    for cache in (key_cache, value_cache):
        blocks, block_size, heads, dim = cache.shape
        token_size = heads * dim
        page_stride = cache.stride(0)
        if (
            cache.stride()[1:] != (token_size, dim, 1)
            or page_stride < block_size * token_size
            or page_stride % token_size
        ):
            raise ValueError("Dense Draft KV requires contiguous ND token rows and aligned physical pages")
        stride_rows = page_stride // token_size
        layouts.append((cache, blocks, block_size, heads, dim, token_size, stride_rows))
    if output_dtype == torch.float32 and key_cache.device.type == "npu":
        import torch_npu

        from vllm_ascend.ops.triton.v2.spec_decode.draft_window import gather_draft_kv_fp32

        # Raw Triton addresses require an ND physical layout. Other formats
        # retain the existing format-aware PyTorch gather/conversion path.
        if all(torch_npu.get_npu_format(cache) in (0, 2) for cache in (key_cache, value_cache)):
            keys, values = gather_draft_kv_fp32(
                read.slots,
                key_cache,
                value_cache,
                retain=read.retain,
                **({"output_buffers": output_buffers} if output_buffers is not None else {}),
            )
            read.retain(keys, values)
            return keys, values
    if output_buffers is not None:
        raise ValueError("Reusable Draft operands require the native ND FP32 gather")
    slots = read.slots.reshape(-1)
    indices_by_stride = {}
    gathered = []
    for cache, blocks, block_size, heads, dim, token_size, stride_rows in layouts:
        # Preserve the original storage offset, including the V offset in a
        # combined K/V page. This is a view, never a full-cache flatten/copy.
        rows = cache.as_strided(((blocks - 1) * stride_rows + block_size, heads, dim), (token_size, dim, 1))
        if stride_rows not in indices_by_stride:
            indices_by_stride[stride_rows] = (
                slots
                if stride_rows == block_size
                else slots + torch.div(slots, block_size, rounding_mode="floor") * (stride_rows - block_size)
            )
        operand = rows.index_select(0, indices_by_stride[stride_rows]).reshape(shape)
        read.retain(operand)
        gathered.append(operand.to(output_dtype or cache.dtype))
    keys, values = gathered
    read.retain(keys, values)
    return keys, values


def dense_draft_attention(
    read: DenseDraftRead | DenseDraftReadBatch,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Consume the device exact mask without a Host exact-length consumer.

    Keep native inputs in FP32: fused BF16 attention lost small cancellation
    results in a controlled real-layer same-window comparison. The native FA
    consumer uses the device mask and padded initialized KV, without reading
    exact lengths on Host or explicitly expanding GQA heads. All private
    operands and returned workspaces remain owned through reader completion.
    """
    # Lazy import keeps leaf planning/ownership usable in CPU-only tests.
    import torch_npu

    if isinstance(read, DenseDraftReadBatch):
        output = query.new_empty((read.num_query_tokens, *query.shape[-2:]))
        read.retain(output)
        for part in read.parts:
            queries = query.index_select(0, part.query_indices)
            read.retain(queries)
            value = dense_draft_attention(part.read, queries, key_cache, value_cache, scale)
            output.index_copy_(0, part.query_indices, value)
        return output
    workspace_key = None
    if (
        key_cache.device.type == "npu"
        and value_cache.device == key_cache.device
        and key_cache.ndim == value_cache.ndim == 4
        and all(torch_npu.get_npu_format(cache) in (0, 2) for cache in (key_cache, value_cache))
    ):
        # Only this immediate native FA consumer reuses scratch. Its next
        # gather is queued after the preceding FA on the same NPU stream,
        # so the overwrite executes after that reader. A different stream,
        # KV geometry, read/group or proposal gets independent storage.
        # Standalone gather callers retain independent materializations.
        stream = torch.npu.current_stream(key_cache.device)
        workspace_key = (stream.npu_stream, read.num_reqs, read.capacity, *key_cache.shape[2:])
    buffers = read._attention_workspaces.get(workspace_key) if workspace_key is not None else None
    keys_fp32, values_fp32 = gather_dense_draft_kv(
        read,
        key_cache,
        value_cache,
        output_dtype=torch.float32,
        **({"output_buffers": buffers} if buffers is not None else {}),
    )
    if workspace_key is not None:
        read._attention_workspaces[workspace_key] = (keys_fp32, values_fp32)
    query = query[: read.num_query_tokens]
    if read.query_indices is not None:
        query = query.index_select(0, read.query_indices.flatten())
        read.retain(query)
    query = query.reshape(read.num_reqs, read.query_width, *query.shape[-2:])
    queries_fp32 = query.float().contiguous()
    read.retain(queries_fp32, keys_fp32, values_fp32)
    outputs = torch_npu.npu_fusion_attention(
        queries_fp32,
        keys_fp32,
        values_fp32,
        head_num=query.shape[-2],
        input_layout="BSND",
        atten_mask=read.mask,
        scale=scale,
        keep_prob=1.0,
        sparse_mode=0,
        inner_precise=0,
    )
    read.retain(*outputs)
    result = outputs[0]
    if read.may_have_empty_rows:
        empty_rows = read.mask.all(-1, keepdim=True).transpose(1, 2)
        read.retain(empty_rows)
        result.masked_fill_(empty_rows, 0)
    output = result.to(query.dtype).flatten(0, 1)
    if read.output_indices is not None:
        output = output.index_select(0, read.output_indices)
    read.retain(output)
    return output
