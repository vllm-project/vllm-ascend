# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Prepared exact reads for ordinary dense, parallel eager proposals.

Capacity controls allocation only. Gather only valid KV tokens, repeating the
last valid token into masked scratch padding: masking uninitialized cache tail
directly is unsafe because zero attention weights can still multiply NaNs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

DENSE_KV_ALIGNMENT = 32


@dataclass(frozen=True)
class DenseDraftRead:
    slots: torch.Tensor
    seq_lens: torch.Tensor  # Completed CPU visible lengths for the normal builder.
    mask: torch.Tensor
    query_width: int
    retain: Callable[..., None]
    may_have_empty_rows: bool = False
    _attention_workspaces: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )

    @property
    def num_query_tokens(self) -> int:
        return self.num_reqs * self.query_width

    @property
    def num_reqs(self) -> int:
        return self.slots.shape[0]


@dataclass(frozen=True)
class DenseDraftReadBatch:
    """Private reads partitioned by phase, queries and Full allocation geometry.

    An unfinished prompt can need Full KV while concurrent decodes need W.
    Separate native operands avoid padding every decode to that Full capacity
    or every decode query to the prompt's query width. This never partitions
    by device E, S, rejection, or any other value requiring a Host read.
    """

    parts: tuple[tuple[torch.Tensor, DenseDraftRead], ...]
    seq_lens: torch.Tensor  # Completed CPU visible lengths for the normal builder.
    num_query_tokens: int
    retain: Callable[..., None]


def _copy_owned_read_tensor(source: torch.Tensor, retain: Callable[..., None], *, device: torch.device) -> torch.Tensor:
    """Own both operands before a copy can partially submit and raise."""
    destination = torch.empty_like(source, device=device)
    retain(source, destination)
    destination.copy_(source, non_blocking=True)
    return destination


def prepare_dense_draft_read(
    block_table: torch.Tensor,
    valid_columns: Sequence[int],
    block_span: int,
    *,
    exact_cpu: torch.Tensor,
    query_ends_cpu: torch.Tensor,
    upper_cpu: torch.Tensor,
    query_width: int,
    causal: bool,
    retain: Callable[..., None],
    frozen_starts: torch.Tensor,
    full_rows_cpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor | None = None,
    query_start_loc_device: torch.Tensor | None = None,
) -> DenseDraftRead | DenseDraftReadBatch:
    """Use completed current exact lengths, not optimistic scheduling bounds.

    The sole Runtime caller supplies typed CPU snapshots and frozen starts;
    BlockTables and the normal builder supply the compact int32 read table.
    Exact lengths are shared across groups from that same preparation. Read
    slots lie in [S,E); capacity padding repeats its last initialized token. Query
    padding and Full continuing rows retain the method's existing semantics.
    """
    n = len(valid_columns)
    if block_table.shape[0] < n or any(count < 0 or count > block_table.shape[1] for count in valid_columns):
        raise ValueError("Invalid initialized compact column extent")
    if n < 1 or query_width < 1:
        raise ValueError("Invalid dense Draft read geometry")
    if any(t.numel() != n for t in (exact_cpu, query_ends_cpu, upper_cpu, frozen_starts, full_rows_cpu)):
        raise ValueError("Draft read inputs must cover the current active requests")
    available = torch.tensor(valid_columns, dtype=torch.int64, device="cpu") * block_span
    retain(block_table)
    # The runtime supplies its one current exact snapshot and frozen starts;
    # this reader never derives lengths or synchronizes device values again.
    if bool(((exact_cpu <= 0) | (exact_cpu > upper_cpu) | (exact_cpu > available)).any()):
        raise ValueError("Dense Draft exact lengths exceed initialized canonical columns or preparation bounds")
    starts = frozen_starts
    if bool(((starts < 0) | (starts >= exact_cpu)).any()):
        raise ValueError("Frozen Draft anchors must match current request rows")
    visible_lengths = exact_cpu - starts
    widths = None
    if query_start_loc_cpu is not None:
        assert query_start_loc_device is not None
        if query_start_loc_cpu.numel() != n + 1 or query_start_loc_device.numel() != n + 1:
            raise ValueError("Packed query boundaries must match the active preparation")
        widths = query_start_loc_cpu.diff()
        if int(query_start_loc_cpu[0]) != 0 or bool(((widths <= 0) | (widths > query_width)).any()):
            raise ValueError("Packed query widths exceed their current allocation bound")

    def materialize(host_rows: torch.Tensor, width: int, device_rows: torch.Tensor | None) -> DenseDraftRead:
        origins = starts.index_select(0, host_rows)
        visible = visible_lengths.index_select(0, host_rows)
        capacity_bound = int(visible.max())
        capacity = (capacity_bound + DENSE_KV_ALIGNMENT - 1) // DENSE_KV_ALIGNMENT * DENSE_KV_ALIGNMENT
        columns = torch.arange(capacity, device="cpu", dtype=torch.int64)
        absolute = origins[:, None] + torch.minimum(columns, visible[:, None] - 1)
        address = _copy_owned_read_tensor(absolute, retain, device=block_table.device)
        logical_columns = torch.div(address, block_span, rounding_mode="floor")
        if host_rows.numel() == n:
            blocks = block_table[:n].gather(1, logical_columns)
        else:
            assert device_rows is not None
            if block_table.is_contiguous():
                indices = device_rows[:, None] * block_table.shape[1] + logical_columns
                blocks = block_table.view(-1).index_select(0, indices.flatten()).reshape_as(address)
            else:
                blocks = block_table[device_rows[:, None], logical_columns]
        slots = blocks.to(torch.int64) * block_span + address % block_span
        excluded = (columns[None, None, :] >= visible[:, None, None]).expand(-1, width, -1)
        may_have_empty_rows = False
        if causal:
            query_end = query_ends_cpu.index_select(0, host_rows)
            queries = query_end[:, None] - origins[:, None] - width + torch.arange(width, device="cpu")
            excluded = excluded | (columns[None, None, :] > queries[:, :, None])
            may_have_empty_rows = bool((queries < 0).any())
        mask = _copy_owned_read_tensor(excluded[:, None].contiguous(), retain, device=block_table.device)
        read = DenseDraftRead(slots, visible, mask, width, retain, may_have_empty_rows)
        retain(read)
        return read

    if widths is None and not bool(full_rows_cpu.any()):
        return materialize(torch.arange(n, device="cpu"), query_width, None)
    groups: dict[tuple[bool, int, int], list[int]] = {}
    for row in range(n):
        full = bool(full_rows_cpu[row])
        capacity = (int(upper_cpu[row]) + DENSE_KV_ALIGNMENT - 1) // DENSE_KV_ALIGNMENT if full else 0
        key = (full, query_width if widths is None else int(widths[row]), capacity)
        groups.setdefault(key, []).append(row)
    if len(groups) == 1:
        _, width, _ = next(iter(groups))
        return materialize(torch.arange(n, device="cpu"), width, None)
    parts = []
    for (_full, width, _capacity), group in groups.items():
        host_rows = torch.tensor(group, dtype=torch.int64, device="cpu")
        selected = _copy_owned_read_tensor(host_rows, retain, device=block_table.device)
        bounds = (
            selected * query_width
            if query_start_loc_device is None
            else query_start_loc_device.index_select(0, selected)
        )
        queries = (bounds[:, None] + torch.arange(width, device=selected.device)[None, :]).flatten()
        retain(bounds, queries)
        read = materialize(host_rows, width, selected)
        parts.append((queries, read))
    result = DenseDraftReadBatch(tuple(parts), visible_lengths, sum(q.numel() for q, _ in parts), retain)
    retain(result)
    return result


def gather_dense_draft_kv(
    read: DenseDraftRead,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare the immediate FA reader, preserving physical KV page strides.

    The only production caller queues attention before the next gather. Its
    per-read ND geometry scratch can therefore be reused in stream order.
    """
    if key_cache.ndim != 4 or key_cache.shape != value_cache.shape:
        raise ValueError("Dense Draft KV requires matching four-dimensional caches")
    if key_cache.dtype not in (torch.float16, torch.bfloat16) or value_cache.dtype != key_cache.dtype:
        raise ValueError("Dense Draft KV requires unquantized float16/bfloat16 storage")
    if key_cache.device != value_cache.device or read.slots.device != key_cache.device:
        raise ValueError("Dense Draft slots and KV caches must share their device")
    blocks, block_size, heads, dim = key_cache.shape
    token_size = heads * dim
    shape = (*read.slots.shape, heads, dim)
    layouts = []
    for cache in (key_cache, value_cache):
        page_stride = cache.stride(0)
        if (
            cache.stride()[1:] != (token_size, dim, 1)
            or page_stride < block_size * token_size
            or page_stride % token_size
        ):
            raise ValueError("Dense Draft KV requires contiguous ND token rows and aligned physical pages")
        stride_rows = page_stride // token_size
        layouts.append((cache, stride_rows))
    if key_cache.device.type == "npu":
        import torch_npu
        from vllm.triton_utils import triton

        from vllm_ascend.ops.triton.spec_decode.draft_window import GATHER_TOKEN_BLOCK_SIZE, _gather_draft_kv_fp32

        # Raw Triton addresses require an ND physical layout. Other formats
        # retain the existing format-aware PyTorch gather/conversion path.
        if all(torch_npu.get_npu_format(cache) in (0, 2) for cache in (key_cache, value_cache)):
            workspace_key = key_cache.shape[2:]
            if not read.slots.is_contiguous():
                raise ValueError("Fused Draft gather requires contiguous private slots")
            buffers = read._attention_workspaces.get(workspace_key)
            if buffers is None:
                keys = torch.empty(shape, dtype=torch.float32, device=key_cache.device)
                buffers = keys, torch.empty_like(keys)
                read._attention_workspaces[workspace_key] = buffers
            keys, values = buffers
            read.retain(read.slots, key_cache, value_cache, keys, values)
            _gather_draft_kv_fp32[(triton.cdiv(read.slots.numel(), GATHER_TOKEN_BLOCK_SIZE),)](
                read.slots,
                key_cache,
                value_cache,
                keys,
                values,
                key_cache.stride(0),
                value_cache.stride(0),
                key_cache.shape[1],
                token_size,
                -read.slots.numel() % GATHER_TOKEN_BLOCK_SIZE,
                GATHER_TOKEN_BLOCK_SIZE,
                triton.next_power_of_2(token_size),
            )
            return keys, values
    slots = read.slots.reshape(-1)
    indices_by_stride = {}
    gathered = []
    for cache, stride_rows in layouts:
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
        gathered.append(operand.float())
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
    """Consume the completed exact read projection with native FP32 attention.

    Keep native inputs in FP32: fused BF16 attention lost small cancellation
    results in a controlled real-layer same-window comparison. The native FA
    consumer uses the device mask and padded initialized KV without explicit
    GQA expansion. All private operands and returned workspaces remain owned
    through reader completion.
    """
    # Lazy import keeps leaf planning/ownership usable in CPU-only tests.
    import torch_npu

    if isinstance(read, DenseDraftReadBatch):
        output = query.new_empty((read.num_query_tokens, *query.shape[-2:]))
        read.retain(output)
        for indices, part in read.parts:
            queries = query.index_select(0, indices)
            read.retain(queries)
            value = dense_draft_attention(part, queries, key_cache, value_cache, scale)
            output.index_copy_(0, indices, value)
        return output
    keys_fp32, values_fp32 = gather_dense_draft_kv(read, key_cache, value_cache)
    query = query[: read.num_query_tokens]
    query = query.reshape(read.num_reqs, read.query_width, *query.shape[-2:])
    queries_fp32 = query.float().contiguous()
    read.retain(queries_fp32)
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
    read.retain(output)
    return output
