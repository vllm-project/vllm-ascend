# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Device exact reads for ordinary dense, parallel eager proposals.

Capacity controls allocation only. Gather only valid KV tokens, repeating the
last valid token into masked scratch padding: masking uninitialized cache tail
directly is unsafe because zero attention weights can still multiply NaNs.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_read import DirectPrefixSource


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

    @property
    def num_reqs(self) -> int:
        return self.slots.shape[0]

    def apply(self, common: Any) -> None:
        n = common.num_reqs
        lengths = self.seq_lens.new_zeros(n)
        lengths[: self.num_reqs] = self.seq_lens
        common.seq_lens = lengths
        # These are explicitly capacity/workspace bounds, never CPU exact.
        common.seq_lens_cpu_upper_bound = torch.full((n,), self.capacity, dtype=torch.int32, device="cpu")
        common.seq_lens_cpu = None
        common._seq_lens_cpu = None
        common.max_seq_len = self.capacity
        common._draft_window_device_read = self

    def capacity_cpu(self, common: Any) -> torch.Tensor:
        if getattr(common, "_draft_window_device_read", None) is not self:
            raise ValueError("Dense Draft read belongs to another common metadata object")
        return torch.full((common.num_reqs,), self.capacity, dtype=torch.int32, device="cpu")


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
) -> DenseDraftRead:
    """Snapshot current device E and canonical addresses without Host exact.

    Caller supplies current preparation's 0<E<=U and matching device request
    order. FullAttentionSpec residency retains all active pages. The Host
    upper must fit initialized columns, proving address capacity for every E
    in that preparation contract; it never chooses S or readable token count.
    An Eagle AR caller may supply the current proposal's validated frozen
    anchor. Its producer guarantees S<=E<=H; it must not follow later E-W.
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
    exact = exact_device[:n].clone()
    if frozen_starts is not None and (
        frozen_starts.shape != (n,)
        or frozen_starts.device != exact.device
        or frozen_starts.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Frozen Draft anchors must match current request rows and device")
    # Both are allocation bounds: V <= E <= U and V <= W+B-1. Avoid
    # window-sized padding on short contexts without using U to choose S/V.
    capacity_bound = min(window_size + anchor_span - 1, int(upper_cpu.max()))
    capacity = (capacity_bound + 31) // 32 * 32
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
        )
    else:
        # Keep a CPU execution path for geometry/ownership contract tests.
        starts = (
            torch.div(torch.clamp(exact - window_size, min=0), anchor_span, rounding_mode="floor") * anchor_span
            if frozen_starts is None
            else frozen_starts.clone()
        )
        visible = exact - starts
        columns = torch.arange(capacity, device=exact.device, dtype=torch.int64)
        absolute = starts[:, None] + torch.minimum(columns[None, :], visible[:, None] - 1)
        block_columns = torch.div(absolute, source.block_span, rounding_mode="floor")
        table_indices = row_indices[:, None] * source.table.shape[1] + block_columns
        blocks = source.table.reshape(-1).index_select(0, table_indices.reshape(-1)).reshape(n, capacity)
        slots = blocks.to(torch.int64) * source.block_span + absolute % source.block_span
        mask = (columns[None, None, :] >= visible[:, None, None]).expand(n, query_width, capacity)
        if causal:
            queries = visible[:, None] - query_width + torch.arange(query_width, device=exact.device)[None, :]
            mask = mask | (columns[None, None, :] > queries[:, :, None])
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
    )
    retain(read, exact)
    return read


def gather_dense_draft_kv(
    read: DenseDraftRead, key_cache: torch.Tensor, value_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather ND token rows, including interleaved/padded physical KV pages."""
    if key_cache.ndim != 4 or key_cache.shape != value_cache.shape:
        raise ValueError("Dense Draft KV requires matching four-dimensional caches")
    if key_cache.dtype not in (torch.float16, torch.bfloat16) or value_cache.dtype != key_cache.dtype:
        raise ValueError("Dense Draft KV requires unquantized float16/bfloat16 storage")
    shape = (read.num_reqs, read.capacity, *key_cache.shape[2:])
    slots = read.slots.reshape(-1)
    indices_by_stride = {}
    gathered = []
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
        # Preserve the original storage offset, including the V offset in a
        # combined K/V page. This is a view, never a full-cache flatten/copy.
        rows = cache.as_strided(((blocks - 1) * stride_rows + block_size, heads, dim), (token_size, dim, 1))
        if stride_rows not in indices_by_stride:
            indices_by_stride[stride_rows] = (
                slots
                if stride_rows == block_size
                else slots + torch.div(slots, block_size, rounding_mode="floor") * (stride_rows - block_size)
            )
        gathered.append(rows.index_select(0, indices_by_stride[stride_rows]).reshape(shape))
    keys, values = gathered
    read.retain(keys, values)
    return keys, values


def dense_draft_attention(
    read: DenseDraftRead, query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, scale: float
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

    keys, values = gather_dense_draft_kv(read, key_cache, value_cache)
    query = query[: read.num_reqs * read.query_width].reshape(read.num_reqs, read.query_width, *query.shape[-2:])
    queries_fp32 = query.float().contiguous()
    keys_fp32, values_fp32 = keys.float().contiguous(), values.float().contiguous()
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
    read.retain(output)
    return output
