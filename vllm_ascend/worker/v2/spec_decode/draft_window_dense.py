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
) -> DenseDraftRead:
    """Snapshot current device E and canonical addresses without Host exact.

    Caller supplies current preparation's 0<E<=U and matching device request
    order. FullAttentionSpec residency retains all active pages. The Host
    upper must fit initialized columns, proving address capacity for every E
    in that preparation contract; it never chooses S or readable token count.
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
    starts = torch.div(torch.clamp(exact - window_size, min=0), anchor_span, rounding_mode="floor") * anchor_span
    visible = exact - starts
    # Align dense FIA mask rows; this allocation rounding never changes S/V.
    capacity = (window_size + anchor_span - 1 + 31) // 32 * 32
    columns = torch.arange(capacity, device=exact.device, dtype=torch.int64)
    # Every gather address denotes an actual token in [S,E), including padding.
    absolute = starts[:, None] + torch.minimum(columns[None, :], visible[:, None] - 1)
    block_columns = torch.div(absolute, source.block_span, rounding_mode="floor")
    # Flat gathers avoid the advanced-index AICPU fallback on current CANN.
    table_indices = row_indices[:, None] * source.table.shape[1] + block_columns
    blocks = source.table.reshape(-1).index_select(0, table_indices.reshape(-1)).reshape(n, capacity)
    slots = blocks.to(torch.int64) * source.block_span + absolute % source.block_span
    mask = (columns[None, None, :] >= visible[:, None, None]).expand(n, query_width, capacity)
    if causal:
        queries = visible[:, None] - query_width + torch.arange(query_width, device=exact.device)[None, :]
        mask = mask | (columns[None, None, :] > queries[:, :, None])
    read = DenseDraftRead(slots, visible, starts, mask[:, None].contiguous(), capacity, query_width, retain)
    retain(read, exact)
    return read


def gather_dense_draft_kv(
    read: DenseDraftRead, key_cache: torch.Tensor, value_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read ND [blocks,B,KV-heads,D] caches into independently owned scratch."""
    if key_cache.ndim != 4 or key_cache.shape != value_cache.shape:
        raise ValueError("Dense Draft KV requires matching four-dimensional caches")
    if not key_cache.is_contiguous() or not value_cache.is_contiguous():
        raise ValueError("Dense Draft KV requires the ordinary contiguous cache layout")
    if key_cache.dtype not in (torch.float16, torch.bfloat16) or value_cache.dtype != key_cache.dtype:
        raise ValueError("Dense Draft KV requires unquantized float16/bfloat16 storage")
    shape = (read.num_reqs, read.capacity, *key_cache.shape[2:])
    indices = read.slots.reshape(-1)
    keys = key_cache.flatten(0, 1).index_select(0, indices).reshape(shape)
    values = value_cache.flatten(0, 1).index_select(0, indices).reshape(shape)
    read.retain(keys, values)
    return keys, values


def dense_draft_attention(
    read: DenseDraftRead, query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, scale: float
) -> torch.Tensor:
    """Consume device exact mask and valid scratch, with Host capacity only."""
    import torch_npu

    keys, values = gather_dense_draft_kv(read, key_cache, value_cache)
    query = query[: read.num_reqs * read.query_width].reshape(read.num_reqs, read.query_width, *query.shape[-2:])
    result = torch_npu.npu_fused_infer_attention_score(
        query,
        keys,
        values,
        input_layout="BSND",
        num_heads=query.shape[-2],
        num_key_value_heads=keys.shape[-2],
        scale=scale,
        sparse_mode=0,
        actual_seq_lengths=[read.query_width] * read.num_reqs,
        actual_seq_lengths_kv=[read.capacity] * read.num_reqs,
        atten_mask=read.mask,
    )[0]
    return result.flatten(0, 1)
