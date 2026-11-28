# SPDX-License-Identifier: Apache-2.0
"""Split-cache A3 decode with one reusable device page schedule per step."""

import math
import struct
from dataclasses import dataclass

import torch

from .flash_mla_a3 import _plain, _require
from .page_schedule import schedule_pages
from .prepared_decode import catlass_ops


@dataclass(frozen=True)
class BalancedPageSchedule:
    lengths: torch.Tensor
    records: torch.Tensor
    page_table_width: int
    cores: int
    query_length: int
    causal: bool


@dataclass(frozen=True)
class BalancedDecodePlan:
    schedule: BalancedPageSchedule
    block_table: torch.Tensor
    tiling: torch.Tensor
    workspace: torch.Tensor
    num_query_heads: int
    page_count: int
    cache_page_strides: tuple[int, int]


def balanced_decode_ops_available() -> bool:
    try:
        ops = catlass_ops()
    except (OSError, RuntimeError):
        return False
    return hasattr(ops, "catlass_balanced_graph_out") and hasattr(ops, "catlass_balanced_graph_workspace_bytes")


def prepare_balanced_schedule(
    cache_seqlens, page_table_width, *, query_length=1, causal=True, cores=None
) -> BalancedPageSchedule:
    """Allocate stable page-schedule storage before any capture or layer launch.

    ``cache_seqlens`` is the persistent int32 device input, including newly
    appended queries. Padding uses zero lengths. ``page_table_width`` counts
    logical 128-token pages and bounds all live KV histories in this schedule.
    This function does not run the device scheduler.
    """
    _require(
        isinstance(cache_seqlens, torch.Tensor)
        and cache_seqlens.device.type == "npu"
        and cache_seqlens.dtype == torch.int32
        and cache_seqlens.ndim == 1
        and cache_seqlens.is_contiguous(),
        "expected contiguous int32 device lengths",
    )
    batches = cache_seqlens.numel()
    _require(
        0 < batches <= 256
        and type(page_table_width) is int
        and 0 < page_table_width <= (2**31 - 1) // max(batches, 128),
        "invalid page schedule geometry",
    )
    if cores is None:
        cores = catlass_ops().catlass_cores(cache_seqlens)
    _require(
        type(cores) is int and 0 < cores <= 64 and type(query_length) is int and query_length in (1, 4),
        "expected Q1/Q4 and a valid Cube core count",
    )
    records = cache_seqlens.new_zeros((batches + 1, 4))
    return BalancedPageSchedule(cache_seqlens, records, page_table_width, cores, query_length, bool(causal))


def update_balanced_schedule(schedule: BalancedPageSchedule) -> torch.Tensor:
    """Run once before all layers on their stream, including inside capture.

    Update ``schedule.lengths`` in place before each eager step or replay. A
    graph records this call once before its layer ops, so replay recomputes the
    shared schedule without a host read or another Python call. Multiple plans
    can reference the same schedule and use different layer cache geometry.
    """
    return schedule_pages(
        schedule.lengths,
        schedule.page_table_width,
        schedule.cores,
        query_width=schedule.query_length,
        causal=schedule.causal,
        out=schedule.records,
        direct_output=True,
    )


def prepare_balanced_decode(
    block_table,
    schedule: BalancedPageSchedule,
    page_count,
    *,
    num_query_heads,
    cache_page_strides,
    softmax_scale=1.0,
    workspace=None,
) -> BalancedDecodePlan:
    """Prepare one layer group without reading lengths or copying page maps.

    The page table is persistent and can use a physical row stride larger than
    its logical width. All live physical page IDs must be valid. Workspace may
    be reused across plans whose launches are serialized on the same stream.
    """
    lengths = schedule.lengths
    batches = lengths.numel()
    _require(
        isinstance(block_table, torch.Tensor)
        and block_table.device == lengths.device
        and block_table.dtype == torch.int32
        and block_table.ndim == 2
        and block_table.size(0) == batches
        and block_table.size(1) >= schedule.page_table_width
        and block_table.stride(1) == 1,
        "expected int32 page table matching the schedule geometry and device",
    )
    row_stride = block_table.stride(0)
    _require(
        block_table.size(1) <= row_stride <= (2**31 - 1) // batches
        and type(page_count) is int
        and 0 < page_count <= 2**31 - 1,
        "invalid page count or physical table stride",
    )
    _require(type(num_query_heads) is int and num_query_heads in (4, 8), "expected four or eight query heads")
    page_strides = tuple(cache_page_strides)
    _require(
        len(page_strides) == 2
        and all(
            type(stride) is int and 128 * channels <= stride <= 2**31 - 1 and stride % 16 == 0
            for stride, channels in zip(page_strides, (512, 64))
        ),
        "invalid split cache page strides",
    )
    scale = float(softmax_scale)
    _require(math.isfinite(scale) and abs(scale) <= torch.finfo(torch.float32).max, "invalid softmax scale")

    ops = catlass_ops()
    _require(
        hasattr(ops, "catlass_balanced_graph_out") and hasattr(ops, "catlass_balanced_graph_workspace_bytes"),
        "the A3 CATLASS adapter lacks balanced graph decode",
    )
    _require(ops.catlass_cores(block_table) == schedule.cores, "page schedule core count differs from the device")
    tokens = batches * schedule.query_length
    heads = num_query_heads * schedule.query_length
    scale_bits = struct.unpack("=i", struct.pack("=f", scale))[0]

    # Keep query offsets, physical strides and NTD geometry in the static
    # graph header. Device schedule records supply all split ranges and scratch
    # offsets. The static span and per-record scratch offsets are unused.
    header = [
        batches,
        heads,
        512,
        page_count,
        128,
        row_stride,
        scale_bits,
        1,
        32,
        32,
        heads,
        1,
        3 if schedule.causal else 0,
        64,
        schedule.page_table_width * 128,
        0,
        schedule.cores,
        1,
        batches,
        0,
        0,
        0,
        tokens,
        0,
        *page_strides,
    ]
    records = [header + [0] * (32 - len(header))]
    for batch in range(batches):
        q_offset = batch * heads * 512
        rope_offset = batch * heads * 64
        record = [1, 0, 0, 128, 0, q_offset, 0, rope_offset]
        records.append(record + [0] * (32 - len(record)))
    tiling = torch.tensor(records, dtype=torch.int32, device=block_table.device)

    size = ops.catlass_balanced_graph_workspace_bytes(block_table, batches, heads)
    if workspace is None:
        workspace = torch.empty(size, dtype=torch.uint8, device=block_table.device)
    else:
        _require(
            workspace.device == block_table.device
            and workspace.dtype == torch.uint8
            and workspace.is_contiguous()
            and workspace.storage_offset() % 32 == 0
            and workspace.numel() >= size,
            "invalid or insufficient balanced decode workspace",
        )
    return BalancedDecodePlan(schedule, block_table, tiling, workspace, num_query_heads, page_count, page_strides)


def flash_mla_decode_balanced(query, cache, plan: BalancedDecodePlan, *, out=None):
    """Launch one layer against an already updated schedule and return NTD output."""
    import torch_npu

    schedule = plan.schedule
    batches = schedule.lengths.numel()
    heads = plan.num_query_heads
    tokens = batches * schedule.query_length
    device = plan.block_table.device
    query_parts, cache_parts = [], []
    _require(len(query) == len(cache) == 2, "expected latent and RoPE pairs")
    for q, k, channels, page_stride in zip(query, cache, (512, 64), plan.cache_page_strides):
        _require(
            q.device == k.device == device
            and q.dtype == k.dtype == torch.bfloat16
            and q.shape == (tokens, heads, channels),
            "invalid balanced query or cache shape, dtype or device",
        )
        if k.ndim == 4:
            k = k.squeeze(1 if k.size(1) == 1 else 2)
        _require(
            k.shape == (plan.page_count, 128, channels)
            and k.stride() == (page_stride, channels, 1)
            and k.storage_offset() % 16 == 0
            and torch_npu.get_npu_format(k) in (0, 2),
            "invalid split cache view",
        )
        query_parts.append(_plain(q).view(batches, schedule.query_length * heads, channels))
        cache_parts.append(k)
    if out is None:
        out = query[0].new_empty((heads, tokens, 512))
    else:
        _require(
            out.shape == (heads, tokens, 512)
            and out.dtype == torch.bfloat16
            and out.device == device
            and out.is_contiguous(),
            "expected contiguous BF16 NTD output",
        )
    catlass_ops().catlass_balanced_graph_out(
        *query_parts,
        *cache_parts,
        plan.block_table,
        schedule.lengths,
        plan.tiling,
        schedule.records,
        plan.workspace,
        out,
        heads * schedule.query_length,
    )
    return out
