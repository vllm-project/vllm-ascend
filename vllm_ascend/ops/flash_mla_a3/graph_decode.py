# SPDX-License-Identifier: Apache-2.0
"""Persistent inputs and scratch space for captured split-cache A3 FlashMLA."""

import math
import struct
from dataclasses import dataclass

import torch

from .flash_mla_a3 import _plain, _require
from .prepared_decode import catlass_ops


@dataclass(frozen=True)
class GraphDecodePlan:
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    tiling: torch.Tensor
    workspace: torch.Tensor
    query_length: int
    num_query_heads: int
    page_count: int
    cache_page_strides: tuple[int, int]
    splits: int


def graph_decode_ops_available() -> bool:
    try:
        ops = catlass_ops()
    except (OSError, RuntimeError):
        return False
    return hasattr(ops, "catlass_graph_out") and hasattr(ops, "catlass_graph_workspace_bytes")


def _split_count(max_kv, batches, cores):
    # Even a one-page replay uses the split epilogue for NTD and zero padding.
    return max(2, min((max_kv + 127) // 128, 64, cores // math.gcd(cores, batches)))


def prepare_decode_graph(
    block_table,
    cache_seqlens,
    query_length,
    page_count,
    *,
    num_query_heads,
    cache_page_strides,
    softmax_scale=1.0,
    causal=True,
    max_kv=None,
    workspace=None,
) -> GraphDecodePlan:
    """Prepare before capture; update the original page table and lengths in place before replay.

    Each physical request occupies one or four consecutive query tokens. Use
    zero KV lengths for padded requests. Replay loads BF16, 128-token KV pages
    from the original component-major cache; its physical page map and lengths
    must be valid. A shared scratch allocation requires serialized launches.
    """
    _require(
        isinstance(block_table, torch.Tensor)
        and block_table.device.type == "npu"
        and block_table.ndim == 2
        and block_table.dtype == torch.int32
        and block_table.stride(1) == 1,
        "expected an NPU int32 page table with contiguous rows",
    )
    batches, width = block_table.shape
    row_stride = block_table.stride(0)
    _require(
        0 < batches < 65536
        and 0 < width <= row_stride
        and 128 * row_stride <= 2**31 - 1
        and 0 < page_count <= 2**31 - 1,
        "invalid graph page table dimensions",
    )
    _require(
        isinstance(cache_seqlens, torch.Tensor)
        and cache_seqlens.device == block_table.device
        and cache_seqlens.shape == (batches,)
        and cache_seqlens.dtype == torch.int32
        and cache_seqlens.is_contiguous(),
        "expected contiguous int32 graph lengths on the page-table device",
    )
    _require(query_length in (1, 4) and num_query_heads in (4, 8), "expected Q1/Q4 with four or eight query heads")
    cache_page_strides = tuple(cache_page_strides)
    _require(
        len(cache_page_strides) == 2
        and all(
            type(stride) is int and 128 * channels <= stride <= 2**31 - 1 and stride % 16 == 0
            for stride, channels in zip(cache_page_strides, (512, 64))
        ),
        "invalid split graph cache page strides",
    )
    max_kv = width * 128 if max_kv is None else min(max_kv, width * 128)
    _require(type(max_kv) is int and max_kv > 0, "invalid graph KV limit")
    scale = float(softmax_scale)
    _require(math.isfinite(scale) and abs(scale) <= torch.finfo(torch.float32).max, "invalid graph softmax scale")

    ops = catlass_ops()
    splits = _split_count(max_kv, batches, ops.catlass_cores(block_table))
    span = ((max_kv + 127) // 128 + splits - 1) // splits * 128
    heads = num_query_heads * query_length
    scale_bits = struct.unpack("=i", struct.pack("=f", scale))[0]
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
        3 if causal else 0,
        64,
        max_kv,
        span,
        splits,
        1,
        batches,
        0,
        0,
        0,
        batches * query_length,
        0,
        *cache_page_strides,
    ]
    records = [header + [0] * (32 - len(header))]
    for batch in range(batches):
        output_offset = batch * heads * 512
        fields = [
            1,
            0,
            0,
            128,
            0,
            output_offset,
            0,
            batch * heads * 64,
            0,
            0,
            0,
            0,
            batch * heads * splits,
            0,
            output_offset,
        ]
        records.append(fields + [0] * (32 - len(fields)))
    tiling = torch.tensor(records, dtype=torch.int32, device=block_table.device)
    size = ops.catlass_graph_workspace_bytes(block_table, batches, heads, splits)
    if workspace is None:
        workspace = torch.empty(size, dtype=torch.uint8, device=block_table.device)
    else:
        _require(
            workspace.device == block_table.device
            and workspace.dtype == torch.uint8
            and workspace.is_contiguous()
            and workspace.numel() >= size,
            "insufficient graph workspace",
        )
    return GraphDecodePlan(
        block_table,
        cache_seqlens,
        tiling,
        workspace,
        query_length,
        num_query_heads,
        page_count,
        cache_page_strides,
        splits,
    )


def flash_mla_decode_graph(query, cache, plan):
    """Record or launch split-cache attention with a persistent graph plan; output NTD."""
    import torch_npu

    batches = plan.block_table.size(0)
    tokens = batches * plan.query_length
    heads = plan.num_query_heads
    query_parts, cache_parts = [], []
    _require(len(query) == len(cache) == 2, "expected latent and RoPE pairs")
    for q, k, channels, page_stride in zip(query, cache, (512, 64), plan.cache_page_strides):
        _require(
            q.device == k.device == plan.block_table.device
            and q.dtype == k.dtype == torch.bfloat16
            and q.shape == (tokens, heads, channels),
            "invalid graph query or cache dtype, shape or device",
        )
        if k.ndim == 4:
            k = k.squeeze(1 if k.size(1) == 1 else 2)
        _require(
            k.shape == (plan.page_count, 128, channels)
            and k.stride() == (page_stride, channels, 1)
            and k.storage_offset() % 16 == 0
            and torch_npu.get_npu_format(k) in (0, 2),
            "invalid graph split cache view",
        )
        query_parts.append(_plain(q).view(batches, plan.query_length * heads, channels))
        cache_parts.append(k)
    out = query[0].new_empty((heads, tokens, 512))
    catlass_ops().catlass_graph_out(
        *query_parts,
        *cache_parts,
        plan.block_table,
        plan.cache_seqlens,
        plan.tiling,
        plan.workspace,
        out,
        plan.splits,
        plan.query_length * heads,
    )
    return out
