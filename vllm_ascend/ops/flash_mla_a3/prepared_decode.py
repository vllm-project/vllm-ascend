# SPDX-License-Identifier: Apache-2.0
"""Per-step MLA execution plan when the scheduler already has CPU sequence lengths."""

import math
import struct
from dataclasses import dataclass
from functools import lru_cache

import torch

from .flash_mla_a3 import _LIBRARY_DIR


@lru_cache(maxsize=1)
def catlass_ops():
    adapter = _LIBRARY_DIR / "libflash_mla_a3_catlass_ops.so"
    if not adapter.is_file():
        raise RuntimeError(f"Missing packaged FlashMLA CATLASS adapter {adapter}")
    torch.ops.load_library(str(adapter))
    return torch.ops.flash_mla_a3


@dataclass(frozen=True)
class DecodePlan:
    block_table: torch.Tensor
    page_count: int
    query_length: int
    softmax_scale: float
    causal: bool
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    mask: torch.Tensor | None
    metadata: torch.Tensor
    tiling: torch.Tensor | None
    splits: int
    native_ntd: bool
    max_kv: int
    split_cache: bool
    valid_rows: torch.Tensor | None
    cache_page_strides: tuple[int, int] | None = None
    num_query_heads: int = 4
    heads_per_launch: int = 4


def _split_pages(width, batches, cores):
    # Allocate equal-length chunks. A last partial chunk is valid. Matching
    # the chunk count to the core grid avoids an underfilled extra wave.
    target = min(width, 64, cores // math.gcd(cores, batches))
    pages_per_split = (width + target - 1) // target
    return pages_per_split * 128, (width + pages_per_split - 1) // pages_per_split


def _split_active_pages(page_counts, cores):
    width = max(page_counts)
    span, splits = _split_pages(width, len(page_counts), cores)
    balanced_pages = span // 128
    average_pages = (sum(page_counts) + cores - 1) // cores
    if min(page_counts) == width or average_pages >= balanced_pages:
        return span, splits

    active = [(batch, count) for batch, count in enumerate(page_counts) if count]

    def split_work(pages_per_split):
        return [
            (batch, split, min(pages_per_split, count - first) + 1)
            for batch, count in active
            for split, first in enumerate(range(0, count, pages_per_split))
        ]

    def peak_load(work, slots):
        load = [0] * cores
        for batch, split, weight in work:
            load[(batch * slots + split) % cores] += weight
        return max(load)

    # Unused split slots are skipped by the kernel. They can change where the
    # next request starts in the Cube grid without adding a live split.
    best = (peak_load(split_work(balanced_pages), splits), splits, balanced_pages)
    for pages in range(max(1, (width + 63) // 64, average_pages - 2), min(balanced_pages, average_pages + 2) + 1):
        work = split_work(pages)
        live_splits = (width + pages - 1) // pages
        for slots in range(live_splits, min(64, live_splits + cores - 1) + 1):
            best = min(best, (peak_load(work, slots), slots, pages))
    return best[2] * 128, best[1]


def prepare_decode(
    block_table,
    kv_lengths,
    query_length,
    page_count,
    *,
    causal=True,
    softmax_scale=1.0,
    attn_mask=None,
    layout_out="NTD",
    split_cache=False,
    validate_page_table=True,
    cache_page_strides: tuple[int, int] | None = None,
    num_query_heads: int = 4,
):
    """Prepare one scheduler step for all layers with BF16 KV pages of 128 tokens.

    kv_lengths are CPU integers, including the new tokens already in the cache.
    Query length is common across the batch. Rebuild if the page map or lengths
    change. The plan owns a snapshot of the live page-table columns. Set
    validate_page_table=False only if the scheduler has checked live page IDs.
    A split cache can specify latent and RoPE page strides in BF16 elements.
    The default uses densely packed pages for each cache component.
    Split caches support four or eight query heads. Fused caches use four.
    """
    from .flash_mla_a3 import _plain, _require, flash_mla_metadata

    _require(
        isinstance(block_table, torch.Tensor)
        and block_table.device.type == "npu"
        and block_table.dtype == torch.int32
        and block_table.ndim == 2,
        "expected NPU int32 block_table[B,M]",
    )
    _require(not isinstance(kv_lengths, torch.Tensor), "kv_lengths must contain CPU integers")
    lengths = list(kv_lengths)
    batches, capacity = block_table.shape
    _require(0 < batches < 65536 and capacity > 0 and 0 < page_count <= 2**31 - 1, "invalid page table dimensions")
    _require(
        len(lengths) == batches and all(isinstance(x, int) and 0 <= x <= 128 * capacity for x in lengths),
        "kv_lengths must contain one valid history length per batch",
    )
    _require(query_length in (1, 4), "expected one-token decode or four-token MTP")
    _require(
        num_query_heads == 4 or (split_cache and num_query_heads == 8),
        "expected four query heads or eight query heads with a split cache",
    )
    if cache_page_strides is not None:
        _require(
            split_cache and isinstance(cache_page_strides, (list, tuple)) and len(cache_page_strides) == 2,
            "cache_page_strides requires a split cache and two page strides",
        )
        cache_page_strides = tuple(cache_page_strides)
        _require(
            all(
                type(stride) is int and 128 * channels <= stride <= 2**31 - 1 and stride % 16 == 0
                for stride, channels in zip(cache_page_strides, (512, 64))
            ),
            "split cache page strides must span a dense page, be 16-element aligned and fit int32",
        )
    elif split_cache:
        cache_page_strides = (128 * 512, 128 * 64)
    scale = float(softmax_scale)
    _require(math.isfinite(scale) and abs(scale) <= torch.finfo(torch.float32).max, "invalid softmax scale")
    _require(layout_out in ("TND", "NTD"), "expected TND or NTD output")
    device = block_table.device
    maximum = max(lengths)
    width = max(1, (maximum + 127) // 128)
    plain_table = _plain(block_table)
    table = (
        plain_table.clone()
        if width == capacity
        else plain_table[:, :width].clone(memory_format=torch.contiguous_format)
    )
    # Page maps can be non-monotonic and can share pages across requests.
    # Verify only slots covered by the live histories. Do this once per plan.
    page_counts = [(length + 127) // 128 for length in lengths]
    if validate_page_table:
        cpu_table = table.cpu()
        for index, count in enumerate(page_counts):
            active = cpu_table[index, :count]
            _require(
                bool(((active >= 0) & (active < page_count)).all()), "page table contains an invalid physical page"
            )
    kv = torch.tensor(lengths, dtype=torch.int32, device=device)
    cu = torch.arange(0, (batches + 1) * query_length, query_length, dtype=torch.int32, device=device)
    mode = 3 if causal else 0
    if mode == 3:
        if attn_mask is None:
            attn_mask = torch.ones((2048, 2048), dtype=torch.int8).triu_(1).to(device)
        _require(
            attn_mask.device == device and attn_mask.dtype == torch.int8 and attn_mask.shape == (2048, 2048),
            "expected a causal mask template on the query device",
        )
    else:
        _require(attn_mask is None, "omit attn_mask for unmasked decode")
    metadata = flash_mla_metadata(kv, num_query_heads, 1, cu_seqlens_q=cu, mask_mode=mode, layout_q="TND")
    packed = None
    splits = 1
    native_ntd = False
    valid_rows = None
    heads_per_launch = 4
    if split_cache and min(lengths) < query_length:
        rows = [
            [length > 0 and (not causal or length - query_length + s >= 0) for s in range(query_length)]
            for length in lengths
        ]
        valid_rows = torch.tensor(rows, dtype=torch.bool, device=device).view(-1)
    if width > 1 and (min(lengths) >= query_length or split_cache):
        ops = catlass_ops()
        if num_query_heads == 8 and hasattr(ops, "catlass_split_wide"):
            heads_per_launch = num_query_heads
        span, splits = _split_active_pages(page_counts, ops.catlass_cores(table))
        heads = heads_per_launch * query_length
        scale_bits = struct.unpack("=i", struct.pack("=f", scale))[0]
        header = [
            batches,
            heads,
            512,
            page_count,
            128,
            width,
            scale_bits,
            1,
            32,
            32,
            heads,
            1,
            mode,
            64,
            maximum,
            span,
            splits,
            1,
            batches,
            0,
            0,
            0 if split_cache else 576,
        ]
        native_ntd = layout_out == "NTD" and splits > 1
        header.append(batches * query_length if native_ntd else 0)
        if cache_page_strides is not None:
            header.extend((0, *cache_page_strides))
        records = [header + [0] * (32 - len(header))]
        for b, length in enumerate(lengths):
            output_offset = b * heads * 512
            rope_offset = b * heads * 64
            lse_offset = b * heads * splits
            fields = [
                int(length > 0),
                length,
                0,
                128,
                0,
                output_offset,
                0,
                rope_offset,
                0,
                0,
                0,
                0,
                lse_offset,
                0,
                output_offset,
            ]
            records.append(fields + [0] * (32 - len(fields)))
        packed = torch.tensor(records, dtype=torch.int32, device=device)
    return DecodePlan(
        table,
        page_count,
        query_length,
        scale,
        bool(causal),
        kv,
        cu,
        attn_mask,
        metadata,
        packed,
        splits,
        native_ntd,
        maximum,
        split_cache,
        valid_rows,
        cache_page_strides,
        num_query_heads,
        heads_per_launch,
    )


def flash_mla_decode(q, k_cache, plan, *, layout_out="NTD"):
    """Apply a prepared scheduler step to fused Q/KV or latent and RoPE pairs."""
    from .flash_mla_a3 import _plain, _require, flash_mla

    if plan.split_cache:
        return _decode_split(q, k_cache, plan, layout_out)
    batches = plan.block_table.size(0)
    tokens = batches * plan.query_length
    _require(layout_out in ("NTD", "TND"), "expected TND or NTD output")
    _require(
        q.shape == (tokens, 4, 576) and q.dtype == torch.bfloat16 and q.device == plan.block_table.device,
        "query must match the plan and have BF16 shape [B*S,4,576]",
    )
    _require(
        k_cache.shape == (plan.page_count, 128, 1, 576) and k_cache.dtype == q.dtype and k_cache.device == q.device,
        "expected BF16 cache[P,128,1,576] on the query device",
    )
    if plan.block_table.size(1) == 1:
        from .one_page import uniform_single_page

        return uniform_single_page(
            _plain(q),
            _plain(k_cache).squeeze(2),
            plan.block_table,
            plan.cache_seqlens,
            plan.query_length,
            plan.max_kv,
            plan.softmax_scale,
            plan.causal,
            layout_out == "NTD",
        )
    if batches == 1 and plan.tiling is not None and plan.block_table.size(1) <= 32:
        from .group_decode import group_decode

        return group_decode(
            q,
            k_cache.squeeze(2),
            plan.block_table,
            plan.cache_seqlens,
            plan.query_length,
            plan.softmax_scale,
            plan.causal,
            layout_out == "NTD",
        )
    if plan.tiling is None:
        return flash_mla(
            q,
            k_cache,
            plan.block_table,
            plan.cache_seqlens,
            plan.cu_seqlens_q,
            metadata=plan.metadata,
            attn_mask=plan.mask,
            mask_mode=3 if plan.causal else 0,
            layout_q="TND",
            layout_out=layout_out,
            softmax_scale=plan.softmax_scale,
        )[0]
    heads = 4 * plan.query_length
    out = catlass_ops().catlass_run(
        _plain(q).view(batches, heads, 576),
        _plain(k_cache).squeeze(2),
        plan.block_table,
        plan.tiling,
        plan.splits,
        heads,
    )
    if plan.native_ntd:
        out = out.view(4, tokens, 512)
        return out if layout_out == "NTD" else out.transpose(0, 1).contiguous()
    out = out.view(tokens, 4, 512)
    return out.transpose(0, 1).contiguous() if layout_out == "NTD" else out


def _decode_split(query, cache, plan, layout_out):
    """Use the exact separate latent and RoPE inputs accepted by the FIA MLA path."""
    from .flash_mla_a3 import _plain, _require
    from .one_page import uniform_single_page

    b = plan.block_table.size(0)
    s = plan.query_length
    t = b * s
    _require(layout_out in ("NTD", "TND"), "expected TND or NTD output")
    _require(
        isinstance(query, (list, tuple)) and len(query) == 2 and isinstance(cache, (list, tuple)) and len(cache) == 2,
        "split_cache requires (latent, rope) pairs for query and cache",
    )
    cache_page_strides = plan.cache_page_strides or (128 * 512, 128 * 64)
    queries, pages = [], []
    for q, k, channels, page_stride in zip(query, cache, (512, 64), cache_page_strides):
        _require(
            q.dtype == k.dtype == torch.bfloat16 and q.device == k.device == plan.block_table.device,
            "split query and cache require BF16 on the plan device",
        )
        _require(q.shape == (t, plan.num_query_heads, channels), "invalid split query shape")
        _require(
            tuple(k.shape)
            in (
                (plan.page_count, 128, channels),
                (plan.page_count, 128, 1, channels),
                (plan.page_count, 1, 128, channels),
            ),
            "expected a split page cache with 128 tokens per page and a single KV head",
        )
        queries.append(_plain(q))
        if page_stride == 128 * channels:
            pages.append(_plain(k).view(plan.page_count, 128, channels))
        else:
            import torch_npu

            view = k if k.ndim == 3 else k.squeeze(1 if k.size(1) == 1 else 2)
            _require(
                torch_npu.get_npu_format(k) in (0, 2)
                and view.storage_offset() % 16 == 0
                and view.stride() == (page_stride, channels, 1),
                "split cache view must match the planned page stride and use aligned ND storage",
            )
            pages.append(view)
    q, qr = queries
    k, kr = pages
    if plan.block_table.size(1) == 1:
        return uniform_single_page(
            q,
            k,
            plan.block_table,
            plan.cache_seqlens,
            s,
            plan.max_kv,
            plan.softmax_scale,
            plan.causal,
            layout_out == "NTD",
            qr,
            kr,
        )
    if plan.num_query_heads == plan.heads_per_launch:
        return _decode_split_heads(q, qr, k, kr, plan, layout_out)

    # Query heads attend to the same KV head independently. Keep the original
    # cache and page map shared while using the four-head launch ABI.
    low = _decode_split_heads(_plain(q[:, :4]), _plain(qr[:, :4]), k, kr, plan, layout_out)
    high = _decode_split_heads(_plain(q[:, 4:]), _plain(qr[:, 4:]), k, kr, plan, layout_out)
    return torch.cat((low, high), dim=0 if layout_out == "NTD" else 1)


def _decode_split_heads(q, qr, k, kr, plan, layout_out):
    b = plan.block_table.size(0)
    s = plan.query_length
    t = b * s
    heads = plan.heads_per_launch
    if plan.tiling is None:
        return q.new_zeros((heads, t, 512) if layout_out == "NTD" else (t, heads, 512))
    ops = catlass_ops()
    launch = ops.catlass_split_wide if heads == 8 else ops.catlass_split
    out = launch(
        q.view(b, heads * s, 512),
        qr.view(b, heads * s, 64),
        k,
        kr,
        plan.block_table,
        plan.tiling,
        plan.splits,
        heads * s,
    )
    out = out.view(heads, t, 512) if plan.native_ntd else out.view(t, heads, 512)
    if plan.native_ntd != (layout_out == "NTD"):
        out = out.transpose(0, 1).contiguous()
    if plan.valid_rows is not None:
        mask = plan.valid_rows.view(1, t, 1) if layout_out == "NTD" else plan.valid_rows.view(t, 1, 1)
        out = torch.where(mask, out, 0.0)
    return out
