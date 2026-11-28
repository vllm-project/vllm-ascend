# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4 Flash c8 attention."""

from functools import lru_cache

import torch
import torch_npu
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_ascend.device.device_config import is_950
from vllm_ascend.hisparse.c8 import C8_ROW_BYTES, dequantize_c8_rows

from .attention_ops import sparse_lengths, window_indices


def attend_early(query, packed, selected, seq_len, sink, scale):
    key = dequantize_c8_rows(packed.reshape(-1, packed.size(-1))[:seq_len].view(torch.uint8))
    positions = selected.masked_fill(selected < 0, seq_len)
    visible = torch.zeros((query.size(0), seq_len + 1), dtype=torch.bool, device=query.device)
    visible.scatter_(1, positions, True)
    logits = torch.matmul(query.float().transpose(0, 1), key.transpose(0, 1)) * scale
    logits.masked_fill_(~visible[:, :-1], -torch.inf)
    sink_logits = sink[:, None, None].expand(-1, query.size(0), 1)
    probs = torch.cat((logits, sink_logits), dim=-1).softmax(-1)[..., :-1]
    context = torch.matmul(probs, key[:, :512])
    return context.transpose(0, 1).contiguous().to(query.dtype)


# Sequence length and head strides may change with every prefill chunk.
@triton.jit(do_not_specialize=["tokens", "max_head_stride", "sum_head_stride", "context_head_stride"])
def _sink_correct_into(
    context,
    row_max,
    row_sum,
    sink,
    output,
    tokens,
    first_head: tl.constexpr,
    context_token_stride: tl.constexpr,
    context_head_stride,
    context_value_stride: tl.constexpr,
    max_head_stride,
    max_token_stride: tl.constexpr,
    sum_head_stride,
    sum_token_stride: tl.constexpr,
    sink_stride: tl.constexpr,
    output_heads: tl.constexpr,
    value_dim: tl.constexpr,
    tile_rows: tl.constexpr,
):
    token = tl.program_id(0) * tile_rows + tl.arange(0, tile_rows)
    head = tl.program_id(1)
    active = token < tokens
    channel = tl.arange(0, value_dim)
    maximum = tl.load(row_max + head * max_head_stride + token * max_token_stride, active, other=0)
    denominator = tl.load(row_sum + head * sum_head_stride + token * sum_token_stride, active, other=1)
    bias = tl.load(sink + (first_head + head) * sink_stride)
    # The learned sink has zero value and contributes only to the softmax denominator.
    weight = tl.sigmoid(maximum + tl.log(denominator) - bias)
    source = token * context_token_stride + head * context_head_stride
    value = tl.load(context + source[:, None] + channel[None, :] * context_value_stride, active[:, None], other=0).to(
        tl.float32
    )
    destination = (token * output_heads + first_head + head) * value_dim
    tl.store(output + destination[:, None] + channel[None, :], value * weight[:, None], active[:, None])


def _write_window_output(context, row_max, row_sum, sink, output, first_head):
    """Write one attention head group into a contiguous [token, head, value] output."""
    tokens, group_heads, value_dim = context.shape
    tile_rows = 16
    _sink_correct_into[(triton.cdiv(tokens, tile_rows), group_heads)](
        context,
        row_max,
        row_sum,
        sink,
        output,
        tokens,
        first_head,
        *context.stride(),
        row_max.stride(1),
        row_max.stride(2),
        row_sum.stride(1),
        row_sum.stride(2),
        sink.stride(0),
        output.size(1),
        value_dim,
        tile_rows,
        num_warps=1,
    )


def attend_compact_c8(query, packed, selected, sink, scale):
    """Treat each verification slot as one independent sparse-attention batch."""
    from cann_ops_transformer import kv_quant_sparse_flash_attention

    queries, heads, _ = query.shape
    width = selected.shape[-1]
    lengths = (selected >= 0).sum(dim=-1, dtype=torch.int32)
    indices = torch.arange(width, device=query.device, dtype=torch.int32)
    indices = indices.view(1, 1, 1, width).expand(queries, 1, 1, width).contiguous()
    # Valid candidates precede padding in both the sorted sparse history and
    # current partial group. Explicit lengths exclude padded tail positions.
    kv = packed.view(queries, width, 1, C8_ROW_BYTES).view(torch.float8_e4m3fn)
    context, _, _ = kv_quant_sparse_flash_attention(
        query.view(queries, 1, heads, 576),
        kv,
        kv,
        indices,
        scale,
        2,
        2,
        actual_seq_lengths_kv=lengths.clamp_min(1),
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=0,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
        sinks=sink.contiguous(),
    )
    return context[:, 0].masked_fill(lengths[:, None, None] == 0, 0)


BANDED_HEADS = 16


def _pack_c8_pages(cache, block_table, block_size, num_pages):
    if HAS_TRITON and 0 < num_pages <= block_table.numel():
        from .prefill_ops.cache import copy_paged_cache_lanes, supports_paged_cache_copy

        if supports_paged_cache_copy(cache, block_table, block_size):
            packed = copy_paged_cache_lanes(cache, block_table, num_pages * block_size, 0, C8_ROW_BYTES)
            return packed.view(num_pages, block_size, C8_ROW_BYTES)

    pages = cache.view(-1, block_size, cache.size(-1))
    # Materialize only the requested pages before stripping indexer state.
    selected_pages = pages.index_select(0, block_table[:num_pages].long())
    return selected_pages[..., :C8_ROW_BYTES].contiguous()


def attend_c8_v2(query, cache, block_table, block_size, selected, seq_len, sink, scale, *, compact_prefix):
    # HY4 appends indexer state to each C8 row. Pack only the request's
    # allocated pages, in logical order, once for the query chunk.
    num_pages = (seq_len + block_size - 1) // block_size
    packed = _pack_c8_pages(cache, block_table, block_size, num_pages)
    packed = packed.view(num_pages, block_size, 1, C8_ROW_BYTES).view(torch.float8_e4m3fn)
    context = []
    if compact_prefix:
        early_kv_len = seq_len - query.size(0) + compact_prefix
        context.append(
            attend_early(
                query[:compact_prefix], packed, selected[:compact_prefix], early_kv_len, sink.contiguous(), scale
            )
        )
    if compact_prefix < query.size(0):
        from cann_ops_transformer import kv_quant_sparse_flash_attention

        query = query[compact_prefix:].contiguous()
        indices = selected[compact_prefix:].to(torch.int32).unsqueeze(1).contiguous()
        if query.shape[0] >= 512 and query.shape[1:] in ((8, 576), (16, 576)) and indices.shape[-1] == 2048:
            # The page gather already placed every requested C8 row in logical
            # order. Truncate the last page and let the kernel index that view.
            dense_kv = packed.reshape(-1, 1, C8_ROW_BYTES)[:seq_len].unsqueeze(0)
            partial, _, _ = kv_quant_sparse_flash_attention(
                query.unsqueeze(0),
                dense_kv,
                dense_kv,
                indices.unsqueeze(0),
                scale,
                2,
                2,
                sparse_block_size=1,
                layout_query="BSND",
                layout_kv="BSND",
                sparse_mode=0,
                attention_mode=2,
                quant_scale_repo_mode=1,
                tile_size=128,
                rope_head_dim=64,
                sinks=sink.contiguous(),
            )
            context.append(partial.squeeze(0))
            return torch.cat(context, dim=0) if len(context) > 1 else context[0]
        compact_table, query_ends, kv_lengths = sparse_lengths(num_pages, query.size(0), seq_len, query.device)
        partial, _, _ = kv_quant_sparse_flash_attention(
            query,
            packed,
            packed,
            indices,
            scale,
            2,
            2,
            block_table=compact_table,
            actual_seq_lengths_query=query_ends,
            actual_seq_lengths_kv=kv_lengths,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=0,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=128,
            rope_head_dim=64,
            sinks=sink.contiguous(),
        )
        context.append(partial)
    return torch.cat(context, dim=0) if len(context) > 1 else context[0]


@lru_cache(maxsize=32)
def _band_mask(device, stream_id):
    # CANN expects the compressed triangular mask for sparse mode 4.
    return torch.ones((2048, 2048), device=device, dtype=torch.bool).triu_(1)


def _prepare_banded_c8(cache, block_table, block_size, seq_len):
    num_pages = (seq_len + block_size - 1) // block_size
    if (
        HAS_TRITON
        and cache.device.type == "npu"
        and cache.dtype == torch.uint8
        and cache.is_contiguous()
        and cache.shape[-1] >= C8_ROW_BYTES
        and cache.shape[-1] % 4 == 0
        and cache.storage_offset() % 4 == 0
        and block_size == 64
        and block_table.ndim == 1
        and block_table.is_contiguous()
        and block_table.device == cache.device
        and block_table.dtype in (torch.int32, torch.int64)
        and 0 < num_pages <= block_table.numel()
    ):
        from .cache import read_c8_banded

        return read_c8_banded(cache, block_table, block_size, seq_len)

    packed = _pack_c8_pages(cache, block_table, block_size, num_pages).flatten(0, 1)[:seq_len]
    key = dequantize_c8_rows(packed).to(torch.bfloat16)
    return key, key[..., :512].contiguous()


def supports_banded_window(query, sink, window):
    return (
        query.device.type == "npu"
        and query.dtype == torch.bfloat16
        and query.ndim == 3
        and query.shape[-1] == 576
        and 512 <= query.shape[0] <= 4096
        and window == 512
        and query.shape[1] in (BANDED_HEADS, 2 * BANDED_HEADS)
        and sink.dtype == torch.float32
        and sink.shape == (query.shape[1],)
        and is_950()
        and hasattr(torch_npu, "npu_fusion_attention")
    )


def _attend_c8_banded(query, cache, block_table, block_size, seq_len, sink, scale, window):
    key, value = _prepare_banded_c8(cache, block_table, block_size, seq_len)
    kv = key.view(1, seq_len, 1, 576)
    value = value.view(1, seq_len, 1, 512)
    stream = torch.npu.current_stream(query.device)
    mask = _band_mask(query.device, stream.npu_stream)
    outputs = []
    split_heads = query.size(1) > BANDED_HEADS
    fused_sink = split_heads and HAS_TRITON and query.device.type == "npu"
    if fused_sink:
        output = query.new_empty((query.size(0), query.size(1), 512))
    for head_start in range(0, query.size(1), BANDED_HEADS):
        heads = slice(head_start, head_start + BANDED_HEADS)
        result, row_max, row_sum, *_ = torch_npu.npu_fusion_attention(
            query[:, heads].contiguous().unsqueeze(0),
            kv,
            value,
            BANDED_HEADS,
            "BSND",
            atten_mask=mask,
            scale=scale,
            sparse_mode=4,
            pre_tockens=window - 1,
            next_tockens=0,
            sink=None if split_heads else sink.contiguous(),
        )
        context = result.squeeze(0)
        if fused_sink:
            _write_window_output(context, row_max, row_sum, sink, output, head_start)
            continue
        if split_heads:
            # The BSND kernel on 950DT returns window-only probabilities for
            # this shape. Add the learned zero-value sink to the denominator.
            log_partition = row_max[0, :, :, 0] + row_sum[0, :, :, 0].log()
            window_weight = torch.sigmoid(log_partition.transpose(0, 1) - sink[heads])
            context = (context.float() * window_weight.unsqueeze(-1)).to(context.dtype)
        outputs.append(context)
    if fused_sink:
        return output
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)


def attend_c8_window(query, cache, block_table, block_size, seq_len, sink, scale, *, window):
    """Attend to a full causal window for every query, using only the needed cache pages."""
    first_query = seq_len - query.size(0)
    first_page = (first_query - window + 1) // block_size
    page_offset = first_page * block_size
    first_index = first_query - window + 1 - page_offset
    window_kv_len = seq_len - page_offset
    if first_page >= 0 and supports_banded_window(query, sink, window):
        return _attend_c8_banded(
            query,
            cache,
            block_table[first_page:],
            block_size,
            window_kv_len,
            sink,
            scale,
            window,
        )
    # Each row has exactly window valid entries. This avoids the unsupported
    # varying effective index width of the C8 kernel at the beginning of a sequence.
    selected = window_indices(first_index, query.size(0), window, query.device)
    return attend_c8_v2(
        query,
        cache,
        block_table[first_page:],
        block_size,
        selected,
        seq_len - page_offset,
        sink,
        scale,
        compact_prefix=0,
    )
