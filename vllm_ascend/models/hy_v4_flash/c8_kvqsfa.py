# SPDX-License-Identifier: Apache-2.0
"""CANN sparse attention adapters for HY4 Prefill."""

from functools import lru_cache

import torch
import torch_npu

from vllm_ascend.device.device_config import is_950
from vllm_ascend.hisparse.c8 import C8_ROW_BYTES, dequantize_c8_rows


def _pack_c8_pages(cache, block_table, block_size, num_pages):
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
        from .c8_prefill_early import attend_early

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
        if query.shape[0] >= 512 and query.shape[1:] == (8, 576) and indices.shape[-1] == 2048:
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
        compact_table = torch.arange(num_pages, dtype=torch.int32, device=query.device).unsqueeze(0)
        query_ends = torch.full((1,), query.size(0), dtype=torch.int32, device=query.device)
        kv_lengths = torch.full((1,), seq_len, dtype=torch.int32, device=query.device)
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


def _attend_c8_banded(query, cache, block_table, block_size, seq_len, sink, scale, window):
    num_pages = (seq_len + block_size - 1) // block_size
    packed = _pack_c8_pages(cache, block_table, block_size, num_pages).flatten(0, 1)[:seq_len]
    kv = dequantize_c8_rows(packed).to(torch.bfloat16).view(1, seq_len, 1, 576)
    value = kv[..., :512].contiguous()
    stream = torch.npu.current_stream(query.device)
    result, *_ = torch_npu.npu_fusion_attention(
        query.unsqueeze(0),
        kv,
        value,
        query.shape[1],
        "BSND",
        atten_mask=_band_mask(query.device, stream.npu_stream),
        scale=scale,
        sparse_mode=4,
        pre_tockens=window - 1,
        next_tockens=0,
        sink=sink.contiguous(),
    )
    return result.squeeze(0)


def attend_c8_window(query, cache, block_table, block_size, seq_len, sink, scale, *, window):
    """Attend to a full causal window for every query, using only the needed cache pages."""
    first_query = seq_len - query.size(0)
    first_page = (first_query - window + 1) // block_size
    page_offset = first_page * block_size
    first_index = first_query - window + 1 - page_offset
    window_kv_len = seq_len - page_offset
    if (
        query.device.type == "npu"
        and query.dtype == torch.bfloat16
        and query.ndim == 3
        and query.shape[-1] == 576
        and 512 <= query.shape[0] <= 4096
        and window == 512
        and query.shape[1] == 16
        and first_page >= 0
        and sink.dtype == torch.float32
        and sink.shape == (query.shape[1],)
        and is_950()
        and hasattr(torch_npu, "npu_fusion_attention")
    ):
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
    starts = torch.arange(first_index, first_index + query.size(0), dtype=torch.int32, device=query.device)
    offsets = torch.arange(window, dtype=torch.int32, device=query.device)
    selected = starts[:, None] + offsets
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
