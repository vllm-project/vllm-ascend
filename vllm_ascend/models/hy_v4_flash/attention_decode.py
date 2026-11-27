# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged Decode attention and graph execution."""

import torch
from vllm.forward_context import get_forward_context
from vllm.models.hy_v4_flash.reference import (
    INDEX_DIM,
    window_candidates,
)

from vllm_ascend.hisparse.c8 import C8_ROW_BYTES, dequantize_c8_rows
from vllm_ascend.logger import init_logger_ascend

from . import triton_ops
from .attention_common import (
    BATCH_VERIFY_MAX_QUERIES,
    BATCH_VERIFY_SPARSE_SHAPE,
    BATCH_VERIFY_SWA_SHAPE,
    PAGED_POOL_DECODE_SHAPE,
    _head_bmm,
)
from .c8_read import empty_compact_c8_rows, gather_compact_c8_rows
from .decode_fusions import (
    native_indexpool_select,
    native_mla_context,
    prepare_indexpool_selection,
)
from .index_pool_select import select_pool_candidates, write_pool_scores

logger = init_logger_ascend(__name__)


class DecodeAttentionMixin:
    """Paged Decode attention and graph execution."""

    def _attend_decode_request(
        self,
        meta,
        request,
        seq_len,
        cache,
        block_size,
        query,
        wv,
        out,
        index_query,
        index_weight,
    ):
        start, stop = meta.query_start_loc[request : request + 2]
        count = stop - start
        visible = torch.arange(seq_len - count, seq_len, device=query.device)
        table = meta.block_table[request]
        if self.indexer is None:
            selected = window_candidates(visible, min(self.window, self.max_model_len))
        elif (seq_len - 1) // self.indexer.ratio > self.indexer.budget:
            ratio = self.indexer.ratio
            groups = seq_len // ratio
            pool = torch.empty(
                (groups, INDEX_DIM),
                dtype=torch.bfloat16 if self.decode_data_fusions else torch.float32,
                device=query.device,
            )
            self.ops.copy_pool_keys(cache, table, pool, 0, block_size, ratio, write=False)
            selected = native_indexpool_select(
                index_query[start:stop],
                index_weight[start:stop],
                pool,
                visible,
                fuse_postprocess=self.decode_data_fusions,
            )
            dense = self._dense_decode_candidates(visible, selected.shape[-1])
            selected = torch.where((visible // ratio > self.indexer.budget)[:, None], selected, dense)
        else:
            width = min(seq_len, (self.indexer.budget + 1) * self.indexer.ratio)
            selected = self._dense_decode_candidates(visible, width)
        if self.topk_snapshot is not None:
            self.topk_snapshot[start:stop, : selected.shape[-1]].copy_(selected)
        if self._can_gather_decode_rows(cache, query.dtype):
            rows = gather_compact_c8_rows(cache, table, block_size, selected)
        else:
            rows = self._gather_attention_rows(cache, table, block_size, selected, mask_padding=True)
        context = native_mla_context(
            query[start:stop],
            selected,
            rows,
            self.learnable_sink_param,
            self.scale,
        )
        out[start:stop].copy_(_head_bmm(context, wv).reshape_as(out[start:stop]))

    def _forward_decode_graph(self, meta, cache, block_size, hidden, q_c, positions, query, wv, out):
        seq_lens = meta.decode_seq_lens
        num_tokens = query.shape[0]
        num_queries = num_tokens // seq_lens.shape[0]
        geometry = {}
        if self.use_pd_decode and query.device.type == "npu":
            # Cache only read-only tensors within this forward and stream.
            context = get_forward_context()
            stream = torch.npu.current_stream(query.device).npu_stream
            key = (id(meta), num_tokens, query.device, stream)
            geometry = context.additional_kwargs.setdefault("hy4_decode_geometry", {}).setdefault(key, {})
        layout = geometry.get("logical")
        if layout is None:
            live = meta.slot_mapping[:num_tokens].view(-1, num_queries) >= 0
            seq_lens = seq_lens.long().masked_fill(~live[:, 0], 0)
            offsets = torch.arange(num_queries, device=query.device)
            logical = (seq_lens[:, None] - num_queries + offsets).masked_fill(~live, -1)
            layout = geometry["logical"] = (live, seq_lens, offsets, logical)
        else:
            logger.info_once("HYV4 Flash reuses Decode query geometry within a forward.")
        live, seq_lens, offsets, logical = layout
        snapshot = self.topk_snapshot
        window = min(self.window, self.max_model_len)
        shared_positions = None
        if (
            self.share_swa_c8_reads
            and self.indexer is None
            and num_queries == BATCH_VERIFY_MAX_QUERIES
            and window == BATCH_VERIFY_SWA_SHAPE[1]
            and (self.use_pd_decode or self.local_heads == BATCH_VERIFY_SWA_SHAPE[0])
            and cache.shape[-1] % torch.int32.itemsize == 0
        ):
            key = ("window_positions", window)
            shared_positions = geometry.get(key)
            if shared_positions is None:
                # The union of consecutive windows has window + num_queries - 1 rows.
                history_offsets = torch.arange(1 - window - num_queries, 0, device=query.device)
                shared_positions = seq_lens[:, None] + history_offsets
                shared_positions.masked_fill_(shared_positions > logical.amax(dim=1, keepdim=True), -1)
                geometry[key] = shared_positions

        batch_size = seq_lens.shape[0]
        if (
            self.cann_c8_swa_decode
            and self.use_pd_decode
            and self.use_fp8_cache
            and self.indexer is None
            and num_queries == BATCH_VERIFY_MAX_QUERIES
            and window == 512
            and self.local_heads == 64
            and query.dtype == torch.bfloat16
            and query.device.type == "npu"
        ):
            from .c8_decode import attend_compact_c8

            lane = torch.arange(window, device=query.device)
            lengths = (logical + 1).clamp(0, window)
            selected = (logical + 1 - window).clamp_min(0)[..., None] + lane
            selected.masked_fill_(lane >= lengths[..., None], -1)
            packed_rows = [
                self._gather(
                    cache, meta.block_table[request], block_size, selected[request].clamp_min(0), slice(C8_ROW_BYTES)
                )
                for request in range(batch_size)
            ]
            selected = selected.flatten(0, 1)
            packed = torch.cat(packed_rows, dim=0)
            context = attend_compact_c8(query, packed, selected, self.learnable_sink_param, self.scale)
            value = _head_bmm(context, wv)
            out[:num_tokens].copy_(value.reshape_as(out[:num_tokens]))
            out[:num_tokens].masked_fill_(~live.reshape(-1, 1), 0)
            if snapshot is not None:
                snapshot[:num_tokens, :window].copy_(selected)
            logger.info_once("HYV4 Flash uses compact CANN C8 attention for sliding-window Decode.")
            return
        if (
            shared_positions is not None
            and self.use_pd_decode
            and self.use_fp8_cache
            and query.device.type == "npu"
            and self.local_heads == 64
            and query.shape[-1] == 576
            and batch_size in (2, 3, 4, 5, 8)
        ):
            logger.info_once("HYV4 Flash batches sliding-window Decode attention across requests.")
            if batch_size == 8:
                logger.info_once("HYV4 Flash batches sliding-window Decode attention for eight requests.")
            key = ("window_selection", window)
            selection = geometry.get(key)
            if selection is None:
                selected = window_candidates(logical.flatten(), window)
                window_rows = (offsets[:, None] + torch.arange(window, device=query.device)).flatten()
                selection = geometry[key] = (selected, window_rows)
            selected, window_rows = selection
            if snapshot is not None:
                snapshot[:num_tokens, :window].copy_(selected)
            direct_rows = self._can_gather_decode_rows(cache, query.dtype)
            if direct_rows:
                rows = empty_compact_c8_rows(num_tokens, window, query.device)
                for request in range(batch_size):
                    start = request * num_queries
                    stop = start + num_queries
                    gather_compact_c8_rows(
                        cache,
                        meta.block_table[request],
                        block_size,
                        selected[start:stop],
                        output=tuple(tensor[start:stop] for tensor in rows),
                    )
            else:
                rows = torch.empty(
                    (num_tokens, window, query.shape[-1]),
                    dtype=torch.float32,
                    device=query.device,
                )
                for request in range(batch_size):
                    history = self._gather_attention_rows(
                        cache,
                        meta.block_table[request],
                        block_size,
                        shared_positions[request],
                        mask_padding=True,
                    )
                    start = request * num_queries
                    target = rows[start : start + num_queries].view(-1, rows.shape[-1])
                    torch.index_select(history, 0, window_rows, out=target)
            self._attend_decoded(query, selected, rows, wv, out[:num_tokens])
            out[:num_tokens].masked_fill_(~live.reshape(-1, 1), 0)
            return

        score_groups = (
            self.indexer is not None
            and not self._short_indexpool_capture
            and self.max_model_len > (self.indexer.budget + 1) * self.indexer.ratio
        )
        use_paged_pool = False
        if score_groups:
            index_query, index_weight = self.indexer.encode_queries(
                hidden[:num_tokens],
                q_c[:num_tokens],
                positions[..., :num_tokens],
                full_graph=True,
            )
            use_paged_pool = (
                self.fuse_indexpool_select
                and self.incremental_index_pool
                and (self.use_pd_decode or self.local_heads == BATCH_VERIFY_SPARSE_SHAPE[0])
                and (
                    num_queries,
                    self.indexer.num_heads,
                    self.indexer.ratio,
                    self.indexer.budget,
                )
                == PAGED_POOL_DECODE_SHAPE
                and cache.device.type == "npu"
                and meta.block_table.stride(-1) == 1
                and index_query.is_contiguous()
                and index_weight.is_contiguous()
            )
        batch_sparse = (
            use_paged_pool
            and self.use_pd_decode
            and self.use_fp8_cache
            and self.ops is triton_ops
            and cache.shape[-1] % torch.int32.itemsize == 0
            and self.local_heads == 32
            and query.shape[-1] == 576
            and (self.indexer.budget + 1) * self.indexer.ratio == 2048
            and batch_size in (2, 4, 8)
        )
        if batch_sparse:
            logger.info_once("HYV4 Flash batches sparse Decode attention across requests.")
            sparse_selected = []
        batched_selected = None
        if use_paged_pool and self.use_pd_decode and self.ops is triton_ops and batch_size in (2, 4, 8):
            logger.info_once("HYV4 Flash batches Decode IndexPool selection across requests.")
            if batch_size == 8:
                logger.info_once("HYV4 Flash batches Decode IndexPool selection for eight requests.")
            ratio = self.indexer.ratio
            budget = self.indexer.budget
            max_groups = self.max_model_len // ratio
            scores = (
                None
                if self.decode_index_pages is not None
                else torch.empty((batch_size, num_queries, max_groups), dtype=torch.float32, device=query.device)
            )
            for request in range(batch_size):
                start = request * num_queries
                stop = start + num_queries
                table = meta.block_table[request]
                self._update_decode_pool(cache, table, block_size, seq_lens[request], num_queries, positions.ndim)
                if scores is not None:
                    write_pool_scores(
                        index_query[start:stop],
                        index_weight[start:stop],
                        cache,
                        table,
                        logical[request],
                        block_size,
                        max_groups,
                        ratio,
                        budget,
                        scores[request],
                    )
            if self.decode_index_pages is not None:
                from .index_pool_decode import select_pages

                batched_selected = select_pages(
                    index_query,
                    index_weight,
                    self.decode_index_pages,
                    meta.block_table[:batch_size],
                    logical,
                    max_groups,
                )
                logger.info_once("HYV4 Flash uses paged CANN LightningIndexerV2 for Decode verification.")
            else:
                batched_selected = select_pool_candidates(
                    scores.flatten(0, 1), logical.flatten(), max_groups, ratio, budget
                ).view(batch_size, num_queries, -1)
        for request in range(seq_lens.shape[0]):
            start = request * num_queries
            stop = start + num_queries
            table = meta.block_table[request]
            visible = logical[request]
            if self.indexer is None:
                candidate_key = ("window_candidates", window)
                candidates = geometry.get(candidate_key)
                if candidates is None:
                    candidates = window_candidates(logical.flatten(), window).view(batch_size, num_queries, -1)
                    geometry[candidate_key] = candidates
                selected = candidates[request]
            else:
                if self.incremental_index_pool and batched_selected is None:
                    self._update_decode_pool(
                        cache,
                        table,
                        block_size,
                        seq_lens[request],
                        num_queries,
                        positions.ndim,
                    )
                if batched_selected is not None:
                    selected = batched_selected[request]
                elif score_groups:
                    ratio = self.indexer.ratio
                    pool = self._read_decode_pool(cache, table, block_size, seq_lens[request], positions.ndim)
                    selection_key = (
                        "indexpool_selection",
                        request,
                        num_queries,
                        pool.shape[0],
                    )
                    selection_state = geometry.get(selection_key)
                    if selection_state is None:
                        selection_state = prepare_indexpool_selection(visible, num_queries, pool.shape[0])
                        geometry[selection_key] = selection_state
                    selected = native_indexpool_select(
                        index_query[start:stop],
                        index_weight[start:stop],
                        pool,
                        visible,
                        selection_state,
                        fuse_postprocess=self.decode_data_fusions,
                    )
                    candidate_key = ("dense_candidates", selected.shape[-1])
                    dense = geometry.get(candidate_key)
                    if dense is None:
                        dense = self._dense_decode_candidates(logical.flatten(), selected.shape[-1]).view(
                            batch_size, num_queries, -1
                        )
                        geometry[candidate_key] = dense
                    mask_key = ("sparse_candidate_rows", ratio, self.indexer.budget)
                    sparse_rows = geometry.get(mask_key)
                    if sparse_rows is None:
                        sparse_rows = logical // ratio > self.indexer.budget
                        geometry[mask_key] = sparse_rows
                    selected = torch.where(sparse_rows[request, :, None], selected, dense[request])
                else:
                    width = min(
                        self.max_model_len,
                        (self.indexer.budget + 1) * self.indexer.ratio,
                    )
                    candidate_key = ("dense_candidates", width)
                    candidates = geometry.get(candidate_key)
                    if candidates is None:
                        candidates = self._dense_decode_candidates(logical.flatten(), width).view(
                            batch_size, num_queries, -1
                        )
                        geometry[candidate_key] = candidates
                    selected = candidates[request]
            if snapshot is not None:
                snapshot[start:stop, : selected.shape[-1]].copy_(selected)
            if batch_sparse:
                sparse_selected.append(selected)
                continue
            if (
                self.cann_c8_decode
                and self.use_pd_decode
                and self.use_fp8_cache
                and self.indexer is not None
                and query.dtype == torch.bfloat16
                and query.device.type == "npu"
                and self.local_heads == 32
                and num_queries == BATCH_VERIFY_MAX_QUERIES
                and selected.shape[-1] == 2048
            ):
                from .c8_decode import attend_compact_c8

                packed = self._gather(cache, table, block_size, selected.clamp_min(0), slice(C8_ROW_BYTES))
                context = attend_compact_c8(query[start:stop], packed, selected, self.learnable_sink_param, self.scale)
                value = _head_bmm(context, wv)
                out[start:stop].copy_(value.reshape_as(out[start:stop]))
                logger.info_once("HYV4 Flash uses compact CANN C8 attention for sparse Decode verification.")
                continue
            q = query[start:stop]
            if self._can_gather_decode_rows(cache, q.dtype):
                rows = gather_compact_c8_rows(cache, table, block_size, selected)
            elif shared_positions is None:
                rows = self._gather_attention_rows(cache, table, block_size, selected, mask_padding=True)
            else:
                history = self._gather_attention_rows(
                    cache,
                    table,
                    block_size,
                    shared_positions[request],
                    mask_padding=True,
                )
                rows = history.unfold(0, window, 1).transpose(1, 2)
            self._attend_decoded(
                q,
                selected,
                rows,
                wv,
                out[start:stop],
            )
        if batch_sparse:
            logger.info_once("HYV4 Flash gathers sparse Decode C8 cache rows for the whole rank with CANN.")
            selected = torch.stack(sparse_selected)
            selected = selected.flatten(0, 1)
            if self.cann_c8_decode:
                from .c8_decode import attend_compact_c8

                packed = self._gather(
                    cache,
                    meta.block_table[:batch_size],
                    block_size,
                    selected.view(batch_size, -1).clamp_min(0),
                    slice(C8_ROW_BYTES),
                ).view(num_tokens, selected.shape[-1], C8_ROW_BYTES)
                context = attend_compact_c8(query, packed, selected, self.learnable_sink_param, self.scale)
                value = _head_bmm(context, wv)
                out[:num_tokens].copy_(value.reshape_as(out[:num_tokens]))
                logger.info_once("HYV4 Flash uses batched compact CANN C8 attention for sparse Decode verification.")
                return
            if self._can_gather_decode_rows(cache, query.dtype):
                rows = empty_compact_c8_rows(num_tokens, selected.shape[-1], query.device)
                for request in range(batch_size):
                    start = request * num_queries
                    stop = start + num_queries
                    gather_compact_c8_rows(
                        cache,
                        meta.block_table[request],
                        block_size,
                        selected[start:stop],
                        output=tuple(tensor[start:stop] for tensor in rows),
                    )
            else:
                logical = selected.view(batch_size, -1).clamp_min(0)
                packed = self._gather(
                    cache,
                    meta.block_table[:batch_size],
                    block_size,
                    logical,
                    slice(self.kv_storage_width),
                )
                rows = dequantize_c8_rows(packed).view(num_tokens, selected.shape[-1], query.shape[-1])
                rows = rows.masked_fill(selected[..., None] < 0, 0)
            self._attend_decoded(query, selected, rows, wv, out[:num_tokens])
        if shared_positions is not None:
            # Inactive queries may overlap live KV rows in the unfolded view.
            out[:num_tokens].masked_fill_(~live.reshape(-1, 1), 0)

    @staticmethod
    def _dense_decode_candidates(visible, width):
        history = torch.arange(width, device=visible.device).expand(visible.shape[0], -1)
        return history.masked_fill(history > visible[:, None], -1)

    def _update_decode_pool(self, cache, table, block_size, seq_len, num_queries, position_ndim):
        ratio = self.indexer.ratio
        first = (seq_len - num_queries).clamp_min(0) // ratio
        end = seq_len // ratio
        # A width-q decode can close at most ceil(q / ratio) groups.
        width = ((num_queries + ratio - 1) // ratio) * ratio
        history = first * ratio + torch.arange(width, device=seq_len.device)
        history = history.masked_fill(history >= end * ratio, 0)
        raw, key_pos = self._read_index_rows(cache, table, block_size, history, position_ndim)
        keys = self.indexer.encode_keys(raw, key_pos, full_graph=True)
        self.ops.copy_pool_keys(cache, table, keys, first, block_size, ratio, write=True, end_group=end)
        if self.decode_index_pages is not None:
            from .index_pool_decode import write_pages

            write_pages(keys, self.decode_index_pages, table, first, end, block_size)

    def _read_decode_pool(self, cache, table, block_size, seq_len, position_ndim):
        ratio = self.indexer.ratio
        groups = self.max_model_len // ratio
        end = seq_len // ratio
        if self.incremental_index_pool:
            keys = torch.empty(
                (groups, INDEX_DIM),
                dtype=torch.bfloat16 if self.decode_data_fusions else torch.float32,
                device=cache.device,
            )
            self.ops.copy_pool_keys(cache, table, keys, 0, block_size, ratio, write=False, end_group=end)
            return keys
        history = torch.arange(groups * ratio, device=cache.device)
        history = history.masked_fill(history >= end * ratio, 0)
        raw, key_pos = self._read_index_rows(cache, table, block_size, history, position_ndim)
        return self.indexer.encode_keys(raw, key_pos, full_graph=True)

    def _can_gather_decode_rows(self, cache, output_dtype):
        return (
            getattr(self, "decode_data_fusions", False)
            and self.fuse_c8_qk
            and output_dtype == torch.bfloat16
            and cache.is_contiguous()
            and cache.shape[-1] % torch.int32.itemsize == 0
            and cache.storage_offset() % torch.int32.itemsize == 0
        )
