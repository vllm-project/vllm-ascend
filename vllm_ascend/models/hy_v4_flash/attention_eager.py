# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager Prefill and Decode request execution."""

import os

import torch
from vllm.models.hy_v4_flash.reference import (
    INDEX_DIM,
    sparse_candidates,
    window_candidates,
)
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import get_ascend_device_type, is_950
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.hisparse.c8 import C8_ROW_BYTES
from vllm_ascend.logger import init_logger_ascend

from . import ops as triton_ops
from .attention_ops import (
    BATCH_VERIFY_MAX_QUERIES,
    BATCH_VERIFY_SPARSE_SHAPE,
    BATCH_VERIFY_SWA_SHAPE,
    C8_PREFILL_MAX_HEADS,
    MAX_FUSED_C8_READ_ROWS,
    QUERY_CHUNK,
    _head_bmm,
    logical_positions,
    reuse_prefill_metadata,
)
from .c8_attention import BANDED_HEADS, attend_c8_v2, attend_c8_window, supports_banded_window
from .cache import supports_c8_qk_heads

logger = init_logger_ascend(__name__)


class EagerAttentionMixin:
    """Eager Prefill and Decode request execution."""

    def _attend_eager(self, meta, cache, block_size, hidden, q_c, positions, query, wv, out, *, reuse_output=False):
        parts = [] if reuse_output else None

        def write_output(start, stop, value):
            value = value.reshape_as(out[start:stop])
            if parts is None:
                out[start:stop].copy_(value)
            else:
                parts.append((start, stop, value))

        num_tokens = query.shape[0]
        snapshot = self.topk_snapshot
        uses_indexpool = self.indexer is not None and any(
            (length - 1) // self.indexer.ratio > self.indexer.budget for length in meta.seq_lens
        )
        index_query = index_weight = None
        all_prefill = uses_indexpool and self.use_cann_prefill and bool(meta.prefill_reqs and all(meta.prefill_reqs))
        if uses_indexpool:
            # Whole indexer chunks can store the FP8 bins in the dtype consumed by LI.
            can_store_bf16 = (
                all_prefill
                and self.fuse_c8_qk
                and query.dtype == torch.bfloat16
                and cache.dtype == torch.uint8
                and all(
                    (stop - start) % QUERY_CHUNK == 0
                    and seq_len - (stop - start) >= (self.indexer.budget + 1) * self.indexer.ratio
                    for seq_len, start, stop in zip(meta.seq_lens, meta.query_start_loc, meta.query_start_loc[1:])
                    if stop > start
                )
            )
            index_inputs = (hidden[:num_tokens], q_c[:num_tokens], positions[..., :num_tokens])
            if all_prefill:
                index_query, index_weight = self.indexer.encode_queries(
                    *index_inputs,
                    shared_positions=positions,
                    output_dtype=torch.bfloat16 if can_store_bf16 else torch.float32,
                )
            else:
                index_query, index_weight = self.indexer.encode_queries(*index_inputs)
        for i, seq_len in enumerate(meta.seq_lens):
            start, stop = meta.query_start_loc[i : i + 2]
            if start == stop:
                continue
            if getattr(self, "native_decode_attention", False) and meta.prefill_reqs and not meta.prefill_reqs[i]:
                self._attend_decode_request(
                    meta,
                    i,
                    seq_len,
                    cache,
                    block_size,
                    query,
                    wv,
                    out,
                    index_query,
                    index_weight,
                )
                continue
            num_queries = stop - start
            verify = num_queries > 1 and bool(meta.precise_decode_reqs and meta.precise_decode_reqs[i])
            batch_verify = (
                self.batch_verify_attention
                and verify
                and num_queries <= BATCH_VERIFY_MAX_QUERIES
                and query.device.type == "npu"
            )
            is_prefill = not verify and bool(meta.prefill_reqs and meta.prefill_reqs[i])
            block_table = meta.block_table[i]
            may_score_groups = uses_indexpool and (seq_len - 1) // self.indexer.ratio > self.indexer.budget
            pooled_history = None
            fallback_pool = None
            if may_score_groups:
                if self.incremental_index_pool:
                    ratio = self.indexer.ratio
                    groups = seq_len // ratio
                    if (
                        is_prefill
                        and self.use_cann_prefill
                        and cache.device.type == "npu"
                        and cache.dtype == torch.uint8
                        and cache.is_contiguous()
                        and block_table.dtype in (torch.int32, torch.int64)
                        and block_table.is_contiguous()
                        and block_size == 64
                        and ratio == 4
                    ):
                        pooled_history = self.ops.read_prefill_pool(cache, block_table, groups, block_size, ratio)
                    else:
                        pooled_history = torch.empty((groups, INDEX_DIM), dtype=torch.float32, device=hidden.device)
                        self.ops.copy_pool_keys(cache, block_table, pooled_history, 0, block_size, ratio, write=False)
                else:
                    history = torch.arange(seq_len, device=hidden.device)
                    raw_keys, key_pos = self._read_index_rows(cache, block_table, block_size, history, positions.ndim)
            loop_stop = stop
            if (
                self.fuse_c8_prefill_window
                and is_prefill
                and self.indexer is None
                and self.window == 512
                and query.device.type == "npu"
                and query.dtype == torch.bfloat16
                and query.is_contiguous()
                and cache.dtype == torch.uint8
                and cache.is_contiguous()
                and block_table.is_contiguous()
            ):
                early = max(0, self.window - 1 - (seq_len - num_queries))
                early = ((early + QUERY_CHUNK - 1) // QUERY_CHUNK) * QUERY_CHUNK
                if num_queries - early >= QUERY_CHUNK:
                    # Preserve the existing C8 path for small head counts. A
                    # 32-head suffix can use two 16-head calls when each query
                    # already has a full causal window.
                    suffix = query[start + early : stop]
                    if 0 < query.shape[1] <= C8_PREFILL_MAX_HEADS or (
                        query.shape[1] == 2 * BANDED_HEADS
                        and supports_banded_window(suffix, self.learnable_sink_param, self.window)
                    ):
                        loop_stop = start + early
            reuse_window = is_prefill and self.indexer is None and num_queries > QUERY_CHUNK and loop_stop > start
            batch_early_window = (
                reuse_window
                and loop_stop - start > QUERY_CHUNK
                and snapshot is None
                and query.device.type == "npu"
                and query.dtype == torch.bfloat16
                and query.is_contiguous()
                and cache.dtype == torch.uint8
                and self.batch_prefill_window
            )
            fuse_sparse_prefill = (
                is_prefill
                and may_score_groups
                and query.device.type == "npu"
                and self.fuse_c8_qk
                and query.dtype == torch.bfloat16
                and query.is_contiguous()
                and cache.dtype == torch.uint8
                and cache.is_contiguous()
                and cache.shape[-1] % 4 == 0
                and block_table.is_contiguous()
                and (
                    0 < query.shape[1] <= C8_PREFILL_MAX_HEADS
                    if self.use_cann_prefill
                    else supports_c8_qk_heads(query.shape[1])
                )
            )
            use_cann_sparse = self.use_cann_prefill and fuse_sparse_prefill
            prefill_c8_v2_rows = []
            history_start = max(0, seq_len - num_queries + 1 - self.window) if self.indexer is None else 0
            shared_kv = None
            if (
                reuse_window
                or verify
                and (not may_score_groups or seq_len <= (stop - start) * self.indexer.ratio * (self.indexer.budget + 1))
            ):
                # The sliding-window union is bounded by this step's query
                # count plus the window width, independent of the prefix length.
                shared_end = seq_len - (stop - loop_stop)
                if batch_early_window:
                    shared_positions = logical_positions(history_start, shared_end, hidden.device)
                else:
                    shared_positions = torch.arange(history_start, shared_end, device=hidden.device)
                if (
                    batch_early_window
                    and self.use_cann_prefill
                    and self.use_fp8_cache
                    and HAS_TRITON
                    and get_ascend_device_type() is AscendDeviceType.A5
                    and cache.ndim == 2
                    and cache.is_contiguous()
                    and cache.shape[-1] >= C8_ROW_BYTES
                    and cache.shape[-1] % 4 == 0
                    and cache.storage_offset() % 4 == 0
                    and block_size == 64
                    and block_table.ndim == 1
                    and block_table.is_contiguous()
                    and block_table.device == cache.device
                    and block_table.dtype in (torch.int32, torch.int64)
                    and 0 < shared_positions.numel() <= MAX_FUSED_C8_READ_ROWS
                    and shared_end <= block_table.numel() * block_size
                ):
                    # Reuse the FP32 C8 reader without changing SWA reduction shapes.
                    shared_kv = triton_ops.gather_c8_rows(cache, block_table, block_size, shared_positions)
                else:
                    shared_kv = self._gather_attention_rows(cache, block_table, block_size, shared_positions)
            prefill_query = query[start:loop_stop].float() if reuse_window else None
            first_position = seq_len - num_queries
            end_position = seq_len - (stop - loop_stop)
            if use_cann_sparse:
                prefill_logical = logical_positions(first_position, end_position, hidden.device)
            elif reuse_window and not batch_early_window:
                prefill_logical = torch.arange(first_position, end_position, device=hidden.device)
            else:
                prefill_logical = None
            if (
                batch_verify
                and self.indexer is None
                and (self.local_heads, self.window) == BATCH_VERIFY_SWA_SHAPE
                and seq_len - num_queries + 1 >= self.window
            ):
                # Every query has a complete window, one row wider in history
                # than the preceding query. Use overlapping views of the gather.
                selected = shared_positions.unfold(0, self.window, 1)
                kv_selected = shared_kv.unfold(0, self.window, 1).transpose(1, 2)
                if snapshot is not None:
                    snapshot[start:stop, : self.window].copy_(selected)
                value = self._attend_decoded(query[start:stop], selected, kv_selected, wv)
                write_output(start, stop, value)
                continue
            batch_sparse = (
                batch_verify
                and may_score_groups
                and (seq_len - num_queries) // self.indexer.ratio > self.indexer.budget
                and (self.local_heads, (self.indexer.budget + 1) * self.indexer.ratio) == BATCH_VERIFY_SPARSE_SHAPE
                and cache.shape[-1] % 4 == 0
            )
            candidate_rows = []
            # Match the candidate and reduction widths of a single-token decode
            # for each verification query. Keep the batched projection and MoE.
            cached_groups = -1
            li_cache = None
            li_tail = None
            processed_until = start
            if batch_early_window:
                from .prefill_ops.fusions import attend_window_chunks

                early_contexts = attend_window_chunks(
                    prefill_query,
                    shared_kv,
                    first_position,
                    history_start,
                    min(self.window, seq_len),
                    self.learnable_sink_param,
                    self.scale,
                    self.latent_dim,
                    QUERY_CHUNK,
                    query.dtype,
                )
                for offset, context in early_contexts:
                    lo = start + offset
                    write_output(lo, lo + context.size(0), _head_bmm(context, wv))
                processed_until = loop_stop
            chunk = 1 if verify else QUERY_CHUNK
            early_fusion = (
                os.getenv("HY4_PREFILL_CAUSAL_PREFIX") == "1"
                and is_prefill
                and snapshot is None
                and query.device.type == "npu"
                and is_950()
            )
            if early_fusion and use_cann_sparse:
                from .prefill_ops.indexer import build_full_recall_rows

                history_budget = self.indexer.budget
                ratio = self.indexer.ratio
                first = seq_len - num_queries
                full_recall_end = (history_budget + 1) * ratio
                full_recall_rows = max(0, min(loop_stop - start, full_recall_end - first))
                if full_recall_rows != loop_stop - start:
                    # Preserve the original 16-row groups at the LI boundary.
                    full_recall_rows = full_recall_rows // chunk * chunk
                if full_recall_rows:
                    dtype = torch.int32 if first >= history_budget * ratio else torch.int64
                    selected = reuse_prefill_metadata(
                        query.device,
                        ("full_recall", first, full_recall_rows, history_budget, ratio, dtype),
                        lambda first=first,
                        full_recall_rows=full_recall_rows,
                        history_budget=history_budget,
                        ratio=ratio,
                        dtype=dtype: build_full_recall_rows(
                            first,
                            full_recall_rows,
                            history_budget,
                            ratio,
                            query.device,
                            dtype,
                        ),
                    )
                    prefill_c8_v2_rows.append(selected)
                    processed_until += full_recall_rows
            for lo in range(processed_until, loop_stop, chunk):
                if lo < processed_until:
                    continue
                hi = min(lo + chunk, loop_stop)
                visible_len = seq_len - (stop - hi)
                history_len = visible_len if verify else seq_len
                score_groups = may_score_groups and (history_len - 1) // self.indexer.ratio > self.indexer.budget
                scores = None
                if shared_kv is not None and not score_groups and not reuse_window:
                    first = max(0, visible_len - self.window) if self.indexer is None else 0
                    selected_slice = slice(first - history_start, visible_len - history_start)
                    selected = shared_positions[selected_slice].unsqueeze(0)
                    kv_selected = shared_kv[selected_slice].unsqueeze(0)
                else:
                    logical = (
                        prefill_logical[lo - start : hi - start]
                        if prefill_logical is not None
                        else torch.arange(visible_len - (hi - lo), visible_len, device=hidden.device)
                    )
                    if self.indexer is None:
                        selected = window_candidates(logical, min(self.window, history_len))
                    elif score_groups:
                        groups = history_len // self.indexer.ratio
                        if groups != cached_groups:
                            if pooled_history is not None:
                                pool = pooled_history[:groups]
                            else:
                                end = groups * self.indexer.ratio
                                pool = self.indexer.encode_keys(raw_keys[:end], key_pos[..., :end])
                            cached_groups = groups
                        q_index, w_index = index_query[lo:hi], index_weight[lo:hi]
                        if use_cann_sparse and self.li_prefill.eligible(
                            q_index,
                            w_index,
                            pool,
                            visible_len,
                            hi - lo,
                            self.indexer.ratio,
                            self.indexer.budget,
                        ):
                            if li_cache is None or li_cache.shape[0] != pool.shape[0]:
                                li_cache = pool.to(torch.bfloat16).contiguous()
                            if li_tail is None:
                                li_tail = reuse_prefill_metadata(
                                    hidden.device,
                                    ("li_tail", first_position, end_position),
                                    lambda prefill_logical=prefill_logical: self.li_prefill.prepare_tail(
                                        prefill_logical
                                    ),
                                )
                            # Full Prefill chunks share the same key pool. Only their causal lengths vary.
                            rows = min(
                                self.li_prefill.BATCH_ROWS,
                                (loop_stop - lo) // chunk * chunk,
                            )
                            if rows > chunk:
                                end = lo + rows
                                selected = self.li_prefill.select_many(
                                    index_query[lo:end],
                                    index_weight[lo:end],
                                    li_cache,
                                    visible_len,
                                    tail=li_tail[lo - start : end - start],
                                    tp_shard=self.shard_prefill_li and all_prefill,
                                )
                                prefill_c8_v2_rows.append(selected)
                                processed_until = end
                                continue
                            selected = self.li_prefill.select(
                                q_index,
                                w_index,
                                li_cache,
                                visible_len,
                                tail=li_tail[lo - start : hi - start],
                            )
                        else:
                            score_pool = pool
                            if pooled_history is not None and pooled_history.dtype == torch.bfloat16:
                                if fallback_pool is None:
                                    fallback_pool = pooled_history.float()
                                score_pool = fallback_pool[:groups]
                            select_candidates = sparse_candidates
                            if use_cann_sparse:
                                from .prefill_ops.indexer import select_partial_prefill_pool

                                select_candidates = select_partial_prefill_pool
                                q_index = q_index.float()
                            selected = select_candidates(
                                q_index,
                                w_index,
                                score_pool,
                                logical,
                                self.indexer.ratio,
                                self.indexer.budget,
                            )
                            if use_cann_sparse and seq_len - num_queries >= self.indexer.budget * self.indexer.ratio:
                                # LightningIndexer emits int32. Cast just the fallback rows so
                                # the mixed batch stays int32 through concatenation and CANN.
                                selected = selected.to(torch.int32)
                        if batch_sparse:
                            candidate_rows.append(selected)
                            continue
                    else:
                        # Every complete group fits the recall budget, so only causality matters.
                        visible = torch.arange(visible_len, device=hidden.device)
                        selected = visible.expand(hi - lo, -1).masked_fill(visible > logical[:, None], -1)
                    if use_cann_sparse:
                        prefill_c8_v2_rows.append(selected)
                        continue
                    if fuse_sparse_prefill:
                        scores, kv_selected = self.ops.gather_c8_qk(
                            query[lo:hi], cache, block_table, block_size, selected
                        )
                    elif shared_kv is None:
                        kv_selected = self._gather_attention_rows(cache, block_table, block_size, selected.clamp_min(0))
                    else:
                        valid_indices = (selected - history_start).clamp_min(0)
                        kv_selected = shared_kv.index_select(0, valid_indices.flatten()).view(*selected.shape, -1)
                if snapshot is not None:
                    snapshot[lo:hi, : selected.shape[-1]].copy_(selected)
                value = self._attend_decoded(
                    query[lo:hi],
                    selected,
                    kv_selected,
                    wv,
                    scores=scores,
                    query_fp32=prefill_query[lo - start : hi - start] if reuse_window else None,
                    fuse_prefill_logits=early_fusion and reuse_window,
                )
                write_output(lo, hi, value)
            if loop_stop < stop:
                context = attend_c8_window(
                    query[loop_stop:stop],
                    cache,
                    block_table,
                    block_size,
                    seq_len,
                    self.learnable_sink_param,
                    self.scale,
                    window=self.window,
                )
                if snapshot is not None:
                    logical = torch.arange(seq_len - (stop - loop_stop), seq_len, device=query.device)
                    snapshot[loop_stop:stop, : self.window].copy_(window_candidates(logical, self.window))
                value = _head_bmm(context, wv)
                write_output(loop_stop, stop, value)
                logger.info_once(
                    "HYV4 Prefill uses CANN KvQuantSparseFlashAttentionV2 for sliding windows. "
                    "seq_len=%d prefill_queries=%d fused_queries=%d",
                    seq_len,
                    num_queries,
                    stop - loop_stop,
                )
            if prefill_c8_v2_rows:
                selected = (
                    prefill_c8_v2_rows[0] if len(prefill_c8_v2_rows) == 1 else torch.cat(prefill_c8_v2_rows, dim=0)
                )
                context = attend_c8_v2(
                    query[start:stop],
                    cache,
                    block_table,
                    block_size,
                    selected,
                    seq_len,
                    self.learnable_sink_param,
                    self.scale,
                    compact_prefix=max(
                        0,
                        min(
                            num_queries,
                            self.indexer.budget * self.indexer.ratio - (seq_len - num_queries),
                        ),
                    ),
                )
                if snapshot is not None:
                    snapshot[start:stop, : selected.size(-1)].copy_(selected)
                value = _head_bmm(context, wv)
                write_output(start, stop, value)
                logger.info_once(
                    "HYV4 Prefill used aclnnKvQuantSparseFlashAttentionV2: queries=%d topk=%d seq_len=%d",
                    num_queries,
                    selected.size(-1),
                    seq_len,
                )
            if batch_sparse:
                selected = torch.cat(candidate_rows, dim=0)
                valid_indices = selected.clamp_min(0)
                if shared_kv is None:
                    kv_selected = self.ops.gather_c8_rows(cache, meta.block_table[i], block_size, valid_indices)
                else:
                    kv_selected = shared_kv.index_select(0, valid_indices.flatten()).view(*selected.shape, -1)
                if snapshot is not None:
                    snapshot[start:stop, : selected.shape[-1]].copy_(selected)
                value = self._attend_decoded(query[start:stop], selected, kv_selected, wv, split_pv=True)
                write_output(start, stop, value)

        if parts:
            # Keep the projection output directly when it covers the batch.
            # If several adjacent pieces cover it, materialize the final layout once.
            cursor = 0
            for start, stop, _ in parts:
                if start != cursor:
                    break
                cursor = stop
            else:
                if cursor == out.shape[0]:
                    if len(parts) == 1:
                        return parts[0][2]
                    if len(parts) <= 32:
                        return torch.cat([value for _, _, value in parts], dim=0)
            for start, stop, value in parts:
                out[start:stop].copy_(value)
        return out
