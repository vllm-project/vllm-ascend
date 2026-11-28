# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.forward_context import get_forward_context
from vllm.v1.worker.workspace import current_workspace_manager

from vllm_ascend.attention.indexer_kpool import Glm5NextKPoolIndexerBackend
from vllm_ascend.device.device_config import is_950
from vllm_ascend.utils import is_mtp_layer

from .mtp import (
    PAGE_SIZE,
    Q6_MAX_CONTEXT,
    Q6_MAX_REQUESTS,
    TOKENS_PER_REQUEST,
    WORKSPACE_PADDING_BYTES,
    mtp_projection_rows,
    mtp_scratch_bytes,
    score_context,
)
from .mtp_attention import MtpQueryAttention
from .weights import prepare_decode_weights


def _dense(tensor, shape, dtype):
    return (
        isinstance(tensor, torch.Tensor) and tensor.shape == shape and tensor.dtype == dtype and tensor.is_contiguous()
    )


def _epsilon(norm):
    return getattr(norm, "eps", getattr(norm, "variance_epsilon", None))


def prepare_decode_layer(impl):
    config = impl.vllm_config
    spec = config.speculative_config
    if (
        spec is None
        or spec.method != "mtp"
        or spec.num_speculative_tokens != TOKENS_PER_REQUEST - 1
        or config.model_config.max_model_len > Q6_MAX_CONTEXT
    ):
        return None
    if (
        not is_950()
        or not impl.runtime_has_indexer
        or impl.skip_topk
        or impl._is_mtp_layer
        or is_mtp_layer(config.model_config.hf_text_config, impl.layer_name)
        or impl.tp_size != 1
        or impl.pcp_enabled
        or config.parallel_config.decode_context_parallel_size != 1
        or config.parallel_config.enable_dbo
        or config.lora_config is not None
        or impl.rl_weight_update_enabled
        or impl.g_proj is not None
        or impl.store_kv_cache_on_host
        or impl.enable_sparse_sfa_turboquant
        or not impl.enable_sparse_sfa_c8
        or impl.c8_cache_dtype != torch.float8_e4m3fn
        or (
            impl.local_num_heads,
            impl.num_kv_heads,
            impl.q_lora_rank,
            impl.kv_lora_rank,
            impl.qk_rope_head_dim,
            impl.qk_nope_head_dim,
            impl.v_head_dim,
            impl.scale,
        )
        != (64, 1, 1536, 512, 0, 256, 256, 256**-0.5)
        or not all(_epsilon(norm) == 1e-5 for norm in (impl.q_a_layernorm, impl.kv_a_layernorm))
    ):
        return None
    ix = impl.indexer.impl
    if (
        not isinstance(ix, Glm5NextKPoolIndexerBackend)
        or not ix.enable_sparse_li_c8
        or (ix.n_head, ix.head_dim, ix.topk_tokens, ix.index_kpool) != (32, 128, 2048, 4)
    ):
        return None
    if any(
        getattr(layer, "weight", None) is None or layer.weight.dtype != torch.float8_e4m3fn
        for layer in (impl.fused_qkv_a_proj, impl.q_proj, impl.o_proj)
    ):
        return None
    return DecodeLayer(impl)


class DecodeLayer:
    def __init__(self, impl):
        self.weights = prepare_decode_weights(impl)
        self.indexer = impl.indexer.impl
        self.epsilon = _epsilon(self.indexer.k_norm)
        scheduler = impl.vllm_config.scheduler_config
        self.model_context = impl.vllm_config.model_config.max_model_len
        self.capacity = min(scheduler.max_num_seqs, Q6_MAX_REQUESTS)
        self.max_tokens = min(self.capacity * TOKENS_PER_REQUEST, scheduler.max_num_batched_tokens)
        rows = mtp_projection_rows(self.max_tokens)
        self.head_panel = MtpQueryAttention.pack_head_panel(self.weights)
        size = mtp_scratch_bytes(self.capacity, self.max_tokens, score_context(self.model_context))
        device = self.weights.qkv_a.device
        # All layers on one worker use the same arena. Reserve it before KV sizing.
        # Kernel scratch starts at zero; index exchange occupies the disjoint suffix.
        scratch_bytes = size + WORKSPACE_PADDING_BYTES
        self.arena = current_workspace_manager().get_persistent_resource(
            ("tilelang_sfa_decode", TOKENS_PER_REQUEST, rows, size),
            lambda: torch.empty(scratch_bytes + rows * 2176 * 4, dtype=torch.uint8, device=device),
        )
        self.scratch = self.arena[:scratch_bytes]
        self.indices = self.arena[scratch_bytes:].view(torch.int32).view(rows, 2176)
        self.runners = {}

    def bind(self, hidden, output, cache, slots, md, idx):
        if idx is None or md.num_prefills or idx.write_metadata is not None or idx.indexer_row_shard is not None:
            return None
        requests = idx.block_table.shape[0]
        tokens = hidden.shape[0]
        if requests < 1 or requests > self.capacity:
            return None
        if (
            not 1 <= tokens <= min(self.max_tokens, TOKENS_PER_REQUEST * requests)
            or self.epsilon != 1e-6
            or not 1 <= md.max_query_len <= TOKENS_PER_REQUEST
        ):
            return None
        context = self.model_context
        output = output[:tokens]
        if not _dense(hidden, (tokens, 4096), torch.bfloat16) or not _dense(output, hidden.shape, torch.bfloat16):
            return None
        if not _dense(cache, (len(cache), 128, 1, 528), torch.float8_e4m3fn):
            return None
        main_cols, index_cols = md.block_table.shape[-1], idx.block_table.shape[-1]
        if (
            min(main_cols, index_cols) < (context + PAGE_SIZE - 1) // PAGE_SIZE
            or not _dense(md.block_table, (requests, main_cols), torch.int32)
            or not _dense(idx.block_table, (requests, index_cols), torch.int32)
            or not _dense(idx.raw_seq_lens, (requests,), torch.int32)
        ):
            return None
        ix = self.indexer
        tail_md = get_forward_context().attn_metadata[ix.tail_cache.prefix]
        keys, scales = ix._bound_caches(ix.k_cache)
        (ring,) = ix._bound_caches(ix.tail_cache)
        ids = tail_md.block_table[:, 0]
        slots = slots[:tokens].view(-1)
        if (
            keys.shape[1:] != (32, 1, 128)
            or keys.dtype != torch.float8_e4m3fn
            or keys.stride() != (4352, 128, 128, 1)
            or scales.shape != (len(keys), 32, 1, 1)
            or scales.dtype != torch.float32
            or scales.stride() != (1088, 1, 1, 1)
            or scales.data_ptr() != keys.data_ptr() + 4096
            or ids.shape != (requests,)
            or ids.dtype != torch.int32
            or ring.dtype != torch.float32
            or not _dense(slots, (tokens,), torch.int32)
        ):
            return None
        storage = keys.view(torch.uint8).as_strided((len(keys), 4352), (4352, 1))
        dest = ix.topk_indices_buffer
        if dest is not None and (
            dest.ndim != 2
            or len(dest) < tokens
            or dest.dtype != torch.int32
            or dest.device != hidden.device
            or not 2051 <= dest.shape[1] <= 2176
        ):
            return None
        width = 2051 if dest is not None and dest.shape[1] == 2051 else 2176
        selected = (
            dest[:tokens]
            if dest is not None and dest.shape[1] == width and dest.is_contiguous()
            else self.indices[:tokens]
        )
        if selected.shape[-1] != width:
            selected = selected.flatten()[: tokens * width].view(tokens, width)
        geometry = (
            requests,
            tokens,
            context,
            len(cache),
            len(keys),
            main_cols,
            index_cols,
            len(ring),
            ring.stride(0),
            ids.stride(0),
            width,
        )
        if (
            ring.shape[1:] != (9, 256)
            or ring.stride() != (289 * 256, 256, 1)
            or tail_md.cann_inputs is None
            or not _dense(tail_md.cann_inputs.write_indices, (requests, 6), torch.int64)
            or not _dense(idx.cum_query_lens, (requests,), torch.int64)
            or not _dense(md.positions, (tokens,), torch.int64)
            or not _dense(md.query_start_loc, (requests + 1,), torch.int32)
            or not _dense(idx.slot_mapping[:tokens], (tokens,), torch.int64)
        ):
            return None
        if geometry not in self.runners:
            if torch.npu.is_current_stream_capturing():
                return None
            self.runners[geometry] = MtpQueryAttention(
                self.weights,
                requests,
                len(keys),
                len(cache),
                len(ring),
                workspace=self.scratch,
                head_panel=self.head_panel,
                tokens=tokens,
                model_context=context,
                table_columns=(main_cols, index_cols),
                ring_table_stride=ids.stride(0),
                index_width=width,
            )
        args = (
            hidden,
            cache,
            md.block_table,
            slots,
            idx.raw_seq_lens,
            idx.cum_query_lens,
            md.positions,
            md.query_start_loc,
            idx.block_table,
            storage,
            ring,
            ids,
            idx.slot_mapping[:tokens],
            tail_md.cann_inputs.write_indices,
            selected,
            output,
        )
        return self.runners[geometry], args, selected
