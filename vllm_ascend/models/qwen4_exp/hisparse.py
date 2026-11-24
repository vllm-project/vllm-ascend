# SPDX-License-Identifier: Apache-2.0
"""QSA with host-backed token KV and device-resident indexer state."""

from dataclasses import dataclass, replace

import torch
from vllm.models.qwen4_exp.common import qsa_indexer
from vllm.models.qwen4_exp.common.qsa import Qwen4ExpQSAMetadata, Qwen4ExpQSAMetadataBuilder
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.index_group import SparseMLAIndexGroupBuilder, get_sparse_mla_index_group_max_rows
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.hisparse.runtime import hisparse_staging_block_capacity
from vllm.v1.kv_cache_interface import SparseCacheRole
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.attention.sfa_hisparse import AscendHiSparsePrefillMetadata, build_hisparse_prefill_staging_plan

from .attention_dispatch import QSARequestPlan, build_qsa_request_plan, dispatch_qsa_attention
from .gbsa import (
    GBSA_COMPRESS_RATIO,
    GBSA_PAGE_SIZE,
    GBSA_RESIDUAL_TOKENS,
    expand_qsa_selected_tokens,
    qsa_gbsa_attention,
    qsa_resident_gbsa_attention,
)
from .qsa import AscendQSABackend, _store_qsa_cache_rows


@dataclass
class QSAHiSparseMetadata(Qwen4ExpQSAMetadata):
    num_reqs: int
    num_decodes: int
    num_decode_tokens: int
    block_size: int
    req_id_per_token: torch.Tensor
    query_start_loc: torch.Tensor
    prefill: AscendHiSparsePrefillMetadata | None
    prefill_dispatch: QSARequestPlan | None


class QSAHiSparseMetadataBuilder(Qwen4ExpQSAMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.NEVER
    supports_update_block_table = False
    supports_draft_decode_metadata_update = False

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        size = vllm_config.scheduler_config.max_num_batched_tokens
        self.request_rows = torch.empty(size, dtype=torch.int32, device=device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        common = common_attn_metadata
        meta = super().build(common_prefix_len, common, fast_build)
        decodes, prefills, decode_rows, _ = split_decodes_and_prefills(common, decode_threshold=1)
        block_size = self.kv_cache_spec.block_size
        prefill = None
        prefill_dispatch = None
        if prefills:
            lengths = common.seq_lens_cpu_upper_bound
            assert lengths is not None
            plan = build_hisparse_prefill_staging_plan(
                common.block_table_tensor[decodes:],
                common.seq_lens[decodes:],
                block_size,
                hisparse_staging_block_capacity(lengths[decodes:], block_size),
            )
            prefill = AscendHiSparsePrefillMetadata(plan)
            prefill_dispatch = build_qsa_request_plan(
                common.query_start_loc_cpu[decodes:] - decode_rows,
                lengths[decodes:],
                meta.num_actual_tokens - decode_rows,
                common.is_prefilling[decodes:],
            )
        return QSAHiSparseMetadata(
            **vars(meta),
            num_reqs=common.num_reqs,
            num_decodes=decodes,
            num_decode_tokens=decode_rows,
            block_size=block_size,
            req_id_per_token=common.token_to_req_indices(self.request_rows),
            query_start_loc=common.query_start_loc,
            prefill=prefill,
            prefill_dispatch=prefill_dispatch,
        )


class AscendQSAHiSparseBackend(AscendQSABackend):
    uses_qsa_page_major_kv_view = False

    @staticmethod
    def get_builder_cls():
        return QSAHiSparseMetadataBuilder

    @classmethod
    def customize_spec(cls, spec):
        # The resolver copies whole token rows, including both K and V.
        return spec

    @classmethod
    def supported_kv_cache_layouts(cls):
        return (KVCacheLayout.BLNHC,)


def build_hisparse_indexer_metadata(common, *buffers, **kwargs):
    if (
        common.max_query_len == 1
        and 0 < common.num_actual_tokens == common.num_reqs
        and common.query_start_loc.device.type == "npu"
        and getattr(common, "pcp_context", None) is None
        and getattr(common, "graph_pad_size", -1) < 0
    ):
        # The upstream kernel disables CUDA PDL on Ascend. One launch produces
        # the coordinates, slots, and compression work for the indexer group.
        return qsa_indexer.build_qsa_metadata_triton(common, *buffers, **kwargs)
    return qsa_indexer._build_qsa_metadata_torch(common, *buffers, **kwargs)


def init_qsa_hisparse(layer, config):
    if config.speculative_config is not None or config.parallel_config.prefill_context_parallel_size != 1:
        raise NotImplementedError("QSA HiSparse requires PCP=1 and ordinary decoding")
    if layer.kv_cache_torch_dtype != torch.bfloat16:
        raise NotImplementedError("QSA HiSparse requires BF16 KV")
    if config.model_config.enforce_eager and hasattr(qsa_indexer, "build_qsa_metadata_triton"):
        qsa_indexer.build_qsa_metadata = build_hisparse_indexer_metadata
    rows = get_sparse_mla_index_group_max_rows(config)
    scratch = layer.topk_indices_buffer.new_empty((rows, layer.indexer.token_topk))
    group, _ = SparseMLAIndexGroupBuilder(scratch, rows).register_layer(
        True, config, row_width=layer.num_kv_heads * 2 * layer.head_dim, kv_cache_dtype=layer.kv_cache_dtype
    )
    layer.index_group = group
    layer.hisparse_cache = group.cache(0)
    layer.attn_backend = AscendQSAHiSparseBackend
    for owner in (layer.indexer.raw_key_cache, layer.indexer.compressed_key_cache):
        owner.uses_core_qsa_indexer_view = True


def sparse_layer_config(config, token_topk):
    return replace(
        config,
        spec=replace(
            config.spec, cache_role=SparseCacheRole.SPARSE, topk_buffer_width=token_topk, is_index_group_leader=True
        ),
    )


def attend_physical(query, packed, physical, num_kv_heads, output):
    """Pack resolved rows into request-local pages for the regular GBSA kernel."""
    n, _, dim = query.shape
    width = physical.shape[1]
    pages = (width + GBSA_PAGE_SIZE - 1) // GBSA_PAGE_SIZE
    valid = physical >= 0
    lengths = valid.sum(dim=1, dtype=torch.int32)
    positions = torch.arange(width, device=query.device, dtype=torch.float32)
    order = (positions + (~valid) * width).argsort(dim=-1)
    indices = physical.gather(1, order)
    indices = torch.nn.functional.pad(indices, (0, pages * GBSA_PAGE_SIZE - width), value=-1)
    rows = packed.view(-1, num_kv_heads, 2 * dim)
    selected = rows.index_select(0, indices.clamp_min(0).long().flatten())
    selected.masked_fill_(indices.reshape(-1, 1, 1) < 0, 0)
    key, value = selected.reshape(n * pages, GBSA_PAGE_SIZE, num_kv_heads, 2 * dim).split(dim, dim=-1)
    groups = (width + GBSA_COMPRESS_RATIO - 1) // GBSA_COMPRESS_RATIO
    compact = torch.arange(groups + GBSA_RESIDUAL_TOKENS, device=query.device, dtype=torch.int32).expand(n, -1)
    table = torch.arange(n * pages, device=query.device, dtype=torch.int32).reshape(n, pages)
    requests = torch.arange(n, device=query.device, dtype=torch.int32)
    qsa_gbsa_attention(
        query, key.contiguous(), value.contiguous(), compact, table, requests, lengths - 1, lengths, output
    )
    return output.masked_fill_(lengths[:, None, None] == 0, 0)


def prepare_resolver_indices(compact, visible, logical):
    # The indexer can expose INT64 positions while LI-v2 and residency
    # both use INT32. Keep residual token IDs in the resolver's dtype.
    tokens = expand_qsa_selected_tokens(compact, visible.to(dtype=logical.dtype))
    # The last partial group can follow holes in LI-v2 output. Keep its
    # valid tokens within the resolver's fixed token_topk budget.
    lanes = torch.arange(tokens.shape[1], device=tokens.device, dtype=torch.float32)
    order = (lanes + (tokens < 0) * tokens.shape[1]).argsort(dim=-1)
    torch.gather(tokens, 1, order[:, : logical.shape[1]], out=logical)


def run_qsa_hisparse(layer, meta, side, selected, query, key, value, output):
    group = layer.index_group
    cache = layer.hisparse_cache
    n = meta.num_actual_tokens
    group.prepare_for_batch(0, meta)
    resident, slots, count = cache.write_target(n, meta.slot_mapping.shape[0])
    packed = torch.cat((key[:count], value[:count]), dim=-1).flatten(1)
    _store_qsa_cache_rows(resident.unsqueeze(2), slots, packed)
    cache.finish_kv_update()
    width = layer.indexer.token_topk // layer.indexer.compress_ratio + GBSA_RESIDUAL_TOKENS
    compact = selected[:n, :width]
    d = meta.num_decode_tokens
    if d and cache.decode_batch and cache.all_context_pages_resident and resident.shape[1] == GBSA_PAGE_SIZE:
        # The connector has already proved every logical page is resident.
        # Its page table is indexed by persistent state slots, not batch rows.
        states = cache.runtime.request_state_indices
        assert states is not None and cache.block_table is not None
        table = cache.block_table.index_select(0, states[: meta.num_reqs])
        qsa_resident_gbsa_attention(
            query[:d],
            resident,
            compact[:d],
            table,
            side.token_to_req[:d],
            side.logical_positions[:d],
            side.seq_lens[: meta.num_reqs],
            output[:d],
        )
        return
    if d:
        requests = side.token_to_req[:d].long()
        visible = torch.minimum(side.logical_positions[:d] + 1, side.seq_lens[requests])
        logical = group.logical_topk_indices[:d]
        prepare_resolver_indices(compact[:d], visible, logical)
        group.set_logical_topk_ready(0)
        physical = cache.swap_in(
            meta.req_id_per_token[:d],
            block_table=cache.source_block_table,
            logical_topk_indices=logical,
            block_size=meta.block_size,
            num_valid_rows=meta.query_start_loc[-1:],
        )
        attend_physical(query[:d], group.physical_kv_cache(0), physical, layer.num_kv_heads, output[:d])
    if d < n:
        host = cache.runtime.host_cache.view(-1, meta.block_size, cache.runtime.row_width)
        staged, table, requests = group.stage_prefill_rows(0, host, meta)
        keys, values = staged.unflatten(-1, (layer.num_kv_heads, 2 * layer.head_dim)).split(layer.head_dim, dim=-1)
        dispatch_qsa_attention(
            query[d:n],
            key[d:n],
            value[d:n],
            keys.contiguous(),
            values.contiguous(),
            compact[d:],
            table,
            requests[: n - d],
            side.logical_positions[d:n],
            side.seq_lens[meta.num_decodes : meta.num_reqs],
            output[d:n],
            meta.prefill_dispatch,
        )
