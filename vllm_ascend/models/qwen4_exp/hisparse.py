# SPDX-License-Identifier: Apache-2.0
"""QSA with host-backed token KV and device-resident indexer state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.models.qwen4_exp.common import qsa_indexer
from vllm.models.qwen4_exp.common.qsa import Qwen4ExpQSAMetadata, Qwen4ExpQSAMetadataBuilder
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.mla.index_group import SparseMLAIndexGroupBuilder, get_sparse_mla_index_group_max_rows
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.hisparse.runtime import hisparse_staging_block_capacity
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayerConfig, SparseCacheRole
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
from .lightning_indexer import QSAIndexerMetadataProtocol
from .qsa import AscendQSABackend, AscendQwen4ExpQSAAttention, _store_qsa_cache_rows


@dataclass
class AscendQSAHiSparseMetadata(Qwen4ExpQSAMetadata):
    num_reqs: int
    num_decodes: int
    num_decode_tokens: int
    block_size: int
    req_id_per_token: torch.Tensor
    query_start_loc: torch.Tensor
    prefill: AscendHiSparsePrefillMetadata | None
    prefill_dispatch: QSARequestPlan | None


class AscendQSAHiSparseMetadataBuilder(Qwen4ExpQSAMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.NEVER
    supports_update_block_table = False
    supports_draft_decode_metadata_update = False

    def __init__(
        self, kv_cache_spec: AttentionSpec, layer_names: list[str], vllm_config: VllmConfig, device: torch.device
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        size = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_to_req_buffer = torch.empty(size, dtype=torch.int32, device=device)

    def build(
        self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False
    ) -> AscendQSAHiSparseMetadata:
        common = common_attn_metadata
        metadata = super().build(common_prefix_len, common, fast_build)
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(common, decode_threshold=1)
        block_size = self.kv_cache_spec.block_size
        prefill = None
        prefill_dispatch = None
        if num_prefills:
            lengths = common.seq_lens_cpu_upper_bound
            assert lengths is not None
            plan = build_hisparse_prefill_staging_plan(
                common.block_table_tensor[num_decodes:],
                common.seq_lens[num_decodes:],
                block_size,
                hisparse_staging_block_capacity(lengths[num_decodes:], block_size),
            )
            prefill = AscendHiSparsePrefillMetadata(plan)
            prefill_dispatch = build_qsa_request_plan(
                common.query_start_loc_cpu[num_decodes:] - num_decode_tokens,
                lengths[num_decodes:],
                metadata.num_actual_tokens - num_decode_tokens,
                common.is_prefilling[num_decodes:],
            )
        return AscendQSAHiSparseMetadata(
            **vars(metadata),
            num_reqs=common.num_reqs,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            block_size=block_size,
            req_id_per_token=common.token_to_req_indices(self.token_to_req_buffer),
            query_start_loc=common.query_start_loc,
            prefill=prefill,
            prefill_dispatch=prefill_dispatch,
        )


class AscendQSAHiSparseBackend(AscendQSABackend):
    uses_qsa_page_major_kv_view = False

    @staticmethod
    def get_builder_cls() -> type[AscendQSAHiSparseMetadataBuilder]:
        return AscendQSAHiSparseMetadataBuilder

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        # The resolver copies whole token rows, including both K and V.
        return spec

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.BLNHC,)


def bind_qsa_hisparse_kv_caches(forward_context: dict[str, AttentionLayerBase], layer_names: Iterable[str]) -> None:
    """Bind layer views of the allocated pages before connector registration."""
    for layer_name in layer_names:
        # Connector-only allocations, such as the resident pool, have no layer.
        layer = forward_context.get(layer_name)
        # Use the already-bound view, which includes Core's kernel block
        # remapping. The original allocation stays available to the connector.
        if isinstance(layer, MambaBase):
            if isinstance(layer.kv_cache, torch.Tensor):
                layer.bind_kv_cache(layer.kv_cache)
        elif isinstance(layer, (qsa_indexer.QSARawKeyCache, qsa_indexer.QSACompressedKeyCache)):
            if layer.kv_cache.ndim == 4 and layer.kv_cache.shape[1] == 1:
                layer.bind_kv_cache(layer.kv_cache)


def init_qsa_hisparse(layer: AscendQwen4ExpQSAAttention, config: VllmConfig) -> None:
    if config.speculative_config is not None or config.parallel_config.prefill_context_parallel_size != 1:
        raise NotImplementedError("QSA HiSparse requires PCP=1 and ordinary decoding")
    if layer.kv_cache_torch_dtype != torch.bfloat16:
        raise NotImplementedError("QSA HiSparse requires BF16 KV")
    rows = get_sparse_mla_index_group_max_rows(config)
    scratch = layer.topk_indices_buffer.new_empty((rows, layer.indexer.token_topk))
    group, _ = SparseMLAIndexGroupBuilder(scratch, rows).register_layer(
        True, config, row_width=layer.num_kv_heads * 2 * layer.head_dim, kv_cache_dtype=layer.kv_cache_dtype
    )
    layer.index_group = group
    layer.hisparse_cache = group.cache(0)
    layer.attn_backend = AscendQSAHiSparseBackend


def get_qsa_hisparse_cache_config(config: KVCacheLayerConfig, token_topk: int) -> KVCacheLayerConfig:
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
    # These small lane positions are exact in float32, which keeps argsort on AiCore.
    positions = torch.arange(width, device=query.device, dtype=torch.float32)
    order = torch.where(valid, positions, positions + width).argsort(dim=-1)
    indices = physical.gather(1, order)
    indices = torch.nn.functional.pad(indices, (0, pages * GBSA_PAGE_SIZE - width), value=-1)
    rows = packed.view(-1, num_kv_heads, 2 * dim)
    selected = rows.index_select(0, indices.clamp_min(0).long().flatten())
    selected.masked_fill_(indices.reshape(-1, 1, 1) < 0, 0)
    key, value = selected.reshape(n * pages, GBSA_PAGE_SIZE, num_kv_heads, 2 * dim).split(dim, dim=-1)
    groups = (width + GBSA_COMPRESS_RATIO - 1) // GBSA_COMPRESS_RATIO
    compact = (
        torch.arange(groups + GBSA_RESIDUAL_TOKENS, device=query.device, dtype=torch.int32).expand(n, -1).contiguous()
    )
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
    # As in attend_physical, use float32 sort keys to avoid an AiCPU fallback.
    width = tokens.shape[1]
    lanes = torch.arange(width, device=tokens.device, dtype=torch.float32)
    order = torch.where(tokens >= 0, lanes, lanes + width).argsort(dim=-1)
    torch.gather(tokens, 1, order[:, : logical.shape[1]], out=logical)


def run_qsa_hisparse(
    layer: AscendQwen4ExpQSAAttention,
    metadata: AscendQSAHiSparseMetadata,
    indexer_metadata: QSAIndexerMetadataProtocol,
    selected: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
) -> None:
    group = layer.index_group
    cache = layer.hisparse_cache
    num_tokens = metadata.num_actual_tokens
    group.prepare_for_batch(0, metadata)
    resident, slots, num_written = cache.write_target(num_tokens, metadata.slot_mapping.shape[0])
    packed = torch.cat((key[:num_written], value[:num_written]), dim=-1).flatten(1)
    _store_qsa_cache_rows(resident.unsqueeze(2), slots, packed)
    cache.finish_kv_update()
    width = layer.indexer.token_topk // layer.indexer.compress_ratio + GBSA_RESIDUAL_TOKENS
    compact = selected[:num_tokens, :width]
    num_decode_tokens = metadata.num_decode_tokens
    decode_rows = slice(num_decode_tokens)
    if num_decode_tokens and cache.all_context_pages_resident and resident.shape[1] == GBSA_PAGE_SIZE:
        # The connector has already confirmed residency. Map batch request
        # rows to persistent state slots as part of GBSA input preparation.
        request_state_indices = cache.runtime.request_state_indices
        assert request_state_indices is not None and cache.block_table is not None
        qsa_resident_gbsa_attention(
            query[decode_rows],
            resident,
            compact[decode_rows],
            cache.block_table,
            indexer_metadata.token_to_req[decode_rows],
            indexer_metadata.logical_positions[decode_rows],
            indexer_metadata.seq_lens[: metadata.num_reqs],
            output[decode_rows],
            request_state_indices=request_state_indices,
        )
    elif num_decode_tokens:
        # Resolve selected host pages before GBSA reads their physical rows.
        requests = indexer_metadata.token_to_req[decode_rows].long()
        visible = torch.minimum(
            indexer_metadata.logical_positions[decode_rows] + 1, indexer_metadata.seq_lens[requests]
        )
        logical = group.logical_topk_indices[decode_rows]
        prepare_resolver_indices(compact[decode_rows], visible, logical)
        group.set_logical_topk_ready(0)
        physical = cache.swap_in(
            metadata.req_id_per_token[decode_rows],
            block_table=cache.source_block_table,
            logical_topk_indices=logical,
            block_size=metadata.block_size,
            num_valid_rows=metadata.query_start_loc[-1:],
        )
        attend_physical(
            query[decode_rows], group.physical_kv_cache(0), physical, layer.num_kv_heads, output[decode_rows]
        )
    if num_decode_tokens < num_tokens:
        prefill_rows = slice(num_decode_tokens, num_tokens)
        host = cache.runtime.host_cache.view(-1, metadata.block_size, cache.runtime.row_width)
        staged, table, requests = group.stage_prefill_rows(0, host, metadata)
        keys, values = staged.unflatten(-1, (layer.num_kv_heads, 2 * layer.head_dim)).split(layer.head_dim, dim=-1)
        dispatch_qsa_attention(
            query[prefill_rows],
            key[prefill_rows],
            value[prefill_rows],
            keys.contiguous(),
            values.contiguous(),
            compact[prefill_rows],
            table,
            requests[: num_tokens - num_decode_tokens],
            indexer_metadata.logical_positions[prefill_rows],
            indexer_metadata.seq_lens[metadata.num_decodes : metadata.num_reqs],
            output[prefill_rows],
            metadata.prefill_dispatch,
        )
