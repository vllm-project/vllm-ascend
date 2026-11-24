# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""DeepSeek V4.1 shared compressed sources on the common HiSparse data plane."""

from dataclasses import replace

import torch
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.utils.math_utils import cdiv
from vllm.v1.hisparse.layout import get_hisparse_kv_cache_groups
from vllm.v1.hisparse.runtime import (
    create_hisparse_cache_handle,
)
from vllm.v1.kv_cache_interface import (
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheGroupSpec,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.attention.sfa_hisparse import build_hisparse_prefill_staging_plan
from vllm_ascend.core.kv_cache_interface import (
    AscendCircularBufferSpec,
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    get_storage_block_size,
)


def offloaded_sources(vllm_config):
    """Follow upstream #65970's selective-owner option (source 20 by default)."""
    transfer = vllm_config.kv_transfer_config
    extra = transfer.kv_connector_extra_config if transfer is not None else {}
    sources = set(extra.get("dsv41_sources", [20]))
    if not sources or not sources <= {2, 8, 14, 20}:
        raise ValueError("dsv41_sources must select KV owners 2, 8, 14, or 20")
    return sources


def enabled(vllm_config):
    return vllm_config.attention_config.hisparse_config is not None


def owner_number(name):
    return int(name.rsplit(".layers.", 1)[1].split(".", 1)[0])


def cache_groups(vllm_config, specs):
    """Separate long-lived GPU KV from reclaimable source copies and SWA."""
    owners = offloaded_sources(vllm_config)
    main = {
        name: replace(
            spec, is_index_group_leader=True, topk_buffer_width=vllm_config.model_config.hf_text_config.index_topk
        )
        for name, spec in specs.items()
        if name.endswith(".long_kv_cache") and owner_number(name) in owners
    }
    if not main:
        raise ValueError("Selected DSV4.1 HiSparse owners are absent from this model")
    selected = main | {
        name: spec
        for name, spec in specs.items()
        if isinstance(spec, AscendMLAAttentionSpec) and spec.cache_role is SparseCacheRole.INDEXER
    }
    groups = get_hisparse_kv_cache_groups(vllm_config, selected)
    assert groups is not None
    # Keep non-offloaded main KV and all indexer K in the same full-length
    # scheduler group. The GPU block IDs are independent of host IDs.
    resident = {
        name: spec for name, spec in specs.items() if isinstance(spec, AscendMLAAttentionSpec) and name not in main
    }
    groups[1] = replace(
        groups[1],
        layer_names=list(resident),
        kv_cache_spec=UniformTypeKVCacheSpecs.from_specs(resident),
    )
    # Four independent SWA layers per group retain the original topology's
    # scheduling granularity, without aliasing the offloaded main bytes.
    for kind in (AscendCircularBufferSpec, AscendSlidingWindowMLASpec):
        matching = {name: spec for name, spec in specs.items() if isinstance(spec, kind)}
        partitions = [matching]
        if kind is AscendSlidingWindowMLASpec:
            partitions = [
                {name: spec for name, spec in matching.items() if spec.is_draft == draft} for draft in (False, True)
            ]
        for partition in partitions:
            width = max(len(partition), 1) if kind is AscendCircularBufferSpec else 4
            names = list(partition)
            for start in range(0, len(names), width):
                subset = {name: partition[name] for name in names[start : start + width]}
                groups.append(
                    KVCacheGroupSpec(
                        list(subset),
                        UniformTypeKVCacheSpecs.from_specs(subset),
                        enable_kv_transfer=False,
                    )
                )

    return groups


def resolve_specs(vllm_config, specs):
    """Preserve model logical blocks and derive compressed resident/hot pages."""
    groups = cache_groups(vllm_config, specs)
    return specs | {
        name: group.kv_cache_spec
        for group in groups
        if isinstance(group.kv_cache_spec, (HiSparseResidentSpec, HiSparseHotSpec))
        for name in group.layer_names
    }


def create_source_handle(vllm_config, layer_number, spec):
    if not enabled(vllm_config) or layer_number not in offloaded_sources(vllm_config):
        return None
    parallel = vllm_config.parallel_config
    if parallel.prefill_context_parallel_size != 1 or parallel.decode_context_parallel_size != 1:
        raise ValueError("DSV4.1 HiSparse currently requires PCP1 and DCP1")
    if spec.dtype != torch.uint8 or spec.head_size != 320:
        raise ValueError("DSV4.1 HiSparse currently requires A5 MXFP4 rows (320 bytes)")
    handle = create_hisparse_cache_handle(
        vllm_config,
        vllm_config.model_config.hf_text_config.index_topk,
        is_index_group_leader=True,
        row_width=spec.head_size,
        kv_dtype=spec.dtype,
        storage_block_size=get_storage_block_size(spec),
    )
    handle.runtime._ascend_v41_source = True
    handle.runtime._ascend_v41_handle = handle
    return handle


def prepare_cache_views(kv_cache_config, kv_caches):
    """Expose A5 PA_BBND views, including separate packed indexer scales."""
    for group in kv_cache_config.kv_cache_groups:
        for name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            if getattr(spec, "model_version", None) != "deepseek_v41" and not (
                isinstance(spec, AscendCircularBufferSpec) and name.endswith(".compressor.state_cache")
            ):
                continue
            cache = kv_caches[name]
            if not isinstance(spec, AscendMLAAttentionSpec) or not spec.scale_dim:
                # Generic allocation uses [B,H,N,C]; the A5 ABI uses [B,N,H,C] with contiguous page dims.
                if cache.shape[1] == 1:
                    kv_caches[name] = torch.as_strided(
                        cache,
                        (cache.shape[0], cache.shape[2], 1, cache.shape[3]),
                        (cache.stride(0), cache.stride(2), cache.stride(2), cache.stride(3)),
                        storage_offset=cache.storage_offset(),
                    )
                else:
                    kv_caches[name] = cache.transpose(1, 2)
                continue
            rows = get_storage_block_size(spec)
            stride = cache.stride(0)
            offset = cache.storage_offset()
            raw = torch.as_strided(cache, (0,), (1,), storage_offset=0)
            key = torch.as_strided(
                raw,
                (cache.shape[0], rows, 1, spec.head_size),
                (stride, spec.head_size, spec.head_size, 1),
                storage_offset=offset,
            )
            scale_offset = offset + rows * spec.head_size
            scale_raw = raw.view(spec.scale_dtype)
            item = spec.scale_dtype.itemsize
            scale = torch.as_strided(
                scale_raw,
                (cache.shape[0], rows, 1, spec.scale_dim),
                (stride // item, spec.scale_dim, spec.scale_dim, 1),
                storage_offset=scale_offset // item,
            )
            kv_caches[name] = (key, scale)


def resident_write_slots(slots, metadata):
    """Only complete C2 rows are written; speculative/padded slots stay masked."""
    slots = slots[: metadata.num_input_tokens]
    ratio = metadata.compress_ratio
    valid = slots >= 0
    if metadata.flat_slot_mapping is not None:
        # Retain replay/draft skip-update decisions made by the model builder.
        valid &= metadata.flat_slot_mapping[: slots.shape[0]] >= 0
    rows = torch.arange(slots.shape[0], device=slots.device)
    valid &= rows < metadata.query_start_loc[metadata.num_actual_reqs]
    valid &= rows < metadata.num_actual_tokens
    if ratio != 1:
        valid &= metadata.positions[: slots.shape[0]].remainder(ratio) == ratio - 1
    return torch.where(valid, slots // ratio, -1)


def finish_source_write(handle, metadata):
    if getattr(metadata, "hisparse_for_capture", False):
        return
    if not has_kv_transfer_group() or not get_kv_transfer_group().has_connector_metadata():
        return
    # Overwriting the tentative end of a C2 group must invalidate any hot copy.
    # The host coordinates, rather than resident block IDs, identify hot rows.
    if (
        metadata.flat_slot_mapping is not None
        and metadata.req_id_per_token is not None
        and handle.runtime.request_state_indices is not None
    ):
        handle.runtime.invalidate_written_slots(metadata.flat_slot_mapping, metadata.req_id_per_token)
    handle.finish_kv_update()


def resolve_attention(handle, metadata, logical_indices, *, new_index_group, logical_lengths=None):
    """Resolve each independent index group once, then reuse it in followers."""
    runtime = handle.runtime
    if not new_index_group:
        return handle._v41_attention_binding
    prefill_binding = getattr(handle, "_v41_prefill_binding", None)
    if metadata.num_decode_tokens != metadata.num_actual_tokens and prefill_binding is not None:
        cache, attention_metadata = prefill_binding
        indices, lengths = logical_indices, logical_lengths
    elif metadata.num_decode_tokens == metadata.num_actual_tokens:
        indices, lengths = handle.swap_in(
            metadata.req_id_per_token,
            handle.source_block_table,
            logical_indices,
            block_size=metadata.storage_block_size,
            num_valid_rows=metadata.query_start_loc[-1:],
            return_valid_counts=True,
        )
        if lengths is not None and lengths.ndim == 1:
            lengths = lengths.unsqueeze(-1)
        cache = runtime.hot.attention_cache.unsqueeze(2)
        # Resolver outputs dense physical row addresses over the whole slab.
        # An identity table makes those addresses meaningful to PA attention.
        table, seq_lens = handle._v41_physical_table, handle._v41_physical_lengths
        attention_metadata = replace(
            metadata,
            block_table=table[: metadata.num_reqs],
            cache_seq_lens=seq_lens[: metadata.num_reqs],
            cmp_residual=None,
        )
    elif handle.all_context_pages_resident:
        cache = handle.view.cache.unsqueeze(2)
        rows = runtime.request_state_indices[: metadata.num_reqs].long().clamp_min(0)
        attention_metadata = replace(
            metadata,
            block_table=handle.block_table.index_select(0, rows),
        )
        indices, lengths = logical_indices, logical_lengths
    else:
        # Prefill/mixed batches gather missing historical pages once per source
        # and index group, while current and shadow-resident rows stay on device.
        assert metadata.seq_lens_cpu is not None
        capacity = sum(
            cdiv(int(length), metadata.logical_block_size)
            for length in metadata.seq_lens_cpu[: metadata.num_actual_reqs]
        )
        plan = build_hisparse_prefill_staging_plan(
            handle.source_block_table[: metadata.num_reqs],
            metadata.cache_seq_lens[: metadata.num_reqs],
            metadata.storage_block_size,
            capacity,
        )
        plan.ensure_gpu_sources(
            handle.block_table,
            runtime.request_state_indices[: metadata.num_reqs],
            metadata.storage_block_size,
        )
        source = runtime.host_cache.view(-1, metadata.storage_block_size, runtime.row_width)
        cache = runtime.gather_prefill_cache(source, plan, handle.view.cache).unsqueeze(2)
        attention_metadata = replace(metadata, block_table=plan.block_table)
        indices, lengths = logical_indices, logical_lengths
    if metadata.num_decode_tokens != metadata.num_actual_tokens:
        handle._v41_prefill_binding = cache, attention_metadata
    handle._v41_attention_binding = cache, attention_metadata, indices, lengths
    return handle._v41_attention_binding


def prepare_source_attention(handle):
    """Allocate immutable physical-address metadata before graph capture."""
    cache = handle.runtime.hot.attention_cache
    runtime = handle.runtime
    # expand shares the same immutable identity row across every request.
    handle._v41_physical_table = (
        torch.arange(
            cache.shape[0],
            dtype=torch.int32,
            device=cache.device,
        )
        .expand(runtime.max_num_reqs, -1)
        .contiguous()
    )
    handle._v41_physical_lengths = torch.full(
        (runtime.max_num_reqs,),
        cache.shape[0] * cache.shape[1],
        dtype=torch.int32,
        device=cache.device,
    )


def prepare_source_batch(handle, metadata):
    """A shared source is written once and staged at most once per forward."""
    handle._v41_prefill_binding = None
    handle._v41_attention_binding = None
    handle.prepare_group_for_batch(metadata)


def build_request_ids(token_rows, query_start_loc, output):
    """Refresh graph-stable request IDs without replacing their storage."""
    output.copy_(torch.bucketize(token_rows, query_start_loc[1:], right=True))
    return output
