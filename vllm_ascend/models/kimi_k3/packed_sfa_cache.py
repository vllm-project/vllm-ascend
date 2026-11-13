# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-outermost K3.1 pools, sized from the layers that actually own KV.

The scheduler still owns disjoint history and recurrent block IDs. One ID
reserves a whole row; a recurrent group may reuse the entire history row.
There is no dependency on a particular draft model or draft layer count.
"""

import math
from collections import Counter
from dataclasses import replace

import torch
import vllm.envs as envs_vllm
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheTensor,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.models.kimi_k3.cache_config import make_kimi_cache_groups
from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.32.0"):
    from vllm.v1.core.kv_cache_utils import _approximate_gcd
else:
    from vllm.v1.core.kv_cache_planning import _approximate_gcd

# All three Ascend MLA/indexer backends consume 128-token kernel pages.
KIMI_KERNEL_BLOCK_SIZE = 128


def uses_block_outermost_sfa(config) -> bool:
    if getattr(config.model_config.hf_config, "model_type", None) != "kimi_k31":
        return False
    # Platform block-size alignment runs before the engine resolves layout.
    # Honor an explicit selection without calling the resolved-only getter.
    layout = config.cache_config.kv_cache_layout
    if layout is None:
        layout = envs_vllm.VLLM_KV_CACHE_LAYOUT
    return layout in ("BLNHC", "BLHNC", "BHLNC")


def _attention_specs(groups):
    return {
        name: group.kv_cache_spec.kv_cache_specs[name]
        for group in groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for name in group.layer_names
    }


def is_packed_sfa_groups(groups) -> bool:
    return any(
        getattr(spec, "block_outermost", False)
        for group in groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for spec in group.kv_cache_spec.kv_cache_specs.values()
    )


def packed_sfa_row_bytes(groups) -> int:
    """LCM alignment permits a hole-spanning view for every kernel page type."""
    attention = _attention_specs(groups)
    payloads = [sum(spec.page_size_bytes for spec in attention.values())]
    payloads.extend(
        len(g.layer_names) * g.kv_cache_spec.page_size_bytes for g in groups if isinstance(g.kv_cache_spec, MambaSpec)
    )
    alignments = []
    for spec in attention.values():
        if spec.block_size % KIMI_KERNEL_BLOCK_SIZE:
            raise ValueError("Block-outermost Kimi cache requires whole 128-token kernel pages.")
        alignments.append(spec.page_size_bytes // (spec.block_size // KIMI_KERNEL_BLOCK_SIZE))
    # A PP stage can own only recurrent layers. Their component views still
    # need dtype alignment, even when no attention page is local.
    alignments.extend(
        get_dtype_size(dtype)
        for g in groups
        if isinstance(g.kv_cache_spec, MambaSpec)
        for dtype in g.kv_cache_spec.dtypes
    )
    alignment = math.lcm(*alignments)
    return cdiv(max(payloads), alignment) * alignment


def packed_sfa_cache_groups(config, specs):
    """Specialize upstream mixed-page packing for Kimi's full + KDA roles."""
    if config.scheduler_config.disable_hybrid_kv_cache_manager:
        raise ValueError("Block-outermost Kimi cache requires the hybrid KV cache manager.")
    parallel = config.parallel_config
    if parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1:
        raise ValueError("Block-outermost Kimi cache currently requires PCP=1 and DCP=1.")
    main = sorted(
        (n for n, s in specs.items() if isinstance(s, AscendMLAAttentionSpec) and not s.is_draft),
        key=extract_layer_index,
    )
    indexer = sorted((n for n, s in specs.items() if isinstance(s, AscendSFAIndexerCacheSpec)), key=extract_layer_index)
    if not main or [extract_layer_index(n) for n in main] != [extract_layer_index(n) for n in indexer]:
        raise ValueError("Every sparse MLA layer must own one matching indexer cache.")
    full = {n: specs[n] for pair in zip(main, indexer) for n in pair}
    full.update(sorted(((n, s) for n, s in specs.items() if s.is_draft), key=lambda item: extract_layer_index(item[0])))
    if any(not isinstance(s, (AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec)) for s in full.values()):
        raise ValueError("Block-outermost Kimi cache requires MLA draft specs.")
    if any(s.num_kv_heads != 1 or s.tokens_per_state != 1 for s in full.values()) or any(
        s.store_on_host for s in full.values() if isinstance(s, AscendMLAAttentionSpec)
    ):
        raise ValueError("Block-outermost Kimi cache requires one uncompressed device-resident latent head.")
    if any(not specs[n].uses_packed_sfa_main_cache for n in main):
        raise ValueError("Block-outermost Kimi cache requires a packed SFA main cache.")
    if len({s.block_size for s in full.values()}) != 1 or len({s.dcp_sharded for s in full.values()}) != 1:
        raise ValueError("Packed history layers must share token block coverage.")
    if any(s.block_size < 1 or s.block_size % KIMI_KERNEL_BLOCK_SIZE for s in full.values()):
        raise ValueError("Block-outermost Kimi cache requires whole 128-token kernel pages.")
    full = {n: replace(s, block_outermost=True) for n, s in full.items()}
    recurrent = {
        n: replace(s, page_size_padded=None)
        for n, s in sorted(specs.items(), key=lambda item: extract_layer_index(item[0]))
        if isinstance(s, MambaSpec)
    }
    if len(full) + len(recurrent) != len(specs):
        raise ValueError("Unexpected cache role in block-outermost Kimi layout.")
    if len(set(recurrent.values())) > 1:
        raise ValueError("Kimi recurrent layers must share state geometry.")
    # Match upstream's balanced repeating-pattern floor. For an unbalanced
    # history bucket (e.g. 24 MLA + 24 indexer + 3 draft), raw state capacity
    # alone determines the recurrent group count. With no draft the 24/24
    # pattern also limits each recurrent group to 24 layers.
    counts = Counter(s.page_size_bytes for s in full.values())
    balanced = len(set(counts.values())) == 1
    pattern_width = len(recurrent) or 1
    if balanced and len(counts) > 1 and recurrent:
        repeats = next(iter(counts.values()))
        pattern_width = _approximate_gcd([repeats, len(recurrent)], lower_bound=repeats)
    anchor = sum(s.page_size_bytes for s in full.values())
    if recurrent:
        state_bytes = next(iter(recurrent.values())).page_size_bytes
        if anchor < state_bytes:
            raise ValueError("The packed history row must cover one recurrent checkpoint; increase --block-size.")
        group_size = min(pattern_width, anchor // state_bytes)
    else:
        group_size = 1
    groups = make_kimi_cache_groups(full, recurrent, recurrent_group_size=group_size, uniform_recurrent=False)
    if groups is None:
        raise ValueError("Packed history layers must share full-attention semantics.")
    groups[0] = replace(
        groups[0], is_eagle_group=any(getattr(s, "non_causal_multi_token_decode", False) for s in full.values())
    )
    specs.update(full)
    specs.update(recurrent)
    return groups


def packed_sfa_cache_config(config, groups, available_memory, override_blocks):
    row_bytes = packed_sfa_row_bytes(groups)
    num_blocks = override_blocks(config, available_memory // row_bytes)
    tensors = []
    offset = 0
    for name, spec in _attention_specs(groups).items():
        tensors.append(
            KVCacheTensor(
                size=num_blocks * row_bytes,
                layers=[name],
                offset=offset,
                layer_stride=spec.page_size_bytes,
                block_stride=row_bytes,
            )
        )
        offset += spec.page_size_bytes
    for group in groups:
        if isinstance(group.kv_cache_spec, MambaSpec):
            tensors.append(
                KVCacheTensor(
                    size=num_blocks * row_bytes,
                    layers=group.layer_names,
                    offset=0,
                    layer_stride=group.kv_cache_spec.page_size_bytes,
                    block_stride=row_bytes,
                )
            )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=config.cache_config.prefix_cache_retention_interval,
    )


def packed_sfa_cache_views(raw, spec, num_blocks, row_bytes, mla_dims=None, fused_mla=False):
    """Expose native kernel pages, with holes between manager blocks.

    The KDA view uses raw state pages at a row stride. Attention views span
    the intervening layers, but metadata never addresses those holes.
    """
    if isinstance(spec, MambaSpec):
        result = []
        offset = 0
        for shape, dtype in zip(spec.shapes, spec.dtypes):
            size = get_dtype_size(dtype)
            typed = raw.view(dtype)
            strides = torch.empty(shape, device="meta").stride()
            result.append(
                typed.as_strided(
                    (num_blocks, *shape),
                    (row_bytes // size, *strides),
                    storage_offset=typed.storage_offset() + offset // size,
                )
            )
            offset += math.prod(shape) * size
        return result
    pages_per_block = spec.block_size // KIMI_KERNEL_BLOCK_SIZE
    page_bytes = spec.page_size_bytes // pages_per_block
    page_stride = row_bytes // page_bytes
    pages = (num_blocks - 1) * page_stride + pages_per_block
    shape = (pages, KIMI_KERNEL_BLOCK_SIZE, 1)

    def view(width, dtype, offset=0):
        size = get_dtype_size(dtype)
        typed = raw.view(dtype)
        return typed.as_strided(
            (*shape, width),
            (page_bytes // size, width, width, 1),
            storage_offset=typed.storage_offset() + offset // size,
        )

    if isinstance(spec, AscendSFAIndexerCacheSpec):
        key_bytes = KIMI_KERNEL_BLOCK_SIZE * spec.head_size * get_dtype_size(spec.dtype)
        key = view(spec.head_size, spec.dtype)
        if not spec.scale_dim:
            return (key,)
        scale = view(spec.scale_dim, spec.scale_dtype, key_bytes)
        if spec.cache_sparse_li_c4:
            scale = scale.unflatten(-1, (spec.scale_dim // 2, 2))
        return (key, scale)
    if spec.uses_packed_sfa_main_cache:
        return (view(spec.head_size, spec.dtype),)
    if fused_mla:
        return (view(spec.head_size, spec.dtype),)
    # Native dense MLA keeps each kernel page's latent and RoPE planes.
    # The model's spec records their combined width; split dimensions are
    # supplied by the allocator, which has access to the actual draft layer.
    if mla_dims is None or sum(mla_dims) != spec.head_size:
        raise ValueError("Dense MLA cache requires its actual latent and RoPE dimensions.")
    nope, rope = mla_dims
    return (view(nope, spec.dtype), view(rope, spec.dtype, KIMI_KERNEL_BLOCK_SIZE * nope * get_dtype_size(spec.dtype)))


def map_packed_sfa_metadata(builder, common, spec, row_bytes):
    """Map dense scheduler/kernel IDs into this role's hole-spanning view.

    Every builder owns its output buffers: history roles sharing ownership
    have different kernel-page byte sizes and must not share mapped tables.
    """
    table, slots = common.block_table_tensor, common.slot_mapping
    pages = spec.block_size // KIMI_KERNEL_BLOCK_SIZE
    page_bytes = spec.page_size_bytes // pages
    step = row_bytes // page_bytes
    buffers = getattr(builder, "_packed_sfa_metadata_buffers", None)
    if buffers is None:
        cfg = builder.vllm_config
        capture_sizes = cfg.compilation_config.cudagraph_capture_sizes or []
        max_rows = max([cfg.scheduler_config.max_num_seqs + 1, *capture_sizes])
        lookahead = cfg.speculative_config.num_speculative_tokens if cfg.speculative_config is not None else 0
        max_cols = cdiv(cfg.model_config.max_model_len + lookahead, spec.block_size) * pages
        max_tokens = max([cfg.scheduler_config.max_num_batched_tokens, *capture_sizes])
        buffers = (
            torch.empty((max_rows, max_cols), dtype=table.dtype, device=table.device),
            torch.empty(max_tokens, dtype=slots.dtype, device=slots.device),
            torch.empty((max_rows, max_cols), dtype=torch.bool, device=table.device),
            torch.empty(max_tokens, dtype=torch.bool, device=slots.device),
        )
        builder._packed_sfa_metadata_buffers = buffers
    mapped_table, mapped_slots, table_mask, slot_mask = buffers
    rows, cols = table.shape
    if rows > mapped_table.shape[0] or cols > mapped_table.shape[1] or slots.numel() > mapped_slots.numel():
        raise ValueError("Packed SFA metadata exceeds its persistent buffer capacity.")
    mapped_table = mapped_table[:rows, :cols]
    mapped_slots = mapped_slots[: slots.numel()]
    table_mask = table_mask[:rows, :cols]
    slot_mask = slot_mask[: slots.numel()]
    torch.div(table, pages, rounding_mode="floor", out=mapped_table)
    mapped_table.mul_(step - pages).add_(table)
    torch.lt(table, 0, out=table_mask)
    mapped_table.masked_fill_(table_mask, -1)
    torch.div(slots, spec.block_size, rounding_mode="floor", out=mapped_slots)
    mapped_slots.mul_(step * KIMI_KERNEL_BLOCK_SIZE - spec.block_size).add_(slots)
    torch.lt(slots, 0, out=slot_mask)
    mapped_slots.masked_fill_(slot_mask, -1)
    return replace(common, block_table_tensor=mapped_table, slot_mapping=mapped_slots)
