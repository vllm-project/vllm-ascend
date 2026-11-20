# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Framework-side V4.1 cache placements within a shared allocation."""

from dataclasses import replace
from itertools import chain, zip_longest

import torch
from vllm.config import VllmConfig
from vllm.logger import logger

# Both supported versions expose planning helpers in kv_cache_planning.
from vllm.v1.core.kv_cache_planning import may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    declared_kwarg,
)
from vllm_ascend.device.hardware_profile import (
    HardwareCapability,
    get_current_hardware_profile,
)

STATE_RING_ROWS = 32

# Fixed dimensions accepted by the packaged mixed-quant attention ABI.
# The actual head sizes come from the model config; these bounds reject
# unsupported layouts before allocating caches, rather than override config.
MLA_HEAD_DIM = 512
INDEX_HEAD_DIM = 128
INDEX_FOLD_ROWS = 8


def uses_a5_packed_cache() -> bool:
    """Select the V4.1 cache ABI from the detected hardware profile."""
    return get_current_hardware_profile().supports(HardwareCapability.DSV41_PACKED_CACHE)


def local_index_copy_enabled() -> bool:
    """Whether the indexer K plane gets its own replicated tensor."""
    from vllm_ascend import envs

    return str(envs.VLLM_ASCEND_PCP_LOCAL_INDEX).strip().lower() in ("1", "true", "yes", "on")


def owner_capacity_enabled() -> bool:
    """Whether owner-sharding savings are converted into extra KV blocks."""
    from vllm_ascend import envs

    return str(envs.VLLM_ASCEND_PCP_OWNER_CAPACITY).strip().lower() in ("1", "true", "yes", "on")


def owner_capacity_shard_size(vllm_config: VllmConfig | None) -> int:
    """Per-block physical divisor for owner-sharded tuples (1 = replicated).

    Only owner-sharded tuples get ``page_size / pcp_size``. The index-only
    tuple created by ``VLLM_ASCEND_PCP_LOCAL_INDEX`` stays replicated, as does
    everything when owner sharing is not a supported configuration.
    """
    if vllm_config is None or not owner_capacity_enabled():
        return 1
    from vllm_ascend.attention.context_parallel.pcp_owner_kv import (
        owner_sharding_enabled,
        validate_owner_sharing,
    )

    if not owner_sharding_enabled(vllm_config):
        return 1
    supported, _ = validate_owner_sharing(vllm_config)
    if not supported:
        return 1
    pcp_size = int(getattr(vllm_config.parallel_config, "prefill_context_parallel_size", 1) or 1)
    return max(pcp_size, 1)


def _is_index_only_tuple(layer_names) -> bool:
    """True for the replicated tuple created by ``VLLM_ASCEND_PCP_LOCAL_INDEX``."""
    names = list(layer_names)
    return bool(names) and all(".indexer.k_cache" in name for name in names)


def _physical_page_sizes(page_sizes, layer_tuples, shard_size: int) -> list[int]:
    """Per-block bytes each rank actually allocates for every layer tuple."""
    if shard_size <= 1:
        return list(page_sizes)
    physical: list[int] = []
    for page_size, layer_names in zip(page_sizes, layer_tuples):
        # Mirrors the index-only split in the worker allocator: that tuple is
        # replicated for every PCP rank and keeps its full page.
        if _is_index_only_tuple(layer_names):
            physical.append(page_size)
            continue
        if page_size % shard_size:
            raise ValueError(
                f"Cannot split a {page_size}-byte KV page across pcp_size={shard_size}; "
                "disable VLLM_ASCEND_PCP_OWNER_CAPACITY."
            )
        physical.append(page_size // shard_size)
    return physical


def make_swa_cache_spec(*, block_size, window_size, head_size, dtype, cache_dtype, bounded_replay=False):
    if uses_a5_packed_cache():
        if head_size != MLA_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {MLA_HEAD_DIM}, got {head_size}")
        # FP8 values followed by one BF16 scale per 32 values.
        head_size += (head_size // 32) * torch.bfloat16.itemsize
        dtype = torch.uint8
        cache_dtype = "a5_mxfp8_bf16_scale"
    return AscendSlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        sliding_window=window_size,
        cache_dtype_str=cache_dtype,
        model_version="deepseek_v41",
        alignment=None,
        **declared_kwarg(AscendSlidingWindowMLASpec, "bounded_replay", bounded_replay),
    )


def make_mla_cache_spec(*, block_size, head_size, compress_ratio):
    if uses_a5_packed_cache():
        if head_size != MLA_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {MLA_HEAD_DIM}, got {head_size}")
        # Two FP4 values per byte, followed by one BF16 scale per 16 values.
        head_size = head_size // 2 + (head_size // 16) * torch.bfloat16.itemsize
        dtype = torch.uint8
        scale_dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16
        scale_dtype = torch.int8
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        tokens_per_state=compress_ratio,
        model_version="deepseek_v41",
        # Both supported vLLM lanes (#57169) omit MLA `storage_block_size`.
        **{},
        scale_dtype=scale_dtype,
    )


def make_index_cache_spec(*, block_size, head_size, compress_ratio):
    if uses_a5_packed_cache():
        if head_size != INDEX_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {INDEX_HEAD_DIM}, got {head_size}")
        scale_dim = head_size // 32
        head_size //= 2
        dtype = torch.uint8
        scale_dtype = torch.uint8
    else:
        dtype = torch.int8
        scale_dim = 1
        scale_dtype = torch.float16
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        tokens_per_state=compress_ratio,
        model_version="deepseek_v41",
        # Both supported vLLM lanes (#57169) omit MLA `storage_block_size`.
        **{},
        scale_dim=scale_dim,
        scale_dtype=scale_dtype,
    )


def make_folded_index_cache_spec(*, block_size):
    """QSLI candidate-source view: eight K/scale rows per 544-byte row."""
    if block_size % INDEX_FOLD_ROWS:
        raise ValueError("A5 folded index page must contain full 8-token groups")
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=INDEX_FOLD_ROWS * (INDEX_HEAD_DIM // 2 + INDEX_HEAD_DIM // 32),
        dtype=torch.uint8,
        tokens_per_state=INDEX_FOLD_ROWS,
        model_version="deepseek_v41",
        # Both supported vLLM lanes (#57169) omit MLA `storage_block_size`.
        **{},
    )


def is_deepseek_v41_cache(specs_or_groups):
    if isinstance(specs_or_groups, dict):
        specs = list(specs_or_groups.values())
    else:
        specs = []
        for item in specs_or_groups:
            spec = getattr(item, "kv_cache_spec", item)
            if isinstance(spec, UniformTypeKVCacheSpecs):
                specs.extend(spec.kv_cache_specs.values())
            else:
                specs.append(spec)
    return any(getattr(spec, "model_version", None) == "deepseek_v41" for spec in specs)


def _layer_number(name):
    return int(name.rsplit(".layers.", 1)[1].split(".", 1)[0])


def _draft_layer_number(name):
    return int(("." + name).rsplit(".mtp.", 1)[1].split(".", 1)[0])


def get_layer_tuples(specs):
    """Return DSV4-style ordered layer tuples and their physical page sizes."""
    mla = {name for name, spec in specs.items() if isinstance(spec, AscendMLAAttentionSpec)}
    state = sorted((name for name, spec in specs.items() if isinstance(spec, CircularBufferSpec)), key=_layer_number)
    swa = {name for name, spec in specs.items() if isinstance(spec, AscendSlidingWindowMLASpec)}

    full = sorted((name for name in mla if name.endswith(".long_kv_cache")), key=_layer_number)
    target_swa = sorted((name for name in swa if ".mtp." not in f".{name}"), key=_layer_number)
    draft_swa = sorted((name for name in swa if ".mtp." in f".{name}"), key=_draft_layer_number)

    # A reduced backbone may contain only the initial SWA layers. Without
    # full-context source slots, each SWA layer needs its own physical slot.
    if not full and len(swa) == len(specs):
        names = target_swa + draft_swa
        return [specs[name].unpadded_page_size_bytes for name in names], [(name,) for name in names]

    layer_tuples: list[tuple[str, ...]] = []
    page_sizes: list[int] = []
    for slot_idx, kv_name in enumerate(full):
        prefix = kv_name.rsplit(".", 1)[0]
        index_name = prefix + ".indexer.k_cache"
        folded_name = index_name + "_folded"
        folded_spec = specs.get(folded_name)
        index_spec = specs[index_name]
        kv_spec = specs[kv_name]
        aliases = ([state[slot_idx]] if slot_idx < len(state) else []) + target_swa[slot_idx :: len(full)]
        kv_bytes = kv_spec.unpadded_page_size_bytes
        index_bytes = index_spec.unpadded_page_size_bytes
        folded_bytes = folded_spec.unpadded_page_size_bytes if folded_spec is not None else 0
        if slot_idx < len(draft_swa):
            aliases.append(draft_swa[slot_idx])
        if local_index_copy_enabled():
            # The indexer K plane leaves the owner-sharded tuple and gets its
            # own replicated tensor. Its page is the unpadded index footprint,
            # so no page padding from the long-KV aliases applies.
            long_capacity = max([kv_bytes, *(specs[name].unpadded_page_size_bytes for name in aliases)])
            layer_tuples.append((kv_name, *aliases))
            page_sizes.append(long_capacity)
            layer_tuples.append((index_name, *((folded_name,) if folded_spec is not None else ())))
            page_sizes.append(index_bytes + folded_bytes)
            continue
        capacity = max(
            kv_bytes + index_bytes + folded_bytes,
            *(specs[name].unpadded_page_size_bytes for name in aliases),
        )
        layer_tuples.append((kv_name, index_name, *((folded_name,) if folded_spec is not None else ()), *aliases))
        page_sizes.append(capacity)
    return page_sizes, layer_tuples


def group_cache_specs(specs):
    """Merge full-context resources and pad layer tuples without mutating inputs."""
    if specs and all(isinstance(spec, AscendSlidingWindowMLASpec) for spec in specs.values()):
        return [UniformTypeKVCacheSpecs.from_specs({name: spec}) for name, spec in specs.items()]
    page_sizes, layer_tuples = get_layer_tuples(specs)
    padded = {}
    for page_size, layer_tuple in zip(page_sizes, layer_tuples):
        names = list(layer_tuple)
        kv_name = next((name for name in names if name.endswith(".long_kv_cache")), None)
        index_name = next((name for name in names if name.endswith(".indexer.k_cache")), None)
        folded_name = index_name + "_folded" if index_name is not None else None
        has_folded = folded_name is not None and folded_name in names
        aliases = [name for name in names if name not in {kv_name, index_name, folded_name}]
        if kv_name is not None and index_name is not None:
            # Shared tuple: the index plane is packed after the long-KV rows.
            kv_bytes = specs[kv_name].unpadded_page_size_bytes
            index_bytes = specs[index_name].unpadded_page_size_bytes
            padded[kv_name] = replace(specs[kv_name], page_size_padded=kv_bytes)
            padded[index_name] = replace(
                specs[index_name], page_size_padded=index_bytes if has_folded else page_size - kv_bytes
            )
            if has_folded:
                padded[folded_name] = replace(specs[folded_name], page_size_padded=page_size - kv_bytes - index_bytes)
        elif index_name is not None:
            # Index-only tuple (VLLM_ASCEND_PCP_LOCAL_INDEX): its own page is
            # the unpadded index footprint, with the folded alias sharing it.
            index_bytes = specs[index_name].unpadded_page_size_bytes
            padded[index_name] = replace(specs[index_name], page_size_padded=index_bytes)
            if has_folded:
                padded[folded_name] = replace(specs[folded_name], page_size_padded=page_size - index_bytes)
        elif kv_name is not None:
            padded[kv_name] = replace(specs[kv_name], page_size_padded=specs[kv_name].unpadded_page_size_bytes)
        padded.update((name, replace(specs[name], page_size_padded=page_size)) for name in aliases)

    mla_names = [name for name, spec in padded.items() if isinstance(spec, AscendMLAAttentionSpec)]
    state_names = [name for name, spec in padded.items() if isinstance(spec, CircularBufferSpec)]
    # A decoder-only PP stage has no compression state buffers.
    groups = [
        UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in names})
        for names in (mla_names, state_names)
        if names
    ]

    # Transpose the physical tuples. Each scheduler SWA group takes one layer
    # from every tuple, so its members use distinct slots at the same block ID.
    swa_columns = [
        [
            name
            for name in layer_tuple
            if isinstance(padded[name], AscendSlidingWindowMLASpec) and ".mtp." not in f".{name}"
        ]
        for layer_tuple in layer_tuples
    ]
    # Stage-local SWA counts need not be divisible by the number of KV sources.
    # zip_longest also absorbs the empty column contributed by an index-only
    # tuple (VLLM_ASCEND_PCP_LOCAL_INDEX).
    for names in zip_longest(*swa_columns):
        groups.append(UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in names if name is not None}))

    draft_names = [
        name
        for layer_tuple in layer_tuples
        for name in layer_tuple
        if isinstance(padded[name], AscendSlidingWindowMLASpec) and ".mtp." in f".{name}"
    ]
    if draft_names:
        groups.append(UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in draft_names}))
    return groups


def make_cache_groups(grouped_specs):
    return [KVCacheGroupSpec(layer_names=list(s.kv_cache_specs), kv_cache_spec=s) for s in grouped_specs]


def _specs_from_groups(groups):
    specs = {}
    for group in groups:
        for name in group.layer_names:
            group_spec = group.kv_cache_spec
            specs[name] = (
                group_spec.kv_cache_specs[name] if isinstance(group_spec, UniformTypeKVCacheSpecs) else group_spec
            )
    return specs


def _get_worker_layer_tuples(groups):
    """Pack scheduler groups whose width may exceed the stage's KV source count."""
    specs = _specs_from_groups(groups)
    if specs and all(isinstance(spec, AscendSlidingWindowMLASpec) for spec in specs.values()):
        # Preserve main's independent slots for SWA-only backbones and drafts.
        _, slots = get_layer_tuples(specs)
        return [max(specs[name].page_size_bytes for name in names) for names in slots], slots
    _, source_tuples = get_layer_tuples(specs)
    # A column's entries need distinct slots; different columns may alias.
    columns = [[]]
    index_slots: list[tuple[str, ...]] = []
    for names in source_tuples:
        mla_names = tuple(name for name in names if isinstance(specs[name], AscendMLAAttentionSpec))
        if _is_index_only_tuple(names):
            # VLLM_ASCEND_PCP_LOCAL_INDEX keeps the indexer K plane on its own
            # replicated backing. It must not take SWA/state aliases from the
            # positional packing below, which would make it a sharded slot.
            index_slots.append(mla_names)
        else:
            columns[0].append(mla_names)
    columns.extend(
        [(name,) for name in group.layer_names if not isinstance(specs[name], AscendMLAAttentionSpec)]
        for group in groups
    )
    slots = [tuple(chain.from_iterable(parts)) for parts in zip_longest(*columns, fillvalue=())]
    slots.extend(index_slots)
    page_sizes = [
        max(
            sum(specs[name].page_size_bytes for name in names if isinstance(specs[name], AscendMLAAttentionSpec)),
            *(specs[name].page_size_bytes for name in names),
        )
        for names in slots
    ]
    return page_sizes, slots


def get_deepseek_v41_pool_bytes_per_block(groups, vllm_config: VllmConfig | None = None) -> int:
    page_sizes, layer_tuples = _get_worker_layer_tuples(groups)
    shard_size = owner_capacity_shard_size(vllm_config)
    return sum(_physical_page_sizes(page_sizes, layer_tuples, shard_size))


def get_deepseek_v41_kv_cache_config(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Place stage-local resource pages within one shared backing pool."""
    specs = _specs_from_groups(groups)
    page_sizes, layer_tuples = _get_worker_layer_tuples(groups)
    shard_size = owner_capacity_shard_size(vllm_config)
    physical_page_sizes = _physical_page_sizes(page_sizes, layer_tuples, shard_size)
    capacity = max(available_memory // sum(physical_page_sizes), 0)
    num_blocks = may_override_num_blocks(vllm_config, capacity)
    if shard_size > 1:
        # Do not subtract the staging scratch here: vLLM re-plans the smaller
        # ranks by passing ``min_num_blocks * pool_bytes_per_block`` and the
        # divisor must stay linear, otherwise the re-planned ranks end up with
        # fewer blocks and fail the scheduler's equal-size assertion. The
        # scratch comes from the gpu-memory-utilization headroom instead.
        from vllm_ascend.attention.context_parallel.pcp_owner_stage import owner_stage_reserve_bytes

        logger.info(
            "DSV41 owner KV capacity: num_blocks=%d shard=%d nominal_bytes_per_block=%d "
            "physical_bytes_per_block=%d staging_scratch=%.3f GiB (utilization headroom) available=%.3f GiB",
            num_blocks,
            shard_size,
            sum(page_sizes),
            sum(physical_page_sizes),
            owner_stage_reserve_bytes() / (1024**3),
            available_memory / (1024**3),
        )
    # VLLM_ASCEND_PCP_LOCAL_INDEX gives the indexer K plane its own tuple so it
    # can stay replicated while long KV is owner-sharded. A replicated tuple
    # must not share the owner slab, so put index-only tuples in their own
    # backing: descriptors carry the size of the backing they address and the
    # worker allocates one slab/one replicated tensor per distinct size.
    split = [_is_index_only_tuple(layer_tuples[i]) for i in range(len(page_sizes))]
    bucket_sizes = {
        index_only: num_blocks * sum(page_sizes[i] for i in range(len(page_sizes)) if split[i] == index_only)
        for index_only in (False, True)
    }
    # Equal bucket sizes cannot be told apart by the allocator's size-based
    # backing selection; keep one backing and accept a sharded index.
    two_backings = any(split) and not all(split) and bucket_sizes[False] != bucket_sizes[True]
    tensors: list[KVCacheTensor] = []
    if two_backings:
        slot_offsets = {False: 0, True: 0}
        for i, (page_size, layer_names) in enumerate(zip(page_sizes, layer_tuples)):
            index_only = split[i]
            slot_offset = slot_offsets[index_only]
            slot_offsets[index_only] += num_blocks * page_size
            component_offset = 0
            for name in layer_names:
                spec = specs[name]
                is_mla = isinstance(spec, AscendMLAAttentionSpec)
                tensors.append(
                    KVCacheTensor(
                        size=bucket_sizes[index_only],
                        layers=[name],
                        offset=slot_offset + (component_offset if is_mla else 0),
                        layer_stride=num_blocks * page_size,
                        block_stride=page_size,
                    )
                )
                if is_mla:
                    component_offset += spec.unpadded_page_size_bytes
    else:
        backing_size = num_blocks * sum(page_sizes)
        slot_offset = 0
        for page_size, layer_names in zip(page_sizes, layer_tuples):
            # Long KV, split indexer and optional folded indexer are
            # consecutive within a page. State/SWA groups reuse that page from
            # its beginning.
            component_offset = 0
            for name in layer_names:
                spec = specs[name]
                is_mla = isinstance(spec, AscendMLAAttentionSpec)
                tensors.append(
                    KVCacheTensor(
                        size=backing_size,
                        layers=[name],
                        offset=slot_offset + (component_offset if is_mla else 0),
                        layer_stride=num_blocks * page_size,
                        block_stride=page_size,
                    )
                )
                if is_mla:
                    component_offset += spec.unpadded_page_size_bytes
            slot_offset += num_blocks * page_size
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )
