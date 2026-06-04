# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import math
from collections import defaultdict
from functools import partial

import vllm
import vllm.v1.core.kv_cache_utils
import vllm.v1.engine.core  # noqa: F401
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_utils import (
    _approximate_gcd,
    _auto_fit_max_model_len,
    _check_enough_kv_cache_memory,
    _estimate_max_model_len_from_groups,
    _max_memory_usage_bytes_from_groups,
    _project_kv_cache_groups_to_worker,
    _report_kv_cache_config,
    get_kv_cache_groups,
    get_num_blocks,
    get_uniform_page_size,
    may_override_num_blocks,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_config import (
    get_layerwise_config,
    get_layerwise_kv_cache_reuse_layers,
)


def get_kv_cache_reuse_layers(max_layers: int | None = None) -> int | None:
    """Return configured KV cache reuse slots, or None when disabled."""
    if max_layers is None:
        return None
    reuse_layers = get_layerwise_kv_cache_reuse_layers(max_layers)
    if reuse_layers is None:
        return None
    if max_layers is not None and reuse_layers > max_layers:
        raise ValueError(
            "VLLM_ASCEND_KV_POOL_LAYERWISE_NUM_SHARED_BUFFERS must not exceed the number "
            f"of KV cache layers, got {reuse_layers} > {max_layers}."
        )
    return reuse_layers


def get_layerwise_storage_indices(num_layers: int) -> list[list[int]]:
    layerwise_config = get_layerwise_config(num_layers)
    if not layerwise_config.has_layer_reuse:
        return [[layer_index] for layer_index in range(num_layers)]

    independent_layers = set(layerwise_config.independent_layers)
    storage_indices: list[list[int]] = [
        [] for _ in range(layerwise_config.num_shared_buffers)
    ]
    reused_layer_index = 0
    for layer_index in range(num_layers):
        if layer_index in independent_layers:
            storage_indices.append([layer_index])
            continue
        storage_indices[reused_layer_index %
                        layerwise_config.num_shared_buffers].append(layer_index)
        reused_layer_index += 1
    return [indices for indices in storage_indices if indices]


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """
    Generate the KV cache configuration from the KV cache groups and spec
    of each layer.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_groups: The KV cache groups
        available_memory: Memory available for KV cache in bytes
    Returns:
        The generated KVCacheConfig
    """
    if len(kv_cache_groups) == 0:
        # Attention free models do not have KV cache.
        # Return num_blocks=1 as BlockPool always needs a null_block.
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        )

    # Determine how model runners should initialize the KV cache tensors.
    if len(kv_cache_groups) == 1 and isinstance(kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs):
        # Special case: all layers have the same type of KV cache but with
        # different hidden size. Allocate different amount of memory for each
        # layer based on its hidden size.
        group = kv_cache_groups[0]
        reuse_layers = get_kv_cache_reuse_layers(len(group.layer_names))
        per_layer_specs = group.kv_cache_spec.kv_cache_specs

        if reuse_layers is not None:
            page_sizes = {per_layer_specs[layer_name].page_size_bytes for layer_name in group.layer_names}
            if len(page_sizes) != 1:
                raise ValueError(
                    "Layerwise KV cache reuse is not supported for "
                    "UniformTypeKVCacheSpecs with different per-layer page "
                    f"sizes, got {sorted(page_sizes)}."
                )

        if reuse_layers is None:
            storage_slots = [[layer_name] for layer_name in group.layer_names]
        else:
            storage_slots = [
                [group.layer_names[layer_index] for layer_index in indices]
                for indices in get_layerwise_storage_indices(
                    len(group.layer_names))
            ]

        page_size = sum(
            max(per_layer_specs[layer_name].page_size_bytes for layer_name in slot) for slot in storage_slots
        )
        num_blocks = available_memory // page_size
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        kv_cache_tensors = [
            KVCacheTensor(
                size=max(per_layer_specs[layer_name].page_size_bytes for layer_name in slot) * num_blocks,
                shared_by=slot,
            )
            for slot in storage_slots
        ]
    else:
        # General case:
        # We will have group_size memory pools, each is shared by one layer from
        # each group. As layers of different groups have different block table,
        # they will use different parts of the shared Tensor.
        # The memory layout for 3 groups (full.0, full.1), (sw.0, sw.2),
        # (sw.1, padding) will be: (group_size = 2)
        # full.0, sw.0, sw.1: share a Tensor with size=available_memory//2
        # full.1, sw.2: share another Tensor with size=available_memory//2
        group_size = max(len(group.layer_names) for group in kv_cache_groups)

        page_size = get_uniform_page_size([group.kv_cache_spec for group in kv_cache_groups])
        assert group_size > 0, "group_size must be greater than 0"
        reuse_layers = get_kv_cache_reuse_layers(group_size)
        storage_indices = (
            get_layerwise_storage_indices(group_size)
            if reuse_layers is not None
            else [[layer_index] for layer_index in range(group_size)]
        )
        num_blocks = get_num_blocks(vllm_config, len(storage_indices), available_memory, page_size)
        kv_cache_tensors = []
        for indices in storage_indices:
            shared_by = []
            for group in kv_cache_groups:
                shared_by.extend(
                    group.layer_names[layer_index]
                    for layer_index in indices
                    if layer_index < len(group.layer_names)
                )
            if shared_by:
                kv_cache_tensors.append(KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by))

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """
    Generates the KV cache configurations for a model.
    Since we use a shared centralized controller for all workers, we need the
    `kv_cache_config` to be consistent across all workers to make sure
    the KV cache allocation can be applied to all workers. However, different
    workers may have different memory available, and different type of layers
    (when pipeline parallel is enabled). To handle the difference between
    workers, the current implementation is:
    1. Merge the KV cache specs of all workers to get the KVCacheSpecs for
       the whole model.
    2. Generate the KV cache groups based on the layer ratio of the whole model.
       This also handles spec unification for hybrid models.
    3. Handle auto-fit max_model_len and memory checks using per-worker
       projected groups to account for PP sharding.
    4. Generate the KV cache configs for each worker based on the KV cache
       grouping strategy. (This is reasonable because the layer ratio of
       different PP stages are similar.)
    5. Change the num_blocks of each worker to the smallest among all workers
       and shrink tensor sizes proportionally to avoid allocating unused memory.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_specs: List of dict[layer_name, KVCacheSpec] for each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.

    Returns:
        The generated KVCacheConfigs for each worker.
    """

    # Merge the KV cache specs of all workers. Different PP stages may have
    # different layer names, and different TP ranks of the same PP stage should
    # have the same KV cache spec.
    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_name] = layer_spec
            else:
                assert merged_kv_cache_specs[layer_name] == layer_spec, (
                    "The KV cache specs for the same layer are different across workers. This is not supported yet."
                )

    # Get global KV cache groups. This also handles spec unification for
    # hybrid models when disable_hybrid_kv_cache_manager is enabled.
    # After this call, merged_kv_cache_specs may be modified in-place.
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    # If original_max_model_len was -1, automatically
    # determine the maximum model length that fits in available GPU memory.
    # We use per-worker projected groups to account for PP sharding.
    projected_groups_per_worker = [
        _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec) for worker_spec in kv_cache_specs
    ]

    if vllm_config.model_config.original_max_model_len == -1:
        _auto_fit_max_model_len(vllm_config, projected_groups_per_worker, available_memory)

    # Check if the available memory is enough per worker.
    total_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    reuse_layers = get_kv_cache_reuse_layers(total_layers)
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        effective_avail_mem = avail_mem
        if reuse_layers is not None:
            effective_avail_mem = avail_mem * total_layers // reuse_layers
        _check_enough_kv_cache_memory(
            effective_avail_mem,
            partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(_estimate_max_model_len_from_groups, vllm_config, groups),
        )

    kv_cache_configs: list[KVCacheConfig] = []
    for projected_groups, kv_cache_spec_one_worker, available_memory_one_worker in zip(
        projected_groups_per_worker, kv_cache_specs, available_memory
    ):
        assert sum(len(group.layer_names) for group in projected_groups) == len(kv_cache_spec_one_worker), (
            "Some layers are not assigned to any group."
        )
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(vllm_config, projected_groups, available_memory_one_worker)
        )

    # Change the num_blocks of each rank to the smallest among all ranks.
    # We also need to shrink the tensor size proportionally to avoid
    # allocating unused memory.
    min_num_blocks = min(kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs)
    for kv_cache_config in kv_cache_configs:
        num_blocks_old = kv_cache_config.num_blocks
        kv_cache_config.num_blocks = min_num_blocks

        # Shrink tensor size proportionally
        for tensor in kv_cache_config.kv_cache_tensors:
            assert tensor.size % num_blocks_old == 0
            tensor.size = tensor.size // num_blocks_old * min_num_blocks

        if len(kv_cache_config.kv_cache_groups) > 0:
            _report_kv_cache_config(vllm_config, kv_cache_config)

    return kv_cache_configs


vllm.v1.core.kv_cache_utils.get_kv_cache_configs = get_kv_cache_configs
vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups = get_kv_cache_config_from_groups

_orig_resolve_kv_cache_block_sizes = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes


def _ascend_resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Ascend-compatible resolve_kv_cache_block_sizes.

    vLLM PR #40860 added a restriction that hybrid KV cache groups with
    multiple block sizes do not support context parallelism (dcp/pcp > 1).
    This restriction is correct for CUDA but not for Ascend, which implements
    context parallelism for MLA and SWA-MLA layers independently.

    For multiple KV cache groups with CP, compute scheduler_block_size as
    lcm(group_block_sizes) * dcp * pcp to maintain alignment, consistent
    with the pre-PR-#40860 behavior of block_size * dcp * pcp.
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    pcp = vllm_config.parallel_config.prefill_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    if len(groups) <= 1:
        bs = cache_config.block_size * dcp * pcp
        return bs, bs

    if dcp != 1 or pcp != 1:
        # Ascend supports CP with multiple KV cache groups; compute
        # scheduler_block_size using the LCM of all group block sizes
        # multiplied by the CP factors for proper alignment.
        group_block_sizes = [g.kv_cache_spec.block_size for g in groups]
        scheduler_block_size = math.lcm(*group_block_sizes) * dcp * pcp
        return scheduler_block_size, scheduler_block_size

    return _orig_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
    """
    if not any(isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()):
        return None

    ratio_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    grouped_swa_mla_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[spec.block_size][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            ratio_specs[spec.compress_ratio][name] = spec

    mla_uniform_specs = []
    for ratio in sorted(ratio_specs, key=lambda r: (r != 4, r)):
        spec_dict = ratio_specs[ratio]
        assert len(spec_dict) > 0
        mla_uniform_specs.append(UniformTypeKVCacheSpecs.from_specs(spec_dict))
    assert mla_uniform_specs is not None

    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    return [*mla_uniform_specs, *swa_uniform_specs]


def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    """
    Generate the KV cache groups from the grouped specs.
    """
    assert len(grouped_specs) > 0 and all(isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs)
    # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
    # containing only MLAAttentionSpec.
    full_mla_spec = grouped_specs[0]
    full_mla_c128_spec = grouped_specs[1]

    assert all(isinstance(spec, MLAAttentionSpec) for spec in full_mla_spec.kv_cache_specs.values())
    full_mla_group = KVCacheGroupSpec(
        layer_names=list(full_mla_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_spec,
    )
    full_mla_c128_group = KVCacheGroupSpec(
        layer_names=list(full_mla_c128_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_c128_spec,
    )

    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.
    # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
    # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
    # The other uniform KV cache specs will be similarly partitioned into layer tuples.
    # Say we have 21 SWA layers, all with the same page size, then we will have "21"
    # layer tuples.
    num_layer_tuples_per_group: list[int] = [g_spec.get_num_layer_tuples() for g_spec in grouped_specs]
    # Choose `num_layer_tuples` to minimize total padding across groups.
    num_layer_tuples = _approximate_gcd(num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0])
    # Round up to the nearest multiple of `num_layer_tuples` (i.e., padding)
    num_layer_tuples_per_group = [round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group]

    # TODO(cmq): this is not general enough
    swa_mla_specs = grouped_specs[2:]

    assert all(
        isinstance(spec, SlidingWindowMLASpec) for group in swa_mla_specs for spec in group.kv_cache_specs.values()
    )

    # Split each SWA UniformKV group into smaller groups to align their #(layer tuples)
    # Possibly padding layer tuples for this.
    # Additionally, we also pad KV blocks in each SWA layer, to align the page size
    # with the corresponding layer in the full-MLA group.
    all_page_sizes = full_mla_spec.get_page_sizes()
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        sm_page_sizes = sm_spec.get_page_sizes()
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        assert max(sm_page_sizes) <= max(all_page_sizes)

        # Unify page size by padding layers' page_size to the nearest larger page_size.
        # Compute candidate (nearest larger page_size) for each unique page size.
        size_to_candidate: dict[int, int] = {}
        for ps in sm_page_sizes:
            size_to_candidate[ps] = min(x for x in all_page_sizes if x >= ps)
        # Pad and collect layer names per page size.
        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            current_size = layer_spec.page_size_bytes
            candidate = size_to_candidate[current_size]
            if current_size < candidate:
                object.__setattr__(layer_spec, "page_size_padded", candidate)
            layers_per_size[candidate].append(layer_name)
        # NOTE(yifan): for now, inside a UniformKV group, each page_size should
        # have the same number of layers. This also means we don't need to pad layers
        # inside a partial-full layer tuple.
        assert len(set(len(layers) for layers in layers_per_size.values())) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))

        # Split layers inside each UniformKV group for aligned #(layers).
        # See `_get_kv_cache_groups_uniform_page_size` for more details.
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            # Flatten tuples and build dict for from_specs
            group_layer_names = [name for layer_tuple in group_layer_tuples for name in layer_tuple]
            group_layer_specs = {name: sm_spec.kv_cache_specs[name] for name in group_layer_names}
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert sub_sm_spec is not None
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_sm_spec,
                )
            )

    return [full_mla_group, full_mla_c128_group, *swa_mla_groups]


def _get_kv_cache_config_deepseek_v4(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    """DeepseekV4 KV cache tensor layout planning.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream (see _get_kv_cache_groups_uniform_groups) so
    every layer's page_size matches one of the full-MLA bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each
    layer at tuple_idx = position-within-bucket. Emit one KVCacheTensor
    per (tuple_idx, bucket) whose shared_by is the union of per-group
    layers at that slot.
    """
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    page_sizes = sorted(full_mla_spec.get_page_sizes())
    layer_tuple_page_bytes = sum(page_sizes)

    # Pre-bucket each group's layers by page_size (registration order within
    # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
    mtp_layer_names = []
    mtp_page_size = 0
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        specs = group.kv_cache_spec.kv_cache_specs
        b: dict[int, list[str]] = defaultdict(list)
        for name in group.layer_names:
            if "mtp" not in name:
                b[specs[name].page_size_bytes].append(name)
            else:
                mtp_layer_names.append(name)
                mtp_page_size = specs[name].page_size_bytes
        bucketed.append(b)

    # num_layer_tuples = longest bucket list across all groups. For the
    # full-MLA group this equals the count of layers in the largest
    # per-page-size bucket (= get_num_layer_tuples()); for SWA sub-groups
    # this equals the sub-group size (each has a single page_size).
    num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values()) + len(mtp_layer_names)

    num_blocks = available_memory // (layer_tuple_page_bytes * num_layer_tuples)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    kv_cache_tensors: list[KVCacheTensor] = []
    for tuple_idx in range(num_layer_tuples - len(mtp_layer_names)):
        for ps in page_sizes:
            shared_by: list[str] = []
            for b in bucketed:
                bucket = b.get(ps)
                if bucket is not None and tuple_idx < len(bucket):
                    shared_by.append(bucket[tuple_idx])
            kv_cache_tensors.append(KVCacheTensor(size=ps * num_blocks, shared_by=shared_by))
    for i in range(len(mtp_layer_names)):
        kv_cache_tensors.append(KVCacheTensor(size=mtp_page_size * num_blocks, shared_by=[mtp_layer_names[i]]))

    return num_blocks, kv_cache_tensors


vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
vllm.v1.core.kv_cache_utils.group_and_unify_kv_cache_specs = group_and_unify_kv_cache_specs
vllm.v1.core.kv_cache_utils._get_kv_cache_config_deepseek_v4 = _get_kv_cache_config_deepseek_v4
vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_groups = _get_kv_cache_groups_uniform_groups

# Also patch the reference used by engine/core.py which imports the function directly.
vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
