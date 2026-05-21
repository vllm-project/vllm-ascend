from functools import partial

import vllm
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import (
    _auto_fit_max_model_len,
    _check_enough_kv_cache_memory,
    _estimate_max_model_len_from_groups,
    _max_memory_usage_bytes_from_groups,
    _project_kv_cache_groups_to_worker,
    _report_kv_cache_config,
    get_kv_cache_groups,
    get_uniform_page_size,
    may_override_num_blocks,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_config import (
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


def get_num_blocks(
    vllm_config: VllmConfig,
    num_layers: int,
    available_memory: int,
    page_size: int,
) -> int:
    """
    Get the number of kv cache blocks.

    Args:
        vllm_config: The global VllmConfig
        num_layers: The number of layers
        available_memory: Memory available for KV cache in bytes.
        page_size: The page size of the KV cache.
    """
    num_blocks = int(available_memory // page_size // num_layers)
    num_blocks = max(num_blocks, 0)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)
    return num_blocks


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
            storage_slots = [[] for _ in range(reuse_layers)]
            for i, layer_name in enumerate(group.layer_names):
                storage_slots[i % reuse_layers].append(layer_name)

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
        storage_group_size = reuse_layers if reuse_layers is not None else group_size
        num_blocks = get_num_blocks(vllm_config, storage_group_size, available_memory, page_size)
        kv_cache_tensors = []
        for i in range(storage_group_size):
            shared_by = []
            for group in kv_cache_groups:
                shared_by.extend(
                    layer_name
                    for layer_index, layer_name in enumerate(group.layer_names)
                    if layer_index % storage_group_size == i
                )
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
