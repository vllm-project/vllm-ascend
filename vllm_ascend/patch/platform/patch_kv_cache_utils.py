from functools import partial

import vllm
from vllm_ascend import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.hashing import sha256_cbor, xxhash_cbor
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import format_gib
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.core.kv_cache_utils import (
    may_override_num_blocks,
    get_kv_cache_groups,
    _project_kv_cache_groups_to_worker,
    _auto_fit_max_model_len,
    _check_enough_kv_cache_memory,
    _max_memory_usage_bytes_from_groups,
    _estimate_max_model_len_from_groups,
    get_uniform_page_size
)
from vllm.v1.request import Request
from vllm.v1.utils import tensor_data

def get_kv_cache_num_layers(vllm_config: VllmConfig) -> int | None:
    """Get the effective number of layers for KV cache allocation.

    When KV cache reuse is enabled (via CacheConfig.kv_cache_reuse_layers
    or the VLLM_KV_CACHE_REUSE_LAYERS env var), only this many layers
    need unique KV cache storage. Returns None if reuse is not enabled.
    """
    # reuse_layers = vllm_config.cache_config.kv_cache_reuse_layers
    # if reuse_layers is None:
    reuse_layers = envs.VLLM_KV_CACHE_REUSE_LAYERS
    return reuse_layers


def get_num_blocks(
    vllm_config: VllmConfig, num_layers: int, available_memory: int, page_size: int
) -> int:
    """
    Get the number of kv cache blocks.

    Args:
        vllm_config: The global VllmConfig
        num_layers: The number of layers
        available_memory: Memory available for KV cache in bytes.
        page_size: The page size of the KV cache.
    """
    reuse_layers = get_kv_cache_num_layers(vllm_config)
    effective_layers = reuse_layers if reuse_layers is not None else num_layers
    num_blocks = int(available_memory // page_size // effective_layers)
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
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # Special case: all layers have the same type of KV cache but with
        # different hidden size. Allocate different amount of memory for each
        # layer based on its hidden size.
        num_blocks = (
            available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
        )
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        kv_cache_tensors = [
            KVCacheTensor(
                size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                shared_by=[layer_name],
            )
            for layer_name in kv_cache_groups[0].layer_names
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

        page_size = get_uniform_page_size(
            [group.kv_cache_spec for group in kv_cache_groups]
        )
        assert group_size > 0, "group_size must be greater than 0"
        num_blocks = get_num_blocks(
            vllm_config, group_size, available_memory, page_size
        )
        kv_cache_tensors = []
        for i in range(group_size):
            shared_by = []
            for j in range(len(kv_cache_groups)):
                if i < len(kv_cache_groups[j].layer_names):
                    shared_by.append(kv_cache_groups[j].layer_names[i])
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            )

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
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet."
                )

    # Get global KV cache groups. This also handles spec unification for
    # hybrid models when disable_hybrid_kv_cache_manager is enabled.
    # After this call, merged_kv_cache_specs may be modified in-place.
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    # If original_max_model_len was -1, automatically
    # determine the maximum model length that fits in available GPU memory.
    # We use per-worker projected groups to account for PP sharding.
    projected_groups_per_worker = [
        _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec)
        for worker_spec in kv_cache_specs
    ]

    if vllm_config.model_config.original_max_model_len == -1:
        _auto_fit_max_model_len(
            vllm_config, projected_groups_per_worker, available_memory
        )

    # Check if the available memory is enough per worker.
    reuse_layers = get_kv_cache_num_layers(vllm_config)
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        effective_avail_mem = avail_mem
        if reuse_layers is not None:
            total_layers = vllm_config.model_config.get_num_layers(
                vllm_config.parallel_config
            )
            effective_avail_mem = (
                avail_mem // reuse_layers * total_layers
            )
        _check_enough_kv_cache_memory(
            effective_avail_mem,
            partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(_estimate_max_model_len_from_groups, vllm_config, groups),
        )

vllm.v1.core.kv_cache_utils.get_num_blocks = get_num_blocks
vllm.v1.core.kv_cache_utils.get_kv_cache_configs = get_kv_cache_configs
vllm.v1.core.kv_cache_utils.get_kv_cache_num_layers = get_kv_cache_num_layers
vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups = get_kv_cache_config_from_groups
