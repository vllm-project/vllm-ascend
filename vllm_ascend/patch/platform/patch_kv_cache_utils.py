from collections import defaultdict

import vllm
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_groups_uniform_page_size, _get_kv_cache_groups_uniform_spec,
    _get_kv_cache_groups_uniform_type, create_kv_cache_group_specs,
    get_num_blocks, get_uniform_page_size, is_kv_cache_spec_uniform,
    is_kv_cache_type_attention_free, may_override_num_blocks,
    unify_hybrid_kv_cache_specs, unify_kv_cache_spec_page_size)
from vllm.v1.kv_cache_interface import (KVCacheConfig, KVCacheGroupSpec,
                                        KVCacheSpec, KVCacheTensor,
                                        UniformTypeKVCacheSpecs)

from vllm_ascend.patch.platform.patch_kv_cache_coordinator import \
    USE_MULTI_BLOCK_POOL


def _get_kv_cache_groups_uniform_block_size(
    kv_cache_spec: dict[str, KVCacheSpec], ) -> list[KVCacheGroupSpec]:
    '''
    Generates the KV cache groups with same block size,
    and there maybe multiple groups with different spec,
    each group has their own block_pool and each layer
    of each group has their own kv_cache_tensor.

    :param kv_cache_spec: The KVCacheSpecs of all the layers
    :type kv_cache_spec: dict[str, KVCacheSpec]
    :return: a list of KVCacheGroupSpecs, there is one type of KVCacheSpec in each group 
    :rtype: list[KVCacheGroupSpec]
    '''
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    _, first_kv_cache_config = next(iter(kv_cache_spec.items()))
    block_size = first_kv_cache_config.block_size
    for layer_name, layer_spec in kv_cache_spec.items():
        assert block_size == layer_spec.block_size, "Layer block size is not equal."
        same_type_layers[layer_spec].append(layer_name)
    grouped_layers = list(same_type_layers.values())
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)


def check_uniform_page_size(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
    kv_cache_specs = [group.kv_cache_spec for group in kv_cache_groups]
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    return len(page_sizes) == 1


def get_kv_cache_groups(
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec]) -> list[KVCacheGroupSpec]:
    """
    Split the layers in the model into groups with the same KV cache spec.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroups
    """

    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)

    if is_kv_cache_type_attention_free(kv_cache_spec):
        # This returns an empty list to allow for the KVCacheManager to handle
        # attention free models.
        return []

    if is_kv_cache_spec_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        # All layers need the same number of token slots (e.g., all layers are
        # full attention, or all layers are sliding window attention with the
        # same window size). Put all layers into one group.
        return _get_kv_cache_groups_uniform_type(uniform_spec)

    elif USE_MULTI_BLOCK_POOL:
        # kv cache group spec with multi groups and same block size without share hybrid blocks
        return _get_kv_cache_groups_uniform_block_size(kv_cache_spec)

    # As KVCacheManager can only allocate memory of one size, we need to unify
    # the page size of the layers. For cases cannot be unified, this function
    # will raise an error.
    kv_cache_spec = unify_kv_cache_spec_page_size(kv_cache_spec)
    # Model contains multiple attention types, but KV cache of all layers
    # have the same physical memory per block per layer. Split the layers
    # into groups with the same number of layers, and thus same total page
    # size.
    return _get_kv_cache_groups_uniform_page_size(kv_cache_spec)


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
            kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs):
        # Special case: all layers have the same type of KV cache but with
        # different hidden size. Allocate different amount of memory for each
        # layer based on its hidden size.
        num_blocks = (available_memory //
                      kv_cache_groups[0].kv_cache_spec.page_size_bytes)
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        kv_cache_tensors = [
            KVCacheTensor(
                size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                shared_by=[layer_name],
            ) for layer_name in kv_cache_groups[0].layer_names
        ]
    elif check_uniform_page_size(kv_cache_groups) is False:
        # Special case: there are multiple groups of KV cache, and the block
        # size of them keeps the same. We will still have `num_layers` memory
        # pools, this means the memory pools won't be shared across the groups.
        total_page_size_bytes = 0
        for kv_cache_group in kv_cache_groups:
            num_layers = len(kv_cache_group.layer_names)
            page_size = kv_cache_group.kv_cache_spec.page_size_bytes
            total_page_size_bytes += page_size * num_layers
        num_blocks = available_memory // total_page_size_bytes
        # TODO(zxr): DONT use magic number
        num_blocks = num_blocks // 128 * 128
        assert num_blocks > 0
        kv_cache_tensors = []
        for i in range(len(kv_cache_groups)):
            for layer_name in kv_cache_groups[i].layer_names:
                shared_by = [layer_name]
                kv_cache_tensors.append(
                    KVCacheTensor(
                        size=kv_cache_groups[i].kv_cache_spec.page_size_bytes *
                        num_blocks,
                        shared_by=shared_by))
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
            [group.kv_cache_spec for group in kv_cache_groups])
        assert group_size > 0, "group_size must be greater than 0"
        num_blocks = get_num_blocks(vllm_config, group_size, available_memory,
                                    page_size)
        kv_cache_tensors = []
        for i in range(group_size):
            shared_by = []
            for j in range(len(kv_cache_groups)):
                if i < len(kv_cache_groups[j].layer_names):
                    shared_by.append(kv_cache_groups[j].layer_names[i])
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks,
                              shared_by=shared_by))

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


vllm.v1.core.kv_cache_utils.get_kv_cache_groups = get_kv_cache_groups
vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups = get_kv_cache_config_from_groups
