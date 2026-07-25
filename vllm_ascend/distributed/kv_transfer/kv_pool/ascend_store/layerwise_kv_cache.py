from __future__ import annotations

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_config import (
    get_gva_layerwise_config,
    get_layerwise_storage_indices,
)


def apply_layerwise_kv_cache_plan(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> None:
    """Rewrite logical layer tensors into shared physical KV cache slots."""
    extra_config = get_gva_layerwise_config(vllm_config.kv_transfer_config)
    if extra_config is None:
        return

    base_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    storage_indices = get_layerwise_storage_indices(base_layers, extra_config)
    if len(storage_indices) >= base_layers:
        return

    if len(kv_cache_config.kv_cache_groups) != 1:
        raise NotImplementedError("Layerwise KV cache reuse requires one KV cache group.")

    old_tensors = kv_cache_config.kv_cache_tensors
    if len(old_tensors) <= 1:
        return
    if any(len(tensor.shared_by) != 1 for tensor in old_tensors):
        raise NotImplementedError(
            "Layerwise KV cache reuse requires one KV cache tensor descriptor per layer."
        )
    if len(old_tensors) != base_layers:
        raise NotImplementedError(
            "Layerwise KV cache reuse currently supports base transformer layers only."
        )

    layer_names = [tensor.shared_by[0] for tensor in old_tensors]
    new_tensors = []
    for slot in storage_indices:
        slot_sizes = {old_tensors[index].size for index in slot}
        if len(slot_sizes) != 1:
            raise ValueError("Layers sharing a layerwise KV buffer must have equal tensor sizes.")
        new_tensors.append(
            KVCacheTensor(
                shared_by=[layer_names[index] for index in slot],
                size=old_tensors[slot[0]].size,
            )
        )
    kv_cache_config.kv_cache_tensors = new_tensors
    logger.info(
        "Layerwise KV cache reuse merged %d tensor descriptors into %d shared slots.",
        len(old_tensors),
        len(new_tensors),
    )
