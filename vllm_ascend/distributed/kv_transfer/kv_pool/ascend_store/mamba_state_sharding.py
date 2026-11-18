# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.utils import get_model_cls
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec, UniformTypeKVCacheSpecs

from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.core.kv_cache_interface import MambaStateSharding


def resolve_mamba_state_sharding(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig | None,
    tp_rank: int = 0,
    pcp_rank: int = 0,
) -> dict[int, MambaStateSharding]:
    """Resolve model-declared state partitions for complete Mamba groups."""
    if kv_cache_config is None or KVPPConfig.from_vllm_config(vllm_config).size > 1:
        return {}
    model_cls = get_model_cls(vllm_config.model_config)
    get_sharding = getattr(model_cls, "get_mamba_state_sharding", None)
    if not callable(get_sharding):
        return {}
    declarations = get_sharding(vllm_config.parallel_config, tp_rank, pcp_rank)
    by_group: dict[int, MambaStateSharding] = {}
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        shardings: set[MambaStateSharding | None] = set()
        for name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            if isinstance(spec, MambaSpec):
                shardings.add(declarations.get(spec.mamba_type))
        if len(shardings) > 1:
            raise ValueError(f"Mamba cache group {group_id} has inconsistent state sharding")
        sharding = next(iter(shardings), None)
        if sharding is not None:
            by_group[group_id] = sharding
    return by_group
