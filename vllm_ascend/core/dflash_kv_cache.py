# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import fields
from typing import Any

from vllm import envs
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec, KVQuantMode, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendReplicatedDraftCacheSpec


def uses_dflash_layer_layout(vllm_config: VllmConfig) -> bool:
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None or spec.method != "dflash":
        return False
    layout = vllm_config.cache_config.kv_cache_layout or envs.VLLM_KV_CACHE_LAYOUT
    return layout in ("HND", "NHD", "LBHNC", "LBNHC")


def apply_replicated_dflash_specs(
    specs: dict[str, KVCacheSpec], vllm_config: VllmConfig, speculator: Any
) -> dict[str, KVCacheSpec]:
    """Pair full-attention draft pages with compatible DCP target allocations.

    Hybrid targets keep their existing planners and independent draft groups.
    Only the physical capacity changes: neither the target block size nor the
    draft model's DCP=1 execution configuration is modified.
    """
    parallel = vllm_config.parallel_config
    if (
        not uses_dflash_layer_layout(vllm_config)
        or parallel.decode_context_parallel_size <= 1
        or parallel.prefill_context_parallel_size != 1
        or getattr(speculator, "dcp_size", None) != 1
    ):
        return specs
    draft_names = speculator.draft_attn_layer_names
    target = {name: spec for name, spec in specs.items() if name not in draft_names}
    if not target or any(getattr(spec, "model_version", None) == "glm5_next" for spec in target.values()):
        return specs
    uniform = UniformTypeKVCacheSpecs.from_specs(target)
    if uniform is None or not uniform.dcp_sharded:
        return specs
    result = dict(specs)
    for name in draft_names & specs.keys():
        spec = specs[name]
        if type(spec) is not FullAttentionSpec or spec.block_size != uniform.block_size:
            continue
        # Padded/quantized layouts need a separate physical view contract.
        if (
            spec.page_size_padded is not None
            or spec.state_content_bytes is not None
            or spec.kv_quant_mode != KVQuantMode.NONE
        ):
            continue
        values = {field.name: getattr(spec, field.name) for field in fields(spec) if field.init}
        values.update(dcp_sharded=True, replication_size=parallel.decode_context_parallel_size)
        result[name] = AscendReplicatedDraftCacheSpec(**values)
    return result
