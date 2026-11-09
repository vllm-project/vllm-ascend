# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-Next spec eligibility, buffer generation and recurrent-state constraints."""

from types import MappingProxyType

from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, is_prefix_cacheable
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import LayerwiseCachePlanner
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    LayerwiseCacheConfig,
    get_physical_layer_index,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.reuse_policy import (
    LayerwiseReuseLayout,
    LayerwiseReusePlan,
)
from vllm_ascend.models.glm5next.cache_config import (
    _is_glm5_next_indexer_spec,
    _is_glm5_next_main_spec,
)
from vllm_ascend.utils import get_kv_cache_tensor_layers


def _can_reuse_mla_spec(spec: MLAAttentionSpec) -> bool:
    is_main = _is_glm5_next_main_spec(spec)
    if not is_main and not _is_glm5_next_indexer_spec(spec):
        raise ValueError("Expected a GLM-Next main or compressed indexer spec")
    return is_main


GLM5_NEXT_SPEC_REUSE_RULES = MappingProxyType(
    {MLAAttentionSpec: _can_reuse_mla_spec, MambaSpec: True, AscendIndexerKPoolTailSpec: False}
)


class Glm5NextCachePlanner(LayerwiseCachePlanner):
    def build(self) -> LayerwiseReusePlan:
        base_layers = self.vllm_config.model_config.get_num_layers(self.vllm_config.parallel_config)
        indices = {name: get_physical_layer_index(name, base_layers) for name in self.specs}
        registered_draft_names = set(getattr(self.config, "dspark_draft_layer_names", ()))
        draft_names = {
            name
            for name, spec in self.specs.items()
            if name in registered_draft_names or getattr(spec, "is_draft", False) or indices[name] >= base_layers
        }
        layers = sorted({index for name, index in indices.items() if name not in draft_names})
        settings = LayerwiseCacheConfig.parse(len(layers), self.extra_config)
        independent = [layers[index] for index in settings.independent_layers]
        members = {
            name
            for name, spec in self.specs.items()
            if name not in draft_names
            and name not in self.excluded_layer_names
            and indices[name] not in independent
            and self.can_reuse_spec(spec, GLM5_NEXT_SPEC_REUSE_RULES)
        }
        group_ids = {
            name: index for index, group in enumerate(self.config.kv_cache_groups) for name in group.layer_names
        }
        packs = []
        for tensor in self.config.kv_cache_tensors:
            original = get_kv_cache_tensor_layers(tensor)
            if tensor.host_resident or any(not is_prefix_cacheable(self.specs[name]) for name in original):
                continue
            names = [name for name in original if name in members]
            if tensor.layer_stride or len({group_ids[name] for name in names}) != len(names):
                packs.extend([name] for name in names)
            elif names:
                packs.append(names)
        packs.sort(key=lambda names: min(indices[name] for name in names))
        count = min(settings.num_shared_buffers, len(packs))
        slots = tuple(tuple(name for pack in packs[slot::count] for name in pack) for slot in range(count))
        return LayerwiseReusePlan(
            layer_indices=indices,
            buffer_slots=slots,
            independent_layers=independent,
            num_prefetch_layers=settings.num_prefetch_layers,
            draft_layer_names=frozenset(draft_names),
        )

    def validate(self, layout: LayerwiseReuseLayout) -> None:
        for name in layout.reused_cache_names:
            spec = self.specs[name]
            if not self.can_reuse_spec(spec, GLM5_NEXT_SPEC_REUSE_RULES):
                raise ValueError(f"GLM-Next cache {name} must remain resident")
            if isinstance(spec, MambaSpec) and (
                spec.mamba_cache_mode != "align" or getattr(spec, "num_prefill_checkpoint_blocks", 0) != 0
            ):
                raise NotImplementedError(
                    f"Reusable cache {name} has unsupported state semantics: {type(spec).__name__}"
                )
