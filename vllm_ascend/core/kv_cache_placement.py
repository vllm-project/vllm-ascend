# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, fields

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionLayer
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec, UniformTypeKVCacheSpecs

from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.quantization.utils import enable_fa_quant
from vllm_ascend.utils import calc_split_factor, enable_sfa

# One buffer for the current layer and one for the next layer's prefetch.
KVPP_SCRATCH_BUFFER_COUNT = 2


@dataclass(frozen=True)
class KVPPPhysicalCachePlan:
    """Complete logical topology and worker-local physical memory cost."""

    logical_cache_spec: dict[str, KVCacheSpec]
    layer_owner_ranks: dict[str, int]
    layer_bundles: dict[str, tuple[str, ...]]
    tensor_sizes: dict[str, tuple[int, ...]]
    kvpp_rank: int

    def is_persistent(self, layer_name: str) -> bool:
        """Draft caches and locally owned target caches have persistent storage."""
        return self.layer_owner_ranks.get(layer_name, self.kvpp_rank) == self.kvpp_rank

    def get_num_blocks(self, available_bytes: int) -> int:
        persistent_bytes = 0
        scratch_bytes = 0
        for name, bundle in self.layer_bundles.items():
            _, size = build_kvpp_layer_layout(bundle, self.tensor_sizes, num_blocks=1)
            owner = self.layer_owner_ranks.get(name)
            if self.is_persistent(name):
                persistent_bytes += size
            if owner is not None:
                scratch_bytes = max(scratch_bytes, size)
        bytes_per_block = persistent_bytes + KVPP_SCRATCH_BUFFER_COUNT * scratch_bytes
        return available_bytes // bytes_per_block if bytes_per_block else 0


def build_layer_cache_bundles(
    cache_spec: dict[str, KVCacheSpec], draft_layers: set[str] | None = None
) -> dict[str, tuple[str, ...]]:
    draft_layers = draft_layers or set()
    by_index: dict[int, list[str]] = defaultdict(list)
    for name in sorted(
        (name for name in cache_spec if name not in draft_layers),
        key=lambda name: (extract_layer_index(name), isinstance(cache_spec[name], AscendSFAIndexerCacheSpec), name),
    ):
        by_index[extract_layer_index(name)].append(name)
    bundles = {names[0]: tuple(names) for names in by_index.values()}
    # Draft names may reuse target indices or have no numeric index at all.
    bundles.update({name: (name,) for name in sorted(draft_layers)})
    return bundles


def get_kvpp_attention_kv_dims(vllm_config: VllmConfig, layer_name: str, spec: KVCacheSpec) -> tuple[int, int]:
    if isinstance(spec, AscendMLAAttentionSpec):
        layer = get_layers_from_vllm_config(vllm_config, AttentionLayerBase, [layer_name])[layer_name]
        if isinstance(layer, MLAAttention):
            return layer.kv_lora_rank, layer.qk_rope_head_dim
        if isinstance(layer, CacheOnlyAttentionLayer):
            return spec.head_size, spec.head_size
        raise TypeError(f"Unsupported KVPP attention layer: {layer_name} ({type(layer).__name__}).")
    return spec.head_size, spec.head_size_v


def build_kvpp_buffer_sizes(
    vllm_config: VllmConfig, logical_spec: dict[str, KVCacheSpec]
) -> dict[str, tuple[int, ...]]:
    result = {}
    for name, spec in logical_spec.items():
        sizes = []
        if isinstance(spec, AscendSFAIndexerCacheSpec):
            elements = spec.sfa_dcp_replicated_indexer_size * spec.block_size * spec.num_kv_heads
            sizes.append(elements * spec.head_size * get_dtype_size(spec.dtype))
            if spec.scale_dim:
                sizes.append(elements * spec.scale_dim * get_dtype_size(spec.scale_dtype))
        elif isinstance(spec, AscendMLAAttentionSpec) and spec.cache_sparse_sfa_c8:
            sizes.append(spec.page_size_bytes)
        else:
            dims = list(get_kvpp_attention_kv_dims(vllm_config, name, spec))
            if not enable_sfa(vllm_config) and enable_fa_quant(vllm_config):
                factors = vllm_config.quant_config.get_kv_quant_split_factor(name, dims)
            else:
                factors = calc_split_factor(dims)
            sizes.extend(int(spec.page_size_bytes // factor) for factor in factors)
        result[name] = tuple(sizes)
    return result


def build_kvpp_layer_layout(
    cache_names: tuple[str, ...], tensor_sizes: dict[str, tuple[int, ...]], num_blocks: int
) -> tuple[dict[str, tuple[tuple[int, int], ...]], int]:
    cursor = 0
    layout = {}
    for name in cache_names:
        parts = []
        for size_per_block in tensor_sizes[name]:
            size = num_blocks * size_per_block
            parts.append((cursor, size))
            cursor += size
        layout[name] = tuple(parts)
    return layout, cursor


def map_kvpp_layers_to_owners(kvpp_size: int, target_bundles: dict[str, tuple[str, ...]]) -> dict[str, int]:
    """Partition target bundles in their deterministic execution order."""
    bundles = list(target_bundles.values())
    base, remainder = divmod(len(bundles), kvpp_size)
    owners: dict[str, int] = {}
    offset = 0
    for rank in range(kvpp_size):
        count = base + int(rank < remainder)
        for bundle in bundles[offset : offset + count]:
            owners.update((name, rank) for name in bundle)
        offset += count
    return owners


def create_kvpp_cache_allocation_plan(
    vllm_config: VllmConfig,
    worker_spec: dict[str, KVCacheSpec],
    kvpp_rank: int,
    *,
    draft_layer_names: Iterable[str],
) -> KVPPPhysicalCachePlan:
    """Keep upstream's logical group while budgeting actual allocations."""
    logical_spec = dict(worker_spec)
    if (
        any(not isinstance(spec, FullAttentionSpec) for spec in logical_spec.values())
        or len({spec.block_size for spec in logical_spec.values()}) > 1
    ):
        raise ValueError("KVPP requires one full-attention cache group with a common block size.")
    draft_layers = set(draft_layer_names).intersection(logical_spec)
    bundles = build_layer_cache_bundles(logical_spec, draft_layers)
    target_bundles = {name: bundle for name, bundle in bundles.items() if name not in draft_layers}
    return KVPPPhysicalCachePlan(
        logical_cache_spec=logical_spec,
        layer_owner_ranks=map_kvpp_layers_to_owners(KVPPConfig.from_vllm_config(vllm_config).size, target_bundles),
        layer_bundles=bundles,
        tensor_sizes=build_kvpp_buffer_sizes(vllm_config, logical_spec),
        kvpp_rank=kvpp_rank,
    )


def get_kvpp_cache_specs(kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
    specs: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        for name in group.layer_names:
            specs[name] = spec.kv_cache_specs[name] if isinstance(spec, UniformTypeKVCacheSpecs) else spec
    return specs


@dataclass
class KVPPCacheConfig(KVCacheConfig):
    """Worker-local cache configuration carrying the plan used for budgeting.

    Attached after the engine selects the block count, before the runner and
    transfer connector initialize. Runner deep copies preserve the plan.
    """

    kvpp_plan: KVPPPhysicalCachePlan = field(kw_only=True)

    @classmethod
    def from_config(cls, config: KVCacheConfig, plan: KVPPPhysicalCachePlan) -> "KVPPCacheConfig":
        if get_kvpp_cache_specs(config) != plan.logical_cache_spec:
            raise ValueError("KVPP allocation specifications differ from the budgeted plan.")
        return cls(**{f.name: getattr(config, f.name) for f in fields(KVCacheConfig)}, kvpp_plan=plan)


def get_kvpp_cache_plan(config: KVCacheConfig) -> KVPPPhysicalCachePlan:
    """Require the worker's plan; never reconstruct ownership in consumers."""
    if not isinstance(config, KVPPCacheConfig):
        raise ValueError("KVPP cache configuration is missing the worker allocation plan.")
    return config.kvpp_plan
