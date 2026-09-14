# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionLayer
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

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

    def get_num_blocks(self, available_bytes: int) -> int:
        persistent_bytes = 0
        scratch_bytes = 0
        for name, bundle in self.layer_bundles.items():
            _, size = build_kvpp_layer_layout(bundle, self.tensor_sizes, num_blocks=1)
            owner = self.layer_owner_ranks.get(name)
            if owner is None or owner == self.kvpp_rank:
                persistent_bytes += size
            if owner is not None:
                scratch_bytes = max(scratch_bytes, size)
        bytes_per_block = persistent_bytes + KVPP_SCRATCH_BUFFER_COUNT * scratch_bytes
        return available_bytes // bytes_per_block if bytes_per_block else 0


def build_layer_cache_bundles(cache_spec: dict[str, KVCacheSpec]) -> dict[str, tuple[str, ...]]:
    by_index: dict[int, list[str]] = defaultdict(list)
    for name in sorted(
        cache_spec,
        key=lambda name: (extract_layer_index(name), isinstance(cache_spec[name], AscendSFAIndexerCacheSpec), name),
    ):
        by_index[extract_layer_index(name)].append(name)
    return {names[0]: tuple(names) for names in by_index.values()}


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


def find_mtp_layers(
    vllm_config: VllmConfig,
    local_layer_names: Iterable[str],
) -> set[str]:
    """Find MTP KV-cache layers among this worker's PP-local cache names.

    Only names present in ``local_layer_names`` are returned. A PP stage
    without MTP caches yields an empty set; KVPP does not assume MTP lives
    on the last pipeline rank.
    """
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.method != "mtp":
        return set()

    hf_config = vllm_config.model_config.hf_config
    mtp_start = hf_config.num_hidden_layers
    num_mtp_layers = hf_config.num_nextn_predict_layers
    mtp_end = mtp_start + num_mtp_layers
    return {layer_name for layer_name in local_layer_names if mtp_start <= extract_layer_index(layer_name) < mtp_end}


def map_kvpp_layers_to_owners(vllm_config: VllmConfig, local_layer_names: Iterable[str]) -> dict[str, int]:
    """Partition PP-local Target KV layers across KVPP ranks.

    ``local_layer_names`` must already be PP-local (typically the keys of the
    current worker's cache spec). MTP layers remain fully allocated on every
    KVPP rank and are therefore absent from the returned owner mapping.
    """
    kvpp_size = KVPPConfig.from_vllm_config(vllm_config).size
    # Workers are separate Python processes and may receive layer names from
    # sets or differently ordered dictionaries. Keep both owner insertion
    # order and per-layer cache-bundle order identical on every rank.
    local_layer_names = tuple(sorted(local_layer_names, key=lambda name: (extract_layer_index(name), name)))
    mtp_layers = find_mtp_layers(
        vllm_config,
        local_layer_names,
    )
    layers_by_index: dict[int, list[str]] = defaultdict(list)
    for layer_name in local_layer_names:
        if layer_name not in mtp_layers:
            layers_by_index[extract_layer_index(layer_name)].append(layer_name)

    layer_indices = sorted(layers_by_index)
    base, remainder = divmod(len(layer_indices), kvpp_size)
    layer_owner_ranks: dict[str, int] = {}
    offset = 0
    for owner_rank in range(kvpp_size):
        partition_size = base + int(owner_rank < remainder)
        for layer_index in layer_indices[offset : offset + partition_size]:
            for layer_name in layers_by_index[layer_index]:
                layer_owner_ranks[layer_name] = owner_rank
        offset += partition_size
    return layer_owner_ranks


def create_kvpp_cache_allocation_plan(
    vllm_config: VllmConfig,
    worker_spec: dict[str, KVCacheSpec],
    kvpp_rank: int,
) -> KVPPPhysicalCachePlan:
    """Keep upstream's logical group while budgeting actual allocations."""
    logical_spec = dict(worker_spec)
    if (
        any(not isinstance(spec, FullAttentionSpec) for spec in logical_spec.values())
        or len({spec.block_size for spec in logical_spec.values()}) > 1
    ):
        raise ValueError("KVPP requires one full-attention cache group with a common block size.")
    return KVPPPhysicalCachePlan(
        logical_cache_spec=logical_spec,
        layer_owner_ranks=map_kvpp_layers_to_owners(vllm_config, logical_spec),
        layer_bundles=build_layer_cache_bundles(logical_spec),
        tensor_sizes=build_kvpp_buffer_sizes(vllm_config, logical_spec),
        kvpp_rank=kvpp_rank,
    )
