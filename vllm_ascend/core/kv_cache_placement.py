# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.ascend_config import KVPPConfig


@dataclass(frozen=True)
class KVPPPhysicalCachePlan:
    """Worker-local physical allocation with a complete logical cache view."""

    logical_cache_spec: dict[str, KVCacheSpec]
    physical_cache_spec: dict[str, KVCacheSpec]
    scratch_layer_aliases: dict[str, list[str]]

    def restore_logical_cache_view(self, kv_cache_config: KVCacheConfig) -> None:
        """Restore logical layer bindings on an upstream physical config."""
        for tensor in kv_cache_config.kv_cache_tensors:
            tensor_names: list[str] = []
            for layer_name in tensor.shared_by:
                tensor_names.extend(self.scratch_layer_aliases.get(layer_name, [layer_name]))
            tensor.shared_by = list(dict.fromkeys(tensor_names))

        logical_groups: list[KVCacheGroupSpec] = []
        for group in kv_cache_config.kv_cache_groups:
            group_names: list[str] = []
            for layer_name in group.layer_names:
                group_names.extend(self.scratch_layer_aliases.get(layer_name, [layer_name]))
            expanded_names = list(dict.fromkeys(group_names))

            group_spec = group.kv_cache_spec
            if expanded_names and isinstance(group_spec, UniformTypeKVCacheSpecs):
                group_spec = UniformTypeKVCacheSpecs(
                    block_size=group_spec.block_size,
                    kv_cache_specs={layer_name: self.logical_cache_spec[layer_name] for layer_name in expanded_names},
                )
            logical_groups.append(
                KVCacheGroupSpec(
                    expanded_names,
                    group_spec,
                    is_eagle_group=group.is_eagle_group,
                )
            )

        kv_cache_config.kv_cache_groups = logical_groups


def project_kv_cache_groups_to_worker(
    global_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Project global logical groups onto one worker's PP-local layers.

    The projected list keeps one entry per global group so logical group
    indices stay aligned with the global topology; groups without local
    layers become empty entries instead of being compacted away.
    """
    projected_groups: list[KVCacheGroupSpec] = []
    for group in global_groups:
        worker_layer_names = [layer_name for layer_name in group.layer_names if layer_name in worker_spec]
        group_spec = group.kv_cache_spec
        if worker_layer_names and isinstance(group_spec, UniformTypeKVCacheSpecs):
            group_spec = UniformTypeKVCacheSpecs(
                block_size=group_spec.block_size,
                kv_cache_specs={layer_name: group_spec.kv_cache_specs[layer_name] for layer_name in worker_layer_names},
            )
        projected_groups.append(
            KVCacheGroupSpec(
                worker_layer_names,
                group_spec,
                is_eagle_group=group.is_eagle_group and bool(worker_layer_names),
            )
        )
    return projected_groups


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
    if len(layer_indices) < kvpp_size:
        raise ValueError(
            f"KVPP size ({kvpp_size}) exceeds the number of KV cache layer bundles ({len(layer_indices)})."
        )
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


def build_cache_allocation_groups(
    logical_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
    layer_owner_ranks: dict[str, int],
    kvpp_rank: int,
) -> tuple[list[KVCacheGroupSpec], dict[str, list[str]]]:
    """Per-KVPP-rank allocation view over PP-local logical groups.

    Target layers owned by this rank stay persistent; other owners' layers map
    onto two alternating scratch caches per layout. MTP layers remain
    unpartitioned by KVPP and are allocated in full on every KVPP rank.
    """
    allocation_spec: dict[str, KVCacheSpec] = {}
    scratch_layer_aliases: dict[str, list[str]] = {}

    for group in logical_groups:
        local_names = list(group.layer_names)
        managed_names = [name for name in local_names if name in layer_owner_ranks]
        allocation_names = [
            name for name in local_names if name not in layer_owner_ranks or layer_owner_ranks[name] == kvpp_rank
        ]
        scratch_layout_groups: list[list[str]] = []
        for name in managed_names:
            if layer_owner_ranks[name] == kvpp_rank:
                continue
            for layout_names in scratch_layout_groups:
                if worker_spec[name] == worker_spec[layout_names[0]]:
                    layout_names.append(name)
                    break
            else:
                scratch_layout_groups.append([name])
        for layout_names in scratch_layout_groups:
            scratch_names = layout_names[:2]
            allocation_names.extend(scratch_names)
            for scratch_index, scratch_name in enumerate(scratch_names):
                scratch_layer_aliases[scratch_name] = layout_names[scratch_index :: len(scratch_names)]
        for layer_name in allocation_names:
            allocation_spec[layer_name] = worker_spec[layer_name]

    return (
        project_kv_cache_groups_to_worker(logical_groups, allocation_spec),
        scratch_layer_aliases,
    )


def create_kvpp_cache_allocation_plan(
    vllm_config: VllmConfig,
    worker_spec: dict[str, KVCacheSpec],
    kvpp_rank: int,
) -> KVPPPhysicalCachePlan:
    """Create the KV cache allocation plan reported by one worker.

    The union of owner specs across KVPP ranks still contains every logical
    layer, so upstream can build the global logical topology normally. Each
    worker only reports its persistent layers, two scratch layers per cache
    layout, and a full MTP cache on each rank; upstream therefore computes
    the desired physical allocation without any KVPP-specific hook.
    """
    kvpp_size = KVPPConfig.from_vllm_config(vllm_config).size
    if not 0 <= kvpp_rank < kvpp_size:
        raise ValueError(f"KVPP rank must be in [0, {kvpp_size}), got {kvpp_rank}.")

    logical_cache_spec = dict(worker_spec)
    # get_kv_cache_groups may normalize its input in place, so keep the spec
    # returned to the engine and the spec retained for logical restoration
    # independent from its working copy.
    logical_cache_groups = get_kv_cache_groups(vllm_config, dict(logical_cache_spec))
    layer_owner_ranks = map_kvpp_layers_to_owners(vllm_config, worker_spec)
    physical_cache_groups, scratch_layer_aliases = build_cache_allocation_groups(
        logical_cache_groups,
        worker_spec,
        layer_owner_ranks,
        kvpp_rank,
    )
    physical_layer_names = {layer_name for group in physical_cache_groups for layer_name in group.layer_names}
    physical_cache_spec = {
        layer_name: logical_cache_spec[layer_name]
        for layer_name in logical_cache_spec
        if layer_name in physical_layer_names
    }
    return KVPPPhysicalCachePlan(
        logical_cache_spec=logical_cache_spec,
        physical_cache_spec=physical_cache_spec,
        scratch_layer_aliases=scratch_layer_aliases,
    )
