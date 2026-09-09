#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Ascend KV cache config builder.

vLLM PR #53558 introduces a pluggable :class:`KVCacheConfigBuilder` resolved as
``platform override > model declaration > default``. Ascend uses it to keep its
DeepSeekV4 KV cache planning (grouping + non-packed shared-tensor layout) without
monkey-patching vLLM internals; this supersedes the DeepSeekV4 patches previously
shipped in ``vllm_ascend.patch.platform.patch_kv_cache_utils``.
"""

from collections import defaultdict
from dataclasses import replace
from functools import partial

import vllm.v1.core.kv_cache_planning as kv_cache_planning
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_config_builder import KVCacheConfigBuilder
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.core.kv_cache_interface import is_deepseek_v4_kv_cache_spec

# The #53558 head rebased onto the September-9 vLLM main renamed two
# kv_cache_planning helpers. Alias them so the DeepSeekV4 builder runs on both
# old and new names; drop the aliases and call the new names once vllm-ascend
# adopts upstream #16009.
if not hasattr(kv_cache_planning, "_may_override_num_blocks"):
    kv_cache_planning._may_override_num_blocks = kv_cache_planning.may_override_num_blocks
if not hasattr(kv_cache_planning, "_annotate_eagle_groups_deepseek_v4"):
    kv_cache_planning._annotate_eagle_groups_deepseek_v4 = (
        lambda vllm_config, kv_cache_spec, kv_cache_groups: kv_cache_planning._annotate_eagle_groups(
            vllm_config, kv_cache_spec, kv_cache_groups, use_deepseek_v4_fallback=True
        )
    )


def _has_deepseek_v4(kv_cache_specs: list[dict[str, KVCacheSpec]]) -> bool:
    """Whether any worker spec contains a DeepSeekV4 (SWA-MLA) layout."""
    return any(is_deepseek_v4_kv_cache_spec(spec) for worker_spec in kv_cache_specs for spec in worker_spec.values())


# ---------------------------------------------------------------------------
# Ascend DeepSeekV4 grouping (ported from patch_kv_cache_utils)
# ---------------------------------------------------------------------------
def _ascend_group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
    """
    if not any(isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()):
        return None

    logical_block_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    grouped_swa_mla_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[spec.block_size][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            logical_block_specs[spec.block_size][name] = spec

    mla_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for block_size in sorted(logical_block_specs):
        spec_dict = logical_block_specs[block_size]
        assert len(spec_dict) > 0
        mla_uniform_specs.append(UniformTypeKVCacheSpecs.from_specs(spec_dict))
    assert mla_uniform_specs

    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    return [*mla_uniform_specs, *swa_uniform_specs]


def _ascend_get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    """
    Generate the KV cache groups from the grouped specs.
    """
    assert len(grouped_specs) > 0 and all(isinstance(s, UniformTypeKVCacheSpecs) for s in grouped_specs)
    # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
    # containing only MLAAttentionSpec.
    full_mla_spec = grouped_specs[0]
    full_mla_c128_spec = grouped_specs[1]

    assert all(isinstance(spec, MLAAttentionSpec) for spec in full_mla_spec.kv_cache_specs.values())
    full_mla_group = KVCacheGroupSpec(
        layer_names=list(full_mla_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_spec,
    )
    full_mla_c128_group = KVCacheGroupSpec(
        layer_names=list(full_mla_c128_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_c128_spec,
    )

    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.
    # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
    # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
    # The other uniform KV cache specs will be similarly partitioned into layer tuples.
    # Say we have 21 SWA layers, all with the same page size, then we will have "21"
    # layer tuples.
    num_layer_tuples_per_group: list[int] = [g_spec.get_max_layers_per_page_size() for g_spec in grouped_specs]
    # Choose `num_layer_tuples` to minimize total padding across groups.
    num_layer_tuples = kv_cache_planning._approximate_gcd(
        num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0]
    )
    # Round up to the nearest multiple of `num_layer_tuples` (i.e., padding)
    num_layer_tuples_per_group = [round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group]

    # TODO(cmq): this is not general enough
    swa_mla_specs = grouped_specs[2:]

    assert all(
        isinstance(spec, SlidingWindowMLASpec) for group in swa_mla_specs for spec in group.kv_cache_specs.values()
    )

    # Split each SWA UniformKV group into smaller groups to align their #(layer tuples)
    # Possibly padding layer tuples for this.
    # Additionally, we also pad KV blocks in each SWA layer, to align the page size
    # with the corresponding layer in the full-MLA group.
    all_page_sizes = _page_sizes(full_mla_spec)
    swa_mla_groups: list[KVCacheGroupSpec] = []
    for sm_spec in swa_mla_specs:
        sm_page_sizes = _page_sizes(sm_spec)
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        assert max(sm_page_sizes) <= max(all_page_sizes)

        # Unify page size by padding layers' page_size to the nearest larger page_size.
        # Compute candidate (nearest larger page_size) for each unique page size.
        size_to_candidate: dict[int, int] = {}
        for ps in sm_page_sizes:
            size_to_candidate[ps] = min(x for x in all_page_sizes if x >= ps)
        # Pad and collect layer names per page size.
        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            current_size = layer_spec.page_size_bytes
            candidate = size_to_candidate[current_size]
            if current_size < candidate:
                object.__setattr__(layer_spec, "page_size_padded", candidate)
            layers_per_size[candidate].append(layer_name)
        # NOTE(yifan): for now, inside a UniformKV group, each page_size should
        # have the same number of layers. This also means we don't need to pad layers
        # inside a partial-full layer tuple.
        assert len(set(len(layers) for layers in layers_per_size.values())) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))

        # Split layers inside each UniformKV group for aligned #(layers).
        # See `_get_kv_cache_groups_uniform_page_size` for more details.
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            # Flatten tuples and build dict for from_specs
            group_layer_names = [name for layer_tuple in group_layer_tuples for name in layer_tuple]
            group_layer_specs = {name: sm_spec.kv_cache_specs[name] for name in group_layer_names}
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert sub_sm_spec is not None
            swa_mla_groups.append(KVCacheGroupSpec(layer_names=group_layer_names, kv_cache_spec=sub_sm_spec))

    return [full_mla_group, full_mla_c128_group, *swa_mla_groups]


# ---------------------------------------------------------------------------
# Ascend DeepSeekV4 layout (ported from _get_kv_cache_config_deepseek_v4)
# ---------------------------------------------------------------------------
def _page_sizes(spec: UniformTypeKVCacheSpecs) -> set[int]:
    """Distinct page sizes across a group's cache specs.

    vLLM #51718 removed ``UniformTypeKVCacheSpecs.get_page_sizes()`` on main;
    the underlying ``kv_cache_specs``/``page_size_bytes`` are unchanged, so
    compute it inline to work on both lanes.
    """
    return {s.page_size_bytes for s in spec.kv_cache_specs.values()}


def _get_deepseek_v4_cache_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> tuple[list[int], list[dict[int, list[str]]], list[str], int, int]:
    """Return the geometry shared by DSV4 planning and rank normalization.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream (see _get_kv_cache_groups_uniform_groups) so
    every layer's page_size matches one of the full-MLA bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each layer
    at tuple_idx = position-within-bucket.
    """
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    page_sizes = sorted(_page_sizes(full_mla_spec))

    # Pre-bucket each group's layers by page_size (registration order within
    # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
    mtp_layer_names = []
    mtp_page_size = 0
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        specs = group.kv_cache_spec.kv_cache_specs
        b: dict[int, list[str]] = defaultdict(list)
        for name in group.layer_names:
            if "mtp" not in name:
                b[specs[name].page_size_bytes].append(name)
            else:
                mtp_layer_names.append(name)
                mtp_page_size = specs[name].page_size_bytes
        bucketed.append(b)

    # num_layer_tuples = longest bucket list across all groups. For the
    # full-MLA group this equals the count of layers in the largest
    # per-page-size bucket (= get_num_layer_tuples()); for SWA sub-groups
    # this equals the sub-group size (each has a single page_size).
    num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values()) + len(mtp_layer_names)

    return page_sizes, bucketed, mtp_layer_names, mtp_page_size, num_layer_tuples


def _get_kv_cache_config_deepseek_v4_main(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    (
        page_sizes,
        bucketed,
        mtp_layer_names,
        mtp_page_size,
        num_tuple_slots,
    ) = _get_deepseek_v4_cache_layout(kv_cache_groups)

    bytes_per_tuple = sum(page_sizes)
    num_blocks = available_memory // (bytes_per_tuple * num_tuple_slots)
    num_blocks = kv_cache_planning._may_override_num_blocks(vllm_config, num_blocks)

    tuple_stride = bytes_per_tuple * num_blocks
    backing_size = tuple_stride * num_tuple_slots

    # Within every tuple slot, page-size buckets are placed consecutively.
    page_offsets: dict[int, int] = {}
    page_prefix = 0
    for page_size in page_sizes:
        page_offsets[page_size] = page_prefix * num_blocks
        page_prefix += page_size

    tensors: list[KVCacheTensor] = []

    # Keep each descriptor inside one cache group. Descriptors from different
    # groups alias corresponding tuple slots by using identical geometry.
    for group_buckets in bucketed:
        for page_size in page_sizes:
            layer_names = group_buckets.get(page_size)
            if not layer_names:
                continue

            tensors.append(
                KVCacheTensor(
                    size=backing_size,
                    layers=list(layer_names),
                    offset=page_offsets[page_size],
                    layer_stride=tuple_stride,
                    block_stride=page_size,
                )
            )

    # MTP layers receive trailing tuple slots. Unused page buckets remain
    # padding so num_blocks and memory accounting retain the existing contract.
    normal_tuple_slots = num_tuple_slots - len(mtp_layer_names)
    for index, layer_name in enumerate(mtp_layer_names):
        slot = normal_tuple_slots + index
        tensors.append(
            KVCacheTensor(
                size=backing_size,
                layers=[layer_name],
                offset=slot * tuple_stride + page_offsets[mtp_page_size],
                layer_stride=0,
                block_stride=mtp_page_size,
            )
        )

    return num_blocks, tensors


def _is_deepseek_v4_groups(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
    """Whether the resolved groups carry a DeepSeekV4 shared-tuple layout."""
    if not kv_cache_groups or not all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) for group in kv_cache_groups
    ):
        return False
    for group in kv_cache_groups:
        group_spec = group.kv_cache_spec
        assert isinstance(group_spec, UniformTypeKVCacheSpecs)
        specs = group_spec.kv_cache_specs.values()
        if any(getattr(spec, "model_version", None) == "deepseek_v4" for spec in specs):
            return True
    return False


def _ascend_pool_bytes_per_block(kv_cache_groups: list[KVCacheGroupSpec]) -> int:
    """Use the same DSV4 divisor as Ascend's shared-tuple planner.

    The #53558 standard per-group layout has a different divisor from Ascend's
    DSV4 shared-tuple layout, so using the upstream value during the re-plan
    would change ``num_blocks`` and leave ranks inconsistent.
    """
    if not _is_deepseek_v4_groups(kv_cache_groups):
        return kv_cache_planning._pool_bytes_per_block(kv_cache_groups)

    page_sizes, _, _, _, num_layer_tuples = _get_deepseek_v4_cache_layout(kv_cache_groups)
    return sum(page_sizes) * num_layer_tuples


def _ascend_max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Keep the DSV4 admission formula for Ascend's shared tuples."""
    if not _is_deepseek_v4_groups(kv_cache_groups):
        return kv_cache_planning._max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)

    assert all(isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) for group in kv_cache_groups)
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    layer_tuple_bytes = sum(_page_sizes(full_mla_spec))
    num_layer_tuples = max(
        group.kv_cache_spec.get_max_layers_per_page_size()
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    )
    return sum(
        num_layer_tuples * group.kv_cache_spec.max_memory_usage_pages(vllm_config) * layer_tuple_bytes
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    )


def _ascend_get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Ascend DSV4 shared-tuple planner for the per-worker config step."""
    if not _is_deepseek_v4_groups(kv_cache_groups):
        return kv_cache_planning.get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory)

    num_blocks, kv_cache_tensors = _get_kv_cache_config_deepseek_v4_main(
        vllm_config,
        kv_cache_groups,
        available_memory,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )


class AscendKVCacheConfigBuilder(KVCacheConfigBuilder):
    """Platform KV cache planner for Ascend.

    Non-DeepSeekV4 models fall back to the default vLLM planning. DeepSeekV4
    models use Ascend's grouping and non-packed shared-tensor layout.
    """

    def build_kv_cache_configs(
        self,
        vllm_config: VllmConfig,
        kv_cache_specs: list[dict[str, KVCacheSpec]],
        available_memory: list[int],
    ) -> list[KVCacheConfig]:
        if not _has_deepseek_v4(kv_cache_specs):
            return super().build_kv_cache_configs(vllm_config, kv_cache_specs, available_memory)

        # --- merge specs (mirrors kv_cache_planning.get_kv_cache_configs) ---
        merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
        for kv_cache_spec_one_worker in kv_cache_specs:
            for layer_name, layer_spec in kv_cache_spec_one_worker.items():
                if layer_name not in merged_kv_cache_specs:
                    merged_kv_cache_specs[layer_name] = layer_spec
                else:
                    assert merged_kv_cache_specs[layer_name] == layer_spec, (
                        "The KV cache specs for the same layer are different across workers."
                    )
        KVCacheSpecRegistry.check_kv_cache_spec_registry(merged_kv_cache_specs)

        # When speculating with more than 1 speculative module (e.g. multi-layered MTP)
        # tag every SlidingWindowSpec with how many extra tokens to retain in the window.
        # (mirrors kv_cache_planning.get_kv_cache_configs)
        extra_retained_tokens = (
            vllm_config.speculative_config.num_speculative_tokens - 1
            if vllm_config.speculative_config is not None and vllm_config.speculative_config.use_multi_module_mtp()
            else 0
        )
        for layer_name, layer_spec in merged_kv_cache_specs.items():
            if isinstance(layer_spec, SlidingWindowSpec):
                merged_kv_cache_specs[layer_name] = replace(layer_spec, extra_retained_tokens=extra_retained_tokens)

        # --- global groups with Ascend DeepSeekV4 grouping ---
        grouped_specs = _ascend_group_and_unify_kv_cache_specs(merged_kv_cache_specs)
        assert grouped_specs is not None
        global_kv_cache_groups = _ascend_get_kv_cache_groups_uniform_groups(grouped_specs)
        kv_cache_planning._annotate_eagle_groups_deepseek_v4(vllm_config, merged_kv_cache_specs, global_kv_cache_groups)

        # --- project global groups onto each worker (PP sharding) ---
        projected_groups_per_worker = [
            kv_cache_planning._project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec)
            for worker_spec in kv_cache_specs
        ]

        # --- num_gpu_blocks_override decouples allocation from profiled memory ---
        override = vllm_config.cache_config.num_gpu_blocks_override
        if override is not None:
            adjusted_memory: list[int] = []
            for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
                if not groups:
                    adjusted_memory.append(avail_mem)
                    continue
                bytes_per_block = _ascend_pool_bytes_per_block(groups)
                adjusted_memory.append(override * bytes_per_block)
            available_memory = adjusted_memory

        # --- reserve the null block BlockPool permanently holds back ---
        check_memory = [
            avail_mem - _ascend_pool_bytes_per_block(groups) if groups else avail_mem
            for groups, avail_mem in zip(projected_groups_per_worker, available_memory)
        ]

        # --- auto-fit max_model_len when set to -1 ---
        if vllm_config.model_config.original_max_model_len == -1:
            kv_cache_planning._auto_fit_max_model_len(vllm_config, projected_groups_per_worker, check_memory)

        # --- check available memory per worker ---
        for groups, avail_mem in zip(projected_groups_per_worker, check_memory):
            if not groups:
                continue
            kv_cache_planning._check_enough_kv_cache_memory(
                avail_mem,
                partial(_ascend_max_memory_usage_bytes_from_groups, vllm_config, groups),
                vllm_config.model_config.max_model_len,
                partial(kv_cache_planning._estimate_max_model_len_from_groups, vllm_config, groups),
            )

        # --- per-worker config with Ascend DSV4 shared-tuple layout ---
        kv_cache_configs: list[KVCacheConfig] = []
        for projected_groups, available_memory_one_worker in zip(projected_groups_per_worker, available_memory):
            kv_cache_configs.append(
                _ascend_get_kv_cache_config_from_groups(vllm_config, projected_groups, available_memory_one_worker)
            )

        # --- shrink each rank to the smallest num_blocks across ranks ---
        min_num_blocks = min(cfg.num_blocks for cfg in kv_cache_configs)
        for i, kv_cache_config in enumerate(kv_cache_configs):
            if kv_cache_config.num_blocks == min_num_blocks:
                continue
            groups = kv_cache_config.kv_cache_groups
            kv_cache_configs[i] = _ascend_get_kv_cache_config_from_groups(
                vllm_config, groups, min_num_blocks * _ascend_pool_bytes_per_block(groups)
            )
        return kv_cache_configs
