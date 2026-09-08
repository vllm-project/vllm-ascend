# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import math
from collections import defaultdict
from dataclasses import replace

import vllm.v1.core.kv_cache_utils
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_utils import _approximate_gcd, may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

import vllm_ascend.envs as envs_ascend
from vllm_ascend.core.typed_kv_cache import (
    TypedAddressPool,
    TypedKVCachePlan,
    TypedPageSpec,
    get_typed_kv_cache_plan,
    set_typed_kv_cache_plan,
)
from vllm_ascend.device.device_config import is_310p

_orig_resolve_kv_cache_block_sizes = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes
_orig_get_kv_cache_configs = vllm.v1.core.kv_cache_utils.get_kv_cache_configs
_orig_update_kv_cache_capacity = vllm.v1.core.kv_cache_utils.update_kv_cache_capacity


def _typed_base_spec(spec: KVCacheSpec) -> KVCacheSpec:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec


def _without_uniform_page_padding(
    spec: KVCacheSpec,
    attention_block_size: int,
) -> KVCacheSpec:
    """Restore each group's physical page instead of the HMA common page."""

    if isinstance(spec, UniformTypeKVCacheSpecs):
        inner = {
            name: _without_uniform_page_padding(layer_spec, attention_block_size)
            for name, layer_spec in spec.kv_cache_specs.items()
        }
        block_sizes = {layer_spec.block_size for layer_spec in inner.values()}
        if len(block_sizes) != 1:
            raise ValueError("typed KV cache group has non-uniform block sizes")
        return UniformTypeKVCacheSpecs(
            block_size=next(iter(block_sizes)),
            kv_cache_specs=inner,
        )
    if isinstance(spec, AttentionSpec):
        return replace(
            spec,
            block_size=attention_block_size,
            page_size_padded=None,
        )
    if isinstance(spec, MambaSpec):
        return replace(spec, page_size_padded=None)
    raise ValueError(f"typed KV cache MVP only supports AttentionSpec and MambaSpec, got {type(spec).__name__}")


def _validate_typed_kv_cache_mode(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> None:
    if is_310p():
        raise ValueError("typed KV cache MVP is not supported on Ascend 310P")
    cache_config = vllm_config.cache_config
    parallel_config = vllm_config.parallel_config
    if cache_config.enable_prefix_caching:
        raise ValueError("typed KV cache MVP requires prefix caching to be disabled")
    if cache_config.num_gpu_blocks_override is not None:
        raise ValueError("typed KV cache MVP does not support num_gpu_blocks_override")
    if vllm_config.speculative_config is not None:
        raise ValueError("typed KV cache MVP does not support speculative decoding")
    if vllm_config.kv_transfer_config is not None:
        raise ValueError("typed KV cache MVP does not support KV transfer/offload")
    if parallel_config.decode_context_parallel_size != 1:
        raise ValueError("typed KV cache MVP does not support DCP")
    if getattr(parallel_config, "prefill_context_parallel_size", 1) != 1:
        raise ValueError("typed KV cache MVP does not support PCP")
    if parallel_config.pipeline_parallel_size != 1:
        raise ValueError("typed KV cache MVP does not support pipeline parallelism")
    if any(tensor.offset or tensor.block_stride for tensor in kv_cache_config.kv_cache_tensors):
        raise ValueError("typed KV cache MVP does not support packed KV tensors")

    base_specs = [_typed_base_spec(group.kv_cache_spec) for group in kv_cache_config.kv_cache_groups]
    if not any(isinstance(spec, AttentionSpec) for spec in base_specs) or not any(
        isinstance(spec, MambaSpec) for spec in base_specs
    ):
        raise ValueError("typed KV cache MVP requires a hybrid Attention/Mamba model")

    group_layers = [set(group.layer_names) for group in kv_cache_config.kv_cache_groups]
    for tensor in kv_cache_config.kv_cache_tensors:
        shared = set(tensor.shared_by)
        if any(len(shared & layers) != 1 for layers in group_layers):
            raise ValueError("each typed raw tensor must be shared by exactly one layer from every KV cache group")


def _enable_typed_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> None:
    _validate_typed_kv_cache_mode(vllm_config, kv_cache_config)
    attention_block_size = getattr(
        vllm_config.cache_config,
        "_ascend_typed_attention_block_size",
        None,
    )
    if attention_block_size is None:
        raise ValueError(
            "typed KV cache requires the native attention kernel block size "
            "captured by NPUPlatform.update_block_size_for_backend"
        )
    for group in kv_cache_config.kv_cache_groups:
        group.kv_cache_spec = _without_uniform_page_padding(
            group.kv_cache_spec,
            attention_block_size,
        )

    specs = tuple(
        TypedPageSpec(
            group_id=group_id,
            page_size_bytes=group.kv_cache_spec.page_size_bytes,
            block_size_tokens=group.kv_cache_spec.block_size,
        )
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
    )
    tensor_sizes = {tensor.size for tensor in kv_cache_config.kv_cache_tensors}
    if len(tensor_sizes) != 1:
        raise ValueError("typed KV cache MVP requires equal raw tensor budgets")
    original_tensor_size = next(iter(tensor_sizes))
    mode = envs_ascend.VLLM_ASCEND_TYPED_KV_CACHE_MODE
    if mode == "address_table":
        # The scheduler and worker independently construct the same immutable
        # address tables. Candidate intervals deliberately overlap across
        # groups; the typed allocator ensures that only non-overlapping pages
        # are live.
        plan = TypedKVCachePlan.addressed(specs, original_tensor_size)
    elif mode == "static_partition":
        # Reserve a fixed address range for every group. Keep using the same
        # address-table worker and kernel path as the dynamic mode so the A/B
        # changes allocation policy only. The end-to-end correctness gate must
        # still verify that every device operator honors the resulting stride.
        # ranges deliberately do not overlap: released Attention bytes can
        # never become Mamba pages (or vice versa).
        blocks_per_request = tuple(
            math.ceil(group.kv_cache_spec.max_memory_usage_bytes(vllm_config) / group.kv_cache_spec.page_size_bytes)
            for group in kv_cache_config.kv_cache_groups
        )
        null_bytes = max(spec.page_size_bytes for spec in specs)
        bytes_per_request = sum(blocks * spec.page_size_bytes for spec, blocks in zip(specs, blocks_per_request))
        full_requests = (original_tensor_size - null_bytes) // bytes_per_request

        def build_tables(
            data_block_counts: list[int],
        ) -> tuple[tuple[tuple[int, ...], ...], int]:
            tables = []
            offset = null_bytes
            for spec, count in zip(specs, data_block_counts):
                offset = (offset + spec.page_size_bytes - 1) // spec.page_size_bytes * spec.page_size_bytes
                tables.append((0,) + tuple(offset + block_id * spec.page_size_bytes for block_id in range(count)))
                offset += count * spec.page_size_bytes
            return tuple(tables), offset

        # Alignment gaps between fixed regions mean the byte-only estimate can
        # be one request too optimistic.  Tighten it against the actual layout.
        while full_requests > 0:
            data_block_counts = [full_requests * blocks for blocks in blocks_per_request]
            _, used_bytes = build_tables(data_block_counts)
            if used_bytes <= original_tensor_size:
                break
            full_requests -= 1
        if full_requests < 1:
            raise ValueError("typed static partition cannot hold one max-length request")

        # Spend remaining whole pages on the group with the lowest equivalent
        # max-length request capacity. Rebuilding the tiny table list also
        # accounts for alignment shifts caused by growing an earlier region.
        while True:
            candidates = []
            for group_id in range(len(specs)):
                candidate_counts = data_block_counts.copy()
                candidate_counts[group_id] += 1
                _, candidate_end = build_tables(candidate_counts)
                if candidate_end <= original_tensor_size:
                    candidates.append(group_id)
            if not candidates:
                break
            group_id = min(
                candidates,
                key=lambda candidate: (
                    data_block_counts[candidate] / blocks_per_request[candidate],
                    specs[candidate].page_size_bytes,
                    candidate,
                ),
            )
            data_block_counts[group_id] += 1

        page_address_tables, _ = build_tables(data_block_counts)
        plan = TypedKVCachePlan.addressed(
            specs,
            original_tensor_size,
            page_address_tables,
        )
    else:
        raise ValueError(f"VLLM_ASCEND_TYPED_KV_CACHE_MODE must be address_table or static_partition, got {mode!r}")

    kv_cache_config.num_blocks = min(plan.num_blocks(spec.group_id) for spec in plan.specs)
    for tensor in kv_cache_config.kv_cache_tensors:
        tensor.size = plan.total_managed_bytes
    set_typed_kv_cache_plan(kv_cache_config, plan)
    logger.info(
        "Enabled typed KV cache: mode=%s superpage_size=%d, num_superpages=%d managed_bytes_per_tensor=%d, groups=%s",
        mode,
        plan.superpage_size_bytes,
        plan.num_superpages,
        plan.total_managed_bytes,
        [
            {
                "group_id": spec.group_id,
                "page_size": spec.page_size_bytes,
                "block_size": spec.block_size_tokens,
                "logical_pages": plan.num_blocks(spec.group_id),
                "last_physical_offset": (
                    plan.page_address_tables_bytes[spec.group_id][-1]
                    if plan.is_addressed
                    else plan.region_offset_bytes(spec.group_id)
                    + (plan.num_blocks(spec.group_id) - 1) * spec.page_size_bytes
                ),
            }
            for spec in plan.specs
        ],
    )


def _ascend_get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    configs = _orig_get_kv_cache_configs(
        vllm_config,
        kv_cache_specs,
        available_memory,
    )
    if envs_ascend.VLLM_ASCEND_ENABLE_TYPED_KV_CACHE:
        for config in configs:
            _enable_typed_kv_cache_config(vllm_config, config)
    return configs


def _typed_max_concurrency(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    plan: TypedKVCachePlan,
) -> float:
    blocks_per_request = []
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        blocks_per_request.append(math.ceil(spec.max_memory_usage_bytes(vllm_config) / spec.page_size_bytes))

    if plan.is_addressed:
        pool = TypedAddressPool(plan)

        def fits(num_requests: int) -> bool:
            return pool.can_allocate(
                {group_id: num_requests * blocks for group_id, blocks in enumerate(blocks_per_request)}
            )

        low, high = 0, 1
        while fits(high):
            low, high = high, high * 2
        while low + 1 < high:
            mid = (low + high) // 2
            if fits(mid):
                low = mid
            else:
                high = mid
        return float(low)

    if plan.is_partitioned:
        return float(
            min((plan.num_blocks(group_id) - 1) // blocks for group_id, blocks in enumerate(blocks_per_request))
        )

    def fits(num_requests: int) -> bool:
        used = sum(
            math.ceil(num_requests * blocks / plan.capacity(group_id))
            for group_id, blocks in enumerate(blocks_per_request)
        )
        return used <= plan.num_superpages - 1

    low, high = 0, 1
    while fits(high):
        low, high = high, high * 2
    while low + 1 < high:
        mid = (low + high) // 2
        if fits(mid):
            low = mid
        else:
            high = mid
    return float(low)


def _ascend_update_kv_cache_capacity(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> None:
    plan = get_typed_kv_cache_plan(kv_cache_config)
    if plan is None:
        return _orig_update_kv_cache_capacity(vllm_config, kv_cache_config)
    concurrency = _typed_max_concurrency(vllm_config, kv_cache_config, plan)
    max_model_len = vllm_config.model_config.max_model_len
    vllm_config.cache_config.kv_cache_size_tokens = int(concurrency * max_model_len)
    vllm_config.cache_config.kv_cache_max_concurrency = concurrency
    logger.info_once(
        "Typed GPU KV cache size: %s tokens, maximum safe concurrency for %s tokens per request: %.2fx",
        f"{int(concurrency * max_model_len):,}",
        f"{max_model_len:,}",
        concurrency,
    )


def _ascend_resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Ascend-compatible resolve_kv_cache_block_sizes.

    vLLM PR #40860 added a restriction that hybrid KV cache groups with
    multiple block sizes do not support DCP.
    This restriction is correct for CUDA but not for Ascend, which implements
    context parallelism for MLA and SWA-MLA layers independently.

    For multiple KV cache groups with CP, compute scheduler_block_size as
    lcm(group_block_sizes) * dcp to maintain alignment.
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    if len(groups) <= 1:
        bs = cache_config.block_size * dcp
        return bs, bs

    if dcp != 1:
        # Ascend supports CP with multiple KV cache groups; compute
        # scheduler_block_size using the LCM of all group block sizes
        # multiplied by the CP factors for proper alignment.
        group_block_sizes = [g.kv_cache_spec.block_size for g in groups]
        scheduler_block_size = math.lcm(*group_block_sizes) * dcp
        if not cache_config.enable_prefix_caching:
            return scheduler_block_size, scheduler_block_size
        hash_block_size = math.gcd(*group_block_sizes)
        return scheduler_block_size, hash_block_size

    return _orig_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
    """
    if not any(isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()):
        return None

    ratio_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    grouped_swa_mla_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[spec.block_size][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            ratio_specs[spec.compress_ratio][name] = spec

    mla_uniform_specs = []
    for ratio in sorted(ratio_specs, key=lambda r: (r != 4, r)):
        spec_dict = ratio_specs[ratio]
        assert len(spec_dict) > 0
        mla_uniform_specs.append(UniformTypeKVCacheSpecs.from_specs(spec_dict))
    assert mla_uniform_specs is not None

    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    return [*mla_uniform_specs, *swa_uniform_specs]


def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    """
    Generate the KV cache groups from the grouped specs.
    """
    assert len(grouped_specs) > 0 and all(isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs)
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
    num_layer_tuples_per_group: list[int] = [g_spec.get_num_layer_tuples() for g_spec in grouped_specs]
    # Choose `num_layer_tuples` to minimize total padding across groups.
    num_layer_tuples = _approximate_gcd(num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0])
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
    all_page_sizes = full_mla_spec.get_page_sizes()
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        sm_page_sizes = sm_spec.get_page_sizes()
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
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_sm_spec,
                )
            )

    return [full_mla_group, full_mla_c128_group, *swa_mla_groups]


def _get_kv_cache_config_deepseek_v4(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    """DeepseekV4 KV cache tensor layout planning.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream (see _get_kv_cache_groups_uniform_groups) so
    every layer's page_size matches one of the full-MLA bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each
    layer at tuple_idx = position-within-bucket. Emit one KVCacheTensor
    per (tuple_idx, bucket) whose shared_by is the union of per-group
    layers at that slot.
    """
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    page_sizes = sorted(full_mla_spec.get_page_sizes())
    layer_tuple_page_bytes = sum(page_sizes)

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

    num_blocks = available_memory // (layer_tuple_page_bytes * num_layer_tuples)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    kv_cache_tensors: list[KVCacheTensor] = []
    for tuple_idx in range(num_layer_tuples - len(mtp_layer_names)):
        for ps in page_sizes:
            shared_by: list[str] = []
            for b in bucketed:
                bucket = b.get(ps)
                if bucket is not None and tuple_idx < len(bucket):
                    shared_by.append(bucket[tuple_idx])
            kv_cache_tensors.append(KVCacheTensor(size=ps * num_blocks, shared_by=shared_by))
    for i in range(len(mtp_layer_names)):
        kv_cache_tensors.append(KVCacheTensor(size=mtp_page_size * num_blocks, shared_by=[mtp_layer_names[i]]))

    return num_blocks, kv_cache_tensors


vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
vllm.v1.core.kv_cache_utils.get_kv_cache_configs = _ascend_get_kv_cache_configs
vllm.v1.core.kv_cache_utils.update_kv_cache_capacity = _ascend_update_kv_cache_capacity
vllm.v1.core.kv_cache_utils.group_and_unify_kv_cache_specs = group_and_unify_kv_cache_specs
vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_groups = _get_kv_cache_groups_uniform_groups
# vLLM v0.24.0 renamed _get_kv_cache_config_deepseek_v4 to _get_kv_cache_config_packed and
# get_kv_cache_config_from_groups now calls _get_kv_cache_config_packed directly, bypassing
# the alias patch above. Patch the canonical name so Ascend's non-packed layout is used.
vllm.v1.core.kv_cache_utils._get_kv_cache_config_packed = _get_kv_cache_config_deepseek_v4

# Also patch the reference used by engine/core.py which imports the function directly.
import vllm.v1.engine.core  # noqa: E402

vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
vllm.v1.engine.core.get_kv_cache_configs = _ascend_get_kv_cache_configs
vllm.v1.engine.core.update_kv_cache_capacity = _ascend_update_kv_cache_capacity
