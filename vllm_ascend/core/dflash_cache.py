# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Workarounds for mixed Full/SWA DFlash caches on Ascend V2.

REMOVAL GUIDE
=============
This module is self-contained: deleting it together with the marked call
sites below removes the workaround entirely. Every call site carries the
marker comment ``# DFLASH-MIXED-WINDOW-CACHE-WORKAROUND`` so a single
``git grep DFLASH-MIXED-WINDOW-CACHE-WORKAROUND`` lists all touchpoints:

- ``vllm_ascend/attention/attention_v1.py`` — ``_mask_dflash_cache_slots``
  and its two call sites (null-block masking on both write paths).
- ``vllm_ascend/worker/v2/attn_utils.py`` — ``align_dflash_cache_specs`` in
  ``get_kv_cache_spec``; view validation and null-block attributes in
  ``_reshape_kv_cache_v2``.
- ``vllm_ascend/patch/platform/patch_kv_cache_utils.py`` — the planner
  wrapper registration at the bottom of the file, plus its documentation
  entry in ``vllm_ascend/patch/__init__.py``.

Re-evaluate — do not blindly keep — on every upstream vLLM bump:

1. The block-size realignment is unnecessary once the upstream hybrid KV
   cache manager stops sharing one backing and one ``num_blocks`` across
   Full and SWA groups, or the Ascend allocator stops materializing
   contiguous per-component planes.
2. The FIA address guard is unnecessary once the CANN FIA operator is fixed
   above 2**16 kernel blocks / 2**32 plane elements. Rerun the NPU probe
   from docs/diagnostics/dflash_mixed_window_cache.md before removing it.
3. The null-block masking is unnecessary once upstream stops inserting null
   block IDs into sliding-window block tables, or DFlash stops prewriting
   context K/V for all target tokens.

The ``tests/ut/worker/test_dflash_cache*.py`` suites encode the upstream
layout contract; if they start failing after an upstream bump, that is the
tripwire to revisit this module.
"""

import copy
import math
from dataclasses import fields, replace
from functools import wraps

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.dflash_cache_layout import (
    get_dflash_aligned_block_size,
    get_dflash_fia_safe_num_blocks,
)


def uses_mixed_dflash_cache(vllm_config) -> bool:
    speculative = getattr(vllm_config, "speculative_config", None)
    if not getattr(vllm_config, "use_v2_model_runner", False) or speculative is None or speculative.method != "dflash":
        return False
    draft_model_config = getattr(speculative, "draft_model_config", None)
    draft_config = getattr(draft_model_config, "hf_config", None)
    layer_types = getattr(draft_config, "layer_types", None) or ()
    return "sliding_attention" in layer_types and "full_attention" in layer_types


class DFlashKVCacheSpecs(dict[str, KVCacheSpec]):
    """Carry loaded draft ownership through the worker's pickled spec RPC.

    Spec objects remain ordinary upstream dataclasses (including across
    ``replace`` and registry checks). Names come from the loaded speculator,
    never from a layer-name prefix or an assumption about the target depth.
    """

    def __init__(self, specs: dict[str, KVCacheSpec], draft_layer_names: set[str]):
        super().__init__(specs)
        self.draft_layer_names = frozenset(draft_layer_names)


def _with_dflash_draft_layers(vllm_config, specs):
    names = getattr(specs, "draft_layer_names", None)
    if not uses_mixed_dflash_cache(vllm_config) or not names:
        return vllm_config
    if not names <= specs.keys() or any(isinstance(specs[name], MambaSpec) for name in names):
        raise ValueError("Mixed DFlash requires valid loaded draft attention layer names.")
    config = copy.copy(vllm_config)
    config._ascend_dflash_draft_layer_names = names
    return config


def annotate_dflash_cache_groups(vllm_config, groups):
    names = getattr(vllm_config, "_ascend_dflash_draft_layer_names", None)
    if uses_mixed_dflash_cache(vllm_config) and names:
        for group in groups:
            group.is_eagle_group = any(name in names for name in group.layer_names)


def wrap_dflash_cache_group_annotation(original_annotation):
    @wraps(original_annotation)
    def annotate(vllm_config, kv_cache_spec, kv_cache_groups, *args, **kwargs):
        original_annotation(vllm_config, kv_cache_spec, kv_cache_groups, *args, **kwargs)
        # Run before upstream's unannotated-drafter warning and before the
        # group projection that transports this flag to the scheduler.
        annotate_dflash_cache_groups(vllm_config, kv_cache_groups)

    return annotate


def wrap_dflash_cache_group_builder(original_builder):
    @wraps(original_builder)
    def build(vllm_config, kv_cache_spec):
        config = _with_dflash_draft_layers(vllm_config, kv_cache_spec)
        selected = getattr(config, "_ascend_dflash_cache_groups", None)
        if uses_mixed_dflash_cache(config) and selected is not None:
            names = [name for group in selected for name in group.layer_names]
            if (
                len(names) != len(kv_cache_spec)
                or set(names) != kv_cache_spec.keys()
                or any(kv_cache_spec[name] != group.kv_cache_spec for group in selected for name in group.layer_names)
            ):
                raise ValueError("Mixed DFlash regrouping changed cache layer coverage or specs.")
            return [replace(group, layer_names=list(group.layer_names)) for group in selected]
        groups = original_builder(config, kv_cache_spec)
        # Also covers upstream grouping fast paths and releases without an
        # annotation helper. No identity guesses if the worker sent no names.
        annotate_dflash_cache_groups(config, groups)
        return groups

    return build


def align_dflash_cache_specs(vllm_config, specs):
    """Align actual contiguous planes, not only their advertised page size."""
    if not uses_mixed_dflash_cache(vllm_config):
        return specs
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if getattr(parallel_config, "pipeline_parallel_size", 1) != 1:
        raise ValueError("Ascend mixed DFlash cache alignment currently requires pipeline_parallel_size=1.")
    mamba_specs = [spec for spec in specs.values() if isinstance(spec, MambaSpec)]
    if not mamba_specs:
        return specs
    if any(type(spec) not in (FullAttentionSpec, SlidingWindowSpec, MambaSpec) for spec in specs.values()):
        raise ValueError("Mixed DFlash with Mamba requires ordinary Full/SWA KV cache specs on Ascend.")
    if any(len(spec.shapes) != 2 or len(spec.dtypes) != 2 for spec in mamba_specs):
        raise ValueError("Mixed DFlash requires exactly two conv/SSM cache components on Ascend.")
    state_layouts = {
        tuple(math.prod(shape) * get_dtype_size(dtype) for shape, dtype in zip(spec.shapes, spec.dtypes))
        for spec in mamba_specs
    }
    if len(state_layouts) != 1 or len(next(iter(state_layouts))) != 2:
        raise ValueError("Mixed DFlash requires a uniform conv/SSM cache layout on Ascend.")
    conv_bytes, ssm_bytes = next(iter(state_layouts))
    common_page_bytes = max(spec.page_size_bytes for spec in specs.values())
    if any(
        getattr(spec, "cache_dtype_str", None) not in (None, "auto", "float16", "bfloat16") for spec in specs.values()
    ):
        raise ValueError("Mixed DFlash cache alignment does not support quantized KV storage on Ascend.")
    result = {}
    for name, spec in specs.items():
        if isinstance(spec, MambaSpec):
            result[name] = replace(spec, page_size_padded=common_page_bytes)
            continue
        if str(spec.dtype) not in ("torch.float16", "torch.bfloat16"):
            raise ValueError("Mixed DFlash contiguous KV alignment requires FP16/BF16 cache on Ascend.")
        dtype_bytes = get_dtype_size(spec.dtype)
        block_size = get_dflash_aligned_block_size(
            conv_page_bytes=conv_bytes,
            ssm_page_bytes=ssm_bytes,
            common_page_size_bytes=common_page_bytes,
            key_row_bytes=spec.num_kv_heads * spec.head_size * dtype_bytes,
            value_row_bytes=spec.num_kv_heads * spec.head_size_v * dtype_bytes,
        )
        if spec.block_size > block_size:
            raise ValueError(f"Mixed DFlash cache block for {name} exceeds its aligned SSM plane.")
        result[name] = replace(spec, block_size=block_size, page_size_padded=common_page_bytes)
        if spec.block_size != block_size:
            logger.info(
                "DFlash mixed cache layout aligned: layer=%s, block_size=%d -> %d, "
                "sliding_window=%s, plane_bytes=%d, page_bytes=%d",
                name,
                spec.block_size,
                block_size,
                getattr(spec, "sliding_window", None),
                ssm_bytes,
                common_page_bytes,
            )
    return result


def validate_dflash_cache_views(vllm_config, cache_config, raw_caches, caches):
    """Check the actual affine addresses once, without reading device data."""
    if not uses_mixed_dflash_cache(vllm_config):
        return
    specs = _layer_specs(cache_config)
    if not any(isinstance(spec, MambaSpec) for spec in specs.values()):
        return
    align_dflash_cache_specs(vllm_config, specs)
    mamba = next(spec for spec in specs.values() if isinstance(spec, MambaSpec))
    conv_bytes, ssm_bytes = (
        math.prod(shape) * get_dtype_size(dtype) for shape, dtype in zip(mamba.shapes, mamba.dtypes)
    )
    count = cache_config.num_blocks
    arena_offsets: dict[int, int] = {}
    for name, spec in specs.items():
        raw = raw_caches[name]
        if isinstance(raw, tuple) or raw.numel() * raw.element_size() != count * spec.page_size_bytes:
            raise ValueError(f"Mixed DFlash cache {name} does not have the planned contiguous backing.")
        arena = raw.untyped_storage().data_ptr()
        layer_bytes = count * spec.page_size_bytes
        offset_mod = (raw.data_ptr() - arena) % layer_bytes
        if arena in arena_offsets and arena_offsets[arena] != offset_mod:
            raise ValueError("Mixed DFlash shared layer arenas are not aligned to whole layer pools.")
        arena_offsets[arena] = offset_mod
        if isinstance(spec, MambaSpec):
            offsets = (0, count * conv_bytes)
            sizes = (count * conv_bytes, count * ssm_bytes)
            dtypes = spec.dtypes
        else:
            offsets = (count * conv_bytes, count * (conv_bytes + ssm_bytes))
            sizes = (count * ssm_bytes, count * ssm_bytes)
            dtypes = (spec.dtype, spec.dtype)
        parts = caches[name]
        if len(parts) != 2:
            raise ValueError(f"Mixed DFlash cache {name} must have exactly two component planes.")
        for part, offset, size, dtype in zip(parts, offsets, sizes, dtypes):
            if (
                not part.is_contiguous()
                or part.dtype != dtype
                or part.numel() * part.element_size() != size
                or part.data_ptr() - raw.data_ptr() != offset
            ):
                raise ValueError(f"Mixed DFlash cache {name} violates the aligned contiguous-plane layout.")
    logger.info(
        "DFlash mixed cache views verified: layers=%d, physical_blocks=%d, conv_bytes=%d, plane_bytes=%d",
        len(specs),
        count,
        conv_bytes,
        ssm_bytes,
    )


def _layer_specs(cache_config):
    specs = {}
    for group in cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            specs.update(spec.kv_cache_specs)
        else:
            specs.update(dict.fromkeys(group.layer_names, spec))
    return specs


def _pool_bytes_per_block(cache_config):
    tensors = cache_config.kv_cache_tensors
    sizes = [tensor.size for tensor in tensors]
    if not sizes or cache_config.num_blocks <= 0:
        raise ValueError("Mixed DFlash needs a non-empty, positive KV cache plan.")
    if hasattr(tensors[0], "layers"):
        # vLLM #51718 descriptors are views of ONE shared hybrid backing.
        if len(set(sizes)) != 1:
            raise ValueError("Mixed DFlash cache descriptors must share one backing size.")
        pool_bytes = sizes[0]
    else:
        # v0.28 shared_by descriptors each own a separate allocation.
        pool_bytes = sum(sizes)
    if pool_bytes % cache_config.num_blocks:
        raise ValueError("Mixed DFlash KV cache size must contain whole physical blocks.")
    return pool_bytes // cache_config.num_blocks


def _cache_group_summary(cache_config) -> str:
    summaries: list[str] = []
    for index, group in enumerate(cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        specs = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
        layouts = sorted(
            {
                f"{type(item).__name__}(block_size={item.block_size},"
                f"sliding_window={getattr(item, 'sliding_window', None)})"
                for item in specs
            }
        )
        summaries.append(f"{index}:layers={len(group.layer_names)}," + "/".join(layouts))
    return "; ".join(summaries)


def wrap_dflash_cache_planner(original_planner):
    """Re-plan before allocation using both the real budget and FIA limits."""

    @wraps(original_planner)
    def plan(vllm_config, kv_cache_specs, available_memory):
        if uses_mixed_dflash_cache(vllm_config) and kv_cache_specs:
            worker_draft_names = [getattr(specs, "draft_layer_names", None) for specs in kv_cache_specs]
            if any(worker_draft_names):
                if any(names != worker_draft_names[0] for names in worker_draft_names):
                    raise ValueError("Mixed DFlash workers disagree on loaded draft cache layers.")
                for specs in kv_cache_specs:
                    _with_dflash_draft_layers(vllm_config, specs)
                # Upstream merges the worker dictionaries into a plain dict.
                # Keep ownership on this call-local config through that merge.
                vllm_config = _with_dflash_draft_layers(vllm_config, kv_cache_specs[0])
        configs = original_planner(vllm_config, kv_cache_specs, available_memory)
        if not uses_mixed_dflash_cache(vllm_config):
            return configs
        if len(configs) != len(available_memory) or len(kv_cache_specs) != len(available_memory):
            raise ValueError("Mixed DFlash requires a cache plan and memory budget for every worker.")
        limits: list[int] = []
        capacity_diagnostics: list[tuple[int, int, int, int | None]] = []
        active_configs = []
        for worker_index, (config, budget) in enumerate(zip(configs, available_memory)):
            specs = _layer_specs(config)
            if not specs or not any(isinstance(spec, MambaSpec) for spec in specs.values()):
                continue
            aligned = align_dflash_cache_specs(vllm_config, specs)
            if any(aligned[name] != spec for name, spec in specs.items()):
                raise ValueError("Mixed DFlash cache specs changed after alignment; refusing unsafe allocation.")
            active_configs.append(config)
            budget_limit = budget // _pool_bytes_per_block(config)
            limits.extend((config.num_blocks, budget_limit))
            fia_limits: list[int] = []
            for spec in specs.values():
                if isinstance(spec, MambaSpec):
                    continue
                dtype_bytes = get_dtype_size(spec.dtype)
                fia_limits.append(
                    get_dflash_fia_safe_num_blocks(
                        storage_block_size=spec.block_size,
                        key_row_bytes=spec.num_kv_heads * spec.head_size * dtype_bytes,
                        value_row_bytes=spec.num_kv_heads * spec.head_size_v * dtype_bytes,
                        key_element_bytes=dtype_bytes,
                        value_element_bytes=dtype_bytes,
                    )
                )
            # Keep diagnostic accounting on the startup path; no device reads.
            fia_limit = min(fia_limits) if fia_limits else None
            limits.extend(fia_limits)
            capacity_diagnostics.append((worker_index, config.num_blocks, budget_limit, fia_limit))
        if not active_configs:
            return configs
        safe_blocks = min(limits)
        if safe_blocks < 2:
            raise ValueError("No safe mixed DFlash KV cache blocks fit the available memory.")
        selected_groups = None
        draft_names = getattr(vllm_config, "_ascend_dflash_draft_layer_names", None)
        if draft_names and len(active_configs) == len(configs):
            specs = _layer_specs(configs[0])
            if all(
                _layer_specs(config) == specs and config.kv_cache_groups == configs[0].kv_cache_groups
                for config in configs
            ):
                selected_groups = choose_dflash_cache_groups(
                    vllm_config,
                    specs,
                    configs[0].kv_cache_groups,
                    draft_layer_names=set(draft_names),
                    available_memory=min(available_memory),
                    max_num_blocks=safe_blocks,
                )
        if selected_groups is not None:
            # A wider pool uses more bytes per physical block. Apply the real
            # budget BEFORE upstream admission/descriptor construction, not
            # after allocation. The per-plane FIA limit is unchanged.
            pool_bytes = (
                max(len(group.layer_names) for group in selected_groups)
                * selected_groups[0].kv_cache_spec.page_size_bytes
            )
            safe_blocks = min(safe_blocks, min(available_memory) // pool_bytes)
            logger.info(
                "DFlash mixed cache regrouped: groups=%d -> %d, pool_width=%d -> %d, "
                "physical_blocks=%d, draft_groups=%d (same specs and FIA limit)",
                len(configs[0].kv_cache_groups),
                len(selected_groups),
                max(len(group.layer_names) for group in configs[0].kv_cache_groups),
                max(len(group.layer_names) for group in selected_groups),
                safe_blocks,
                sum(group.is_eagle_group for group in selected_groups),
            )
        if selected_groups is not None or any(config.num_blocks > safe_blocks for config in active_configs):
            logger.warning(
                "DFlash mixed cache guard: physical_blocks=%d -> %d; "
                "applying the profiled memory budget and FIA address limits before allocation.",
                max(config.num_blocks for config in active_configs),
                safe_blocks,
            )
            safe_config = copy.copy(vllm_config)
            safe_config.cache_config = copy.copy(vllm_config.cache_config)
            safe_config.cache_config.num_gpu_blocks_override = safe_blocks
            if selected_groups is not None:
                safe_config._ascend_dflash_cache_groups = selected_groups
            # Re-run admission, auto-fit and descriptor construction together.
            # In particular, offsets/layer strides on main depend on num_blocks.
            configs = original_planner(safe_config, kv_cache_specs, available_memory)
        if len(configs) != len(available_memory):
            raise ValueError("Replanned mixed DFlash cache is missing worker plans.")
        for config, budget in zip(configs, available_memory):
            if selected_groups is not None and config.kv_cache_groups != selected_groups:
                raise ValueError("Replanned mixed DFlash KV cache lost its selected groups or draft ownership.")
            specs = _layer_specs(config)
            if not specs:
                continue
            if config.num_blocks > safe_blocks or config.num_blocks * _pool_bytes_per_block(config) > budget:
                raise ValueError("Replanned mixed DFlash KV cache exceeds its memory/address limit.")
            aligned = align_dflash_cache_specs(vllm_config, specs)
            if any(aligned[name] != spec for name, spec in specs.items()):
                raise ValueError("Replanned mixed DFlash KV cache lost its aligned layout.")
        for worker_index, original_blocks, budget_limit, fia_limit in capacity_diagnostics:
            config = configs[worker_index]
            logger.info(
                "DFlash mixed cache capacity: worker=%d, original_planned_blocks=%d, "
                "budget_limit_blocks=%d, fia_limit_blocks=%s, override=%s, effective_blocks=%d, "
                "pool_bytes=%d, budget_bytes=%d, effective_budget_limit_blocks=%d, groups=%d [%s]",
                worker_index,
                original_blocks,
                budget_limit,
                fia_limit,
                vllm_config.cache_config.num_gpu_blocks_override,
                config.num_blocks,
                config.num_blocks * _pool_bytes_per_block(config),
                available_memory[worker_index],
                available_memory[worker_index] // _pool_bytes_per_block(config),
                len(config.kv_cache_groups),
                _cache_group_summary(config),
            )
        logger.info(
            "DFlash mixed cache plan ready: physical_blocks=%d (memory/address safe)",
            min(config.num_blocks for config in configs if config.kv_cache_groups),
        )
        return configs

    return plan


def _same_group_metadata(left: KVCacheGroupSpec, right: KVCacheGroupSpec) -> bool:
    return all(
        getattr(left, field.name) == getattr(right, field.name)
        for field in fields(left)
        if field.name not in ("layer_names", "is_eagle_group")
    )


def choose_dflash_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    original_groups: list[KVCacheGroupSpec],
    *,
    draft_layer_names: set[str],
    available_memory: int,
    max_num_blocks: int,
) -> list[KVCacheGroupSpec] | None:
    """Reduce cap-induced waste, keeping exact specs and target/draft ownership.

    The caller must first align and validate mixed DFlash's physical planes.
    ``available_memory`` and ``max_num_blocks`` are the minimum real worker
    budget and FIA block limit, respectively; this routine never raises them.
    Only widths up to the first one that can use the budget below the cap are
    considered. A candidate must strictly improve startup admission capacity;
    this estimate is not a prediction of runtime concurrency or throughput.
    """
    speculative = getattr(vllm_config, "speculative_config", None)
    if (
        not getattr(vllm_config, "use_v2_model_runner", False)
        or speculative is None
        or speculative.method != "dflash"
        or getattr(vllm_config.parallel_config, "pipeline_parallel_size", 1) != 1
        or not original_groups
    ):
        return None
    if any(type(spec) not in (FullAttentionSpec, SlidingWindowSpec, MambaSpec) for spec in kv_cache_spec.values()):
        return None
    if not draft_layer_names or not draft_layer_names <= kv_cache_spec.keys():
        raise ValueError("DFlash grouping requires an explicit, valid draft-layer set.")
    if any(not any(field.name == "is_eagle_group" for field in fields(group)) for group in original_groups):
        # Older upstream groups cannot preserve explicit draft ownership when
        # projected to workers. Keep their existing safe grouping.
        return None
    draft_types = {type(kv_cache_spec[name]) for name in draft_layer_names}
    if MambaSpec in draft_types:
        raise ValueError("DFlash grouping cannot mark target Mamba states as draft layers.")
    if draft_types != {FullAttentionSpec, SlidingWindowSpec} or not any(
        isinstance(spec, MambaSpec) for spec in kv_cache_spec.values()
    ):
        return None
    page_sizes = {spec.page_size_bytes for spec in kv_cache_spec.values()}
    if len(page_sizes) != 1 or min(available_memory, max_num_blocks) <= 0:
        return None
    page_size = next(iter(page_sizes))
    if page_size <= 0:
        return None
    original_names = [name for group in original_groups for name in group.layer_names]
    if len(original_names) != len(kv_cache_spec) or set(original_names) != kv_cache_spec.keys():
        raise ValueError("DFlash groups must contain every cache layer exactly once.")
    if any(kv_cache_spec[name] != group.kv_cache_spec for group in original_groups for name in group.layer_names):
        # Do not discard merged or UniformType per-layer metadata.
        return None

    block_limit = max_num_blocks
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        block_limit = min(block_limit, override)
    if block_limit < 2:
        return None
    original_width = max(len(group.layer_names) for group in original_groups)
    original_blocks = min(available_memory // (page_size * original_width), block_limit)
    if original_blocks < 2 or available_memory // (page_size * original_width) <= block_limit:
        return None

    # Group metadata, including enable_kv_transfer, is part of ownership.
    # Never merge a target/draft pair, even when its attention specs match.
    buckets: list[tuple[KVCacheGroupSpec, list[str], bool]] = []
    for group in original_groups:
        for name in group.layer_names:
            is_draft = name in draft_layer_names
            for template, names, bucket_is_draft in buckets:
                if is_draft == bucket_is_draft and _same_group_metadata(group, template):
                    names.append(name)
                    break
            else:
                buckets.append((group, [name], is_draft))

    per_request_blocks = {
        name: (spec.max_memory_usage_bytes(vllm_config) + page_size - 1) // page_size
        for name, spec in kv_cache_spec.items()
    }
    if any(count <= 0 for count in per_request_blocks.values()):
        return None
    best_request_blocks = sum(per_request_blocks[group.layer_names[0]] for group in original_groups)
    best_blocks = original_blocks
    best_groups: list[KVCacheGroupSpec] | None = None
    # Wider pools beyond this point cannot recover any more cap-limited bytes.
    max_width = min(
        (available_memory + page_size * block_limit - 1) // (page_size * block_limit),
        max(len(names) for _, names, _ in buckets),
    )
    for width in range(original_width + 1, max_width + 1):
        groups: list[KVCacheGroupSpec] = []
        for template, names, is_draft in buckets:
            num_groups = (len(names) + width - 1) // width
            groups.extend(
                replace(template, layer_names=names[index::num_groups], is_eagle_group=is_draft)
                for index in range(num_groups)
            )
        actual_width = max(len(group.layer_names) for group in groups)
        blocks = min(available_memory // (page_size * actual_width), block_limit)
        request_blocks = sum(per_request_blocks[group.layer_names[0]] for group in groups)
        # Account for the permanent null block, and compare exact ratios.
        if blocks > 1 and (blocks - 1) * best_request_blocks > (best_blocks - 1) * request_blocks:
            best_groups = groups
            best_blocks = blocks
            best_request_blocks = request_blocks
    return best_groups
