# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import copy
import math
from dataclasses import replace
from functools import wraps

from vllm.logger import logger
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
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
    arena_offsets = {}
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


def wrap_dflash_cache_planner(original_planner):
    """Re-plan before allocation using both the real budget and FIA limits."""

    @wraps(original_planner)
    def plan(vllm_config, kv_cache_specs, available_memory):
        configs = original_planner(vllm_config, kv_cache_specs, available_memory)
        if not uses_mixed_dflash_cache(vllm_config):
            return configs
        if len(configs) != len(available_memory) or len(kv_cache_specs) != len(available_memory):
            raise ValueError("Mixed DFlash requires a cache plan and memory budget for every worker.")
        limits = []
        active_configs = []
        for config, budget in zip(configs, available_memory):
            specs = _layer_specs(config)
            if not specs or not any(isinstance(spec, MambaSpec) for spec in specs.values()):
                continue
            aligned = align_dflash_cache_specs(vllm_config, specs)
            if any(aligned[name] != spec for name, spec in specs.items()):
                raise ValueError("Mixed DFlash cache specs changed after alignment; refusing unsafe allocation.")
            active_configs.append(config)
            limits.extend((config.num_blocks, budget // _pool_bytes_per_block(config)))
            for spec in specs.values():
                if isinstance(spec, MambaSpec):
                    continue
                dtype_bytes = get_dtype_size(spec.dtype)
                limits.append(
                    get_dflash_fia_safe_num_blocks(
                        storage_block_size=spec.block_size,
                        key_row_bytes=spec.num_kv_heads * spec.head_size * dtype_bytes,
                        value_row_bytes=spec.num_kv_heads * spec.head_size_v * dtype_bytes,
                        key_element_bytes=dtype_bytes,
                        value_element_bytes=dtype_bytes,
                    )
                )
        if not active_configs:
            return configs
        safe_blocks = min(limits)
        if safe_blocks < 2:
            raise ValueError("No safe mixed DFlash KV cache blocks fit the available memory.")
        if any(config.num_blocks > safe_blocks for config in active_configs):
            logger.warning(
                "DFlash mixed cache guard: physical_blocks=%d -> %d; "
                "applying the profiled memory budget and FIA address limits before allocation.",
                max(config.num_blocks for config in active_configs),
                safe_blocks,
            )
            safe_config = copy.copy(vllm_config)
            safe_config.cache_config = copy.copy(vllm_config.cache_config)
            safe_config.cache_config.num_gpu_blocks_override = safe_blocks
            # Re-run admission, auto-fit and descriptor construction together.
            # In particular, offsets/layer strides on main depend on num_blocks.
            configs = original_planner(safe_config, kv_cache_specs, available_memory)
        if len(configs) != len(available_memory):
            raise ValueError("Replanned mixed DFlash cache is missing worker plans.")
        for config, budget in zip(configs, available_memory):
            specs = _layer_specs(config)
            if not specs:
                continue
            if config.num_blocks > safe_blocks or config.num_blocks * _pool_bytes_per_block(config) > budget:
                raise ValueError("Replanned mixed DFlash KV cache exceeds its memory/address limit.")
            aligned = align_dflash_cache_specs(vllm_config, specs)
            if any(aligned[name] != spec for name, spec in specs.items()):
                raise ValueError("Replanned mixed DFlash KV cache lost its aligned layout.")
        logger.info(
            "DFlash mixed cache plan ready: physical_blocks=%d (memory/address safe)",
            min(config.num_blocks for config in configs if config.kv_cache_groups),
        )
        return configs

    return plan
