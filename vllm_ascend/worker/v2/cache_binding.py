# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext

from vllm.logger import logger
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import allocate_kv_cache

from vllm_ascend.ascend_config import KVPPConfig, get_ascend_config
from vllm_ascend.attention.cache_components import bind_component
from vllm_ascend.patch.worker.patch_bind_kv_cache import bind_kv_cache_to_layers


def _resolve_components(config, context, kernel_block_sizes):
    placements = {}
    for tensor in config.kv_cache_tensors:
        if tensor.host_resident or tensor.layer_stride <= 0 or tensor.block_stride <= 0:
            return None
        for index, name in enumerate(tensor.layers):
            if name in placements:
                return None
            placements[name] = (tensor, tensor.offset + index * tensor.layer_stride)
    if len({tensor.size for tensor, _ in placements.values()}) != 1:
        return None

    plans = {}
    for group_id, group in enumerate(config.kv_cache_groups):
        if group_id >= len(kernel_block_sizes):
            return None
        for name in group.layer_names:
            if name not in placements or name not in context:
                return None
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            describe = getattr(context[name].get_attn_backend(), "describe_cache_components", None)
            if describe is None:
                return None
            placement, start = placements[name]
            blocks = config.num_blocks_of(placement)
            if blocks <= 0 or placement.block_stride < spec.page_size_bytes:
                return None
            components = describe(spec, blocks, kernel_block_sizes[group_id], placement.block_stride)
            if not components:
                return None
            span = (blocks - 1) * placement.block_stride + spec.page_size_bytes
            if start < 0 or start + span > placement.size:
                return None
            for component in components:
                itemsize = component.dtype.itemsize
                if (
                    (start + component.offset_bytes) % itemsize
                    or any(s % itemsize for s in component.strides_bytes)
                    or component.end_bytes() > span
                ):
                    return None
            plans[name] = (start, span, components)
    return plans if set(plans) == set(placements) else None


def init_component_kv_cache(
    forward_context,
    kv_cache_config,
    device,
    kernel_block_sizes,
    vllm_config,
    kv_cache_allocation_context=None,
):
    """Bind backend components when all cache owners support this path."""
    if (
        vllm_config.kv_transfer_config is not None
        or vllm_config.attention_config.hisparse_config is not None
        or vllm_config.speculative_config is not None
        or vllm_config.cache_config.use_replayssm
        or vllm_config.parallel_config.prefill_context_parallel_size > 1
        or vllm_config.parallel_config.decode_context_parallel_size > 1
        or KVPPConfig.from_vllm_config(vllm_config).size > 1
        or get_ascend_config().sparse_kv_offload_config.enabled
        or not kv_cache_config.kv_cache_tensors
    ):
        return None
    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    if not layout.is_layer_compact or not layout.is_block_compact:
        return None
    plans = _resolve_components(kv_cache_config, forward_context, kernel_block_sizes)
    if not plans:
        return None

    with kv_cache_allocation_context or nullcontext():
        manager_views = allocate_kv_cache(kv_cache_config, device, layout)
    runtime_views = {}
    for name, (start, span, components) in plans.items():
        manager_view = manager_views[name]
        if manager_view.storage_offset() * manager_view.element_size() != start:
            raise ValueError(f"Unexpected upstream KV placement for {name}")
        runtime_views[name] = tuple(bind_component(manager_view, c, span) for c in components)
    bind_kv_cache_to_layers(runtime_views, forward_context, kv_cache_groups=kv_cache_config.kv_cache_groups)
    logger.info("Bound %d cache layers from upstream allocation using backend component layouts", len(plans))
    return manager_views
