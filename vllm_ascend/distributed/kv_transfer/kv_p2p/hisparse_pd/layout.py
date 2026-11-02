# SPDX-License-Identifier: Apache-2.0
"""Translate existing HiSparse SOURCE/INDEXER views to byte descriptors."""

from __future__ import annotations

import torch

from .protocol import BF16_FORMAT, C8_FORMAT, INDEXER_FP8_FORMAT, LayerLayout, validate_layouts


def _fp8_indexer_layouts(name, caches, owner, tensor_config, config, group_id, block_size, model_role, allow_cpu):
    key, scale = caches
    if key.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32:
        raise ValueError("INDEXER requires FP8 E4M3FN key and FP32 scale, not MXFP8 or numeric conversion")
    if getattr(owner, "hisparse_indexer_payload_format", None) != INDEXER_FP8_FORMAT:
        raise ValueError("INDEXER byte format must be declared by its backend owner")
    if key.shape[:3] != scale.shape[:3]:
        raise ValueError("INDEXER key/scale token geometry differs")
    factor = owner.hisparse_block_stride
    if type(factor) is not int or factor <= 0:
        raise ValueError("INDEXER manager expansion factor must be positive")
    layouts, regions = [], {}
    for component, cache in (("key", key), ("scale", scale)):
        if cache.device.type != "npu" and not (allow_cpu and cache.device.type == "cpu"):
            raise ValueError("INDEXER must be on the Ascend device")
        if cache.ndim != 4 or cache.shape[1] != block_size:
            raise ValueError("INDEXER requires physical pages with complete token rows")
        row_bytes = cache.shape[2] * cache.shape[3] * cache.element_size()
        if (
            cache.stride(3) != 1
            or cache.stride(2) != cache.shape[3]
            or cache.stride(1) * cache.element_size() != row_bytes
        ):
            raise ValueError("INDEXER component rows must be dense")
        block_stride = factor * cache.stride(0) * cache.element_size()
        if block_stride != tensor_config.block_stride:
            raise ValueError("INDEXER component view differs from manager block stride")
        blocks = config.num_blocks
        if cache.shape[0] < (blocks - 1) * factor + 1:
            raise ValueError("INDEXER expanded view lacks manager pages")
        storage = cache.untyped_storage()
        end = cache.data_ptr() + (blocks - 1) * block_stride + block_size * row_bytes
        if end > storage.data_ptr() + storage.nbytes():
            raise ValueError("INDEXER component exceeds backing allocation")
        regions[(storage.data_ptr(), storage.nbytes())] = "hisparse_indexer"
        layouts.append(
            LayerLayout(
                name,
                group_id,
                "hisparse_indexer",
                block_size,
                row_bytes,
                cache.data_ptr(),
                block_stride,
                blocks,
                INDEXER_FP8_FORMAT,
                component,
                model_role,
            )
        )
    return layouts, regions


def build_layouts(
    kv_caches, kv_cache_config, forward_context, *, producer: bool, allow_cpu_indexer=False, include_draft=False
):
    """Return descriptors, deduplicated registration ranges and strong owners.

    SOURCE uses a raw host range on both endpoints. Producer SOURCE is copied
    into an explicit bounded HBM stage; only HBM allocations are registered.
    RESIDENT/HOT are local replicas, absent from the transfer protocol.
    """
    if not isinstance(producer, bool):
        raise TypeError("HiSparse P/D layout requires an explicit producer boolean")
    tensors = {name: config for config in kv_cache_config.kv_cache_tensors for name in config.layers}
    layouts = []
    regions = {}
    owners = []
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        role = str(getattr(group.role, "value", group.role))
        if not group.enable_kv_transfer:
            continue
        if role not in ("hisparse_source", "hisparse_indexer") or (group.is_eagle_group and not include_draft):
            raise ValueError("Draft handoff must be explicitly enabled with a complete restoration policy")
        block_size = group.kv_cache_spec.block_size
        for name in group.layer_names:
            owner = forward_context[name]
            declared = getattr(owner, "hisparse_pd_model_role", None)
            if include_draft and declared not in ("target", "draft"):
                raise ValueError("MTP transfer cache has no loaded ownership binding")
            model_role = declared or "target"
            if model_role == "draft" and not include_draft:
                raise ValueError("Draft transfer requires the explicit MTP prefix policy")
            if role == "hisparse_source" and bool(owner.hisparse_cache.draft_layer) != (model_role == "draft"):
                raise ValueError("SOURCE transfer owner differs from the common HiSparse handle")
            cache = kv_caches[name]
            if isinstance(cache, tuple):
                if role == "hisparse_indexer" and len(cache) == 2:
                    components, ranges = _fp8_indexer_layouts(
                        name,
                        cache,
                        forward_context[name],
                        tensors[name],
                        kv_cache_config,
                        group_id,
                        block_size,
                        model_role,
                        allow_cpu_indexer,
                    )
                    layouts.extend(components)
                    regions.update(ranges)
                    owners.extend(cache)
                    continue
                if len(cache) != 1:
                    raise ValueError("Unsupported INDEXER component tuple")
                cache = cache[0]
            payload_format = BF16_FORMAT
            if cache.dtype != torch.bfloat16:
                impl = getattr(forward_context[name], "impl", None)
                declared = getattr(impl, "hisparse_c8_payload_format", None)
                if (
                    role != "hisparse_source"
                    or cache.dtype != torch.float8_e4m3fn
                    or cache.shape[-1] != 656
                    or declared != C8_FORMAT
                ):
                    raise ValueError(
                        "HiSparse P/D requires BF16 or explicitly declared Ascend C8 SOURCE; INDEXER stays BF16"
                    )
                payload_format = C8_FORMAT
            if role == "hisparse_source":
                if cache.device.type != "cpu" or not cache.is_contiguous():
                    raise ValueError("SOURCE must be a contiguous packed host layer")
                handle = forward_context[name].hisparse_cache
                mapping = handle.runtime.cache_ops.host_mapping
                # An existing ACL mapping is borrowed, never re-registered or closed here.
                mapping.address_of(cache)  # Validate the existing mapping bounds and liveness; do not export its alias.
                base = cache.data_ptr()
                row_bytes = cache.shape[-1] * cache.element_size()
                num_blocks = kv_cache_config.hisparse_host_num_blocks
                block_stride = block_size * row_bytes
                if cache.numel() * cache.element_size() != num_blocks * block_stride:
                    raise ValueError("SOURCE shape disagrees with its host manager capacity")
                key = (mapping.host_address, mapping.nbytes)
                regions[key] = role
                owners.extend((cache, mapping))
            else:
                if cache.device.type != "npu" and not (allow_cpu_indexer and cache.device.type == "cpu"):
                    raise ValueError("INDEXER must be on the Ascend device")
                if cache.ndim != 4 or cache.shape[1] != block_size:
                    raise ValueError("INDEXER must have the Ascend [physical blocks, tokens, heads, dim] view")
                row_bytes = cache.shape[2] * cache.shape[3] * cache.element_size()
                if (
                    cache.stride(3) != 1
                    or cache.stride(2) != cache.shape[3]
                    or cache.stride(1) * cache.element_size() != row_bytes
                ):
                    raise ValueError("INDEXER token rows must be dense")
                base = cache.data_ptr()
                block_stride = tensors[name].block_stride
                num_blocks = kv_cache_config.num_blocks
                factor = forward_context[name].hisparse_block_stride
                if block_stride != factor * block_size * row_bytes:
                    raise ValueError("Expanded INDEXER view disagrees with manager block stride")
                storage = cache.untyped_storage()
                extent = (num_blocks - 1) * block_stride + block_size * row_bytes
                if base + extent > storage.data_ptr() + storage.nbytes():
                    raise ValueError("INDEXER manager view extends past its shared storage")
                key = (storage.data_ptr(), storage.nbytes())
                regions[key] = role
                owners.append(cache)
            layouts.append(
                LayerLayout(
                    name,
                    group_id,
                    role,
                    block_size,
                    row_bytes,
                    base,
                    block_stride,
                    num_blocks,
                    payload_format,
                    model_role=model_role,
                )
            )
    validate_layouts(layouts)
    if include_draft and {x.model_role for x in layouts} != {"target", "draft"}:
        raise ValueError("MTP transfer requires complete target and draft SOURCE/INDEXER")
    # Shared slabs are registered once; actual per-layer offsets stay in layouts.
    return layouts, regions, owners
