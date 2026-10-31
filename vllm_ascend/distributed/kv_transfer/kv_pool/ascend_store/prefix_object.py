# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ordinary component-major full-block objects for pooled PD.

This adapter describes the existing single-stage, single-context-shard wire
layout. It never assigns a content key to a partial component or to a layout
it cannot prove equivalent to ordinary layerwise pooling.
"""

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    get_physical_layer_index,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_writer import LayerWriteRange


def component_prefix_id(cache, binding, spec, request, *, group_block_size):
    """Content identity for a component which cannot use the legacy packing.

    CP placement and state shapes are already explicit in the cache slice.
    Physical addresses, engine lifetime and write-attempt IDs never enter the
    identity. A partial page/checkpoint without its own prefix hash can only
    be reused for the same complete request inputs and boundary.
    """
    compression = getattr(spec, "tokens_per_state", 1)
    if cache.format_id.startswith("mla-page-v2:"):
        rows = json.loads(cache.format_id.split(":", 1)[1])["rows"]
        start, end = cache.token_start * rows * compression, cache.token_end * rows * compression
    elif cache.format_id.startswith("mamba-v2:") or '"family":"circular-v2"' in cache.format_id:
        start, end = cache.token_end - 1, cache.token_end
    else:
        start, end = cache.token_start * compression, cache.token_end * compression
    index = (end - 1) // group_block_size
    hashes = request.group_hashes[binding.group_id]
    block_end = (index + 1) * group_block_size
    if end <= request.token_end and block_end <= request.token_end and index < len(hashes):
        content = ("prefix", hashes[index])
    else:
        content = ("request", request.request_digest, request.token_end)
    layout = binding.to_wire()
    # Group numbering and the external key builder handle topology naming;
    # retain actual byte geometry to avoid aliasing equal-size packed formats.
    descriptor = (content, start, end, asdict(cache), layout)
    return (
        "pool-component-v1:"
        + hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )


def full_prefix_objects(
    bindings,
    pages,
    request,
    profile,
    *,
    rank,
    stage,
    group_block_sizes,
    base_layers,
    layer_offset=0,
    state_rank=None,
    excluded_groups=frozenset(),
    global_layers=None,
    schema_qualified=False,
):
    # Imports are deferred to avoid the metadata -> layout -> metadata cycle.
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_metadata import PoolObjectRef, PoolObjectSlice

    grouped = defaultdict(list)
    for components in bindings.values():
        for binding in components:
            grouped[binding.group_id].append(binding)
    objects, layers, refs, covered = {}, defaultdict(list), [], set()
    page_index = defaultdict(list)
    for index, page in enumerate(pages):
        page_index[page.cache.layer, page.cache.component, page.address].append((index, page))
    builder = profile.key_identity.bind()
    for group, components in grouped.items():
        if group in excluded_groups:
            continue
        object_rank = (
            state_rank
            if state_rank is not None and any(item.spec_type.rsplit(".", 1)[-1] == "MambaSpec" for item in components)
            else rank
        )
        # Ordinary pooling sorts physical layers, then names, then tensor
        # components. In particular target and draft layer 0 are distinct.
        components.sort(
            key=lambda item: (get_physical_layer_index(item.layer, base_layers), item.layer, item.component)
        )
        offsets, size = {}, 0
        for binding in components:
            offsets[binding.layer, binding.component] = size
            size += math.prod(binding.shape[1:]) * binding.element_size * binding.block_size_scale
        # Ordinary GVA pooling reserves per-layer bytes for the model-wide
        # layer count even when a hybrid group contains only some layers.
        # The unused tail is not additional logical cache coverage.
        local_layers = len({get_physical_layer_index(item.layer, base_layers) for item in components})
        all_layers = (
            global_layers
            if global_layers is not None
            else max((get_physical_layer_index(name, base_layers) + 1 for name in bindings), default=0)
        )
        per_layer = (size + local_layers - 1) // local_layers
        origin = layer_offset * per_layer
        size = max(origin + size, per_layer * max(base_layers, all_layers, local_layers))
        offsets = {key: origin + offset for key, offset in offsets.items()}
        block_size = group_block_sizes[group]
        for index, block in enumerate(request.block_ids[group]):
            start, end = index * block_size, (index + 1) * block_size
            if start < request.token_start or end > request.token_end or index >= len(request.group_hashes[group]):
                continue
            candidates = []
            writes = defaultdict(list)
            valid = True
            for binding in components:
                physical_bytes = math.prod(binding.shape[1:]) * binding.element_size
                # A strided view may include a packed neighbouring component.
                # Such objects require the ordinary pool's segment recipe,
                # not an inferred contiguous slice.
                if binding.page_span != physical_bytes:
                    valid = False
                    break
                scale = binding.block_size_scale
                first = binding.address + block * scale * binding.byte_strides[0]
                selected = []
                for slot in range(scale):
                    address = first + slot * binding.byte_strides[0]
                    for index_in_pages, page in page_index.get((binding.layer, str(binding.component), address), ()):
                        cache = page.cache
                        if cache.format_id.startswith("mamba-v2:") or '"family":"circular-v2"' in cache.format_id:
                            if cache.token_end != end:
                                continue
                        if cache.span_bytes == physical_bytes:
                            offset = offsets[binding.layer, binding.component] + slot * physical_bytes
                            selected.append((index_in_pages, PoolObjectSlice(cache, offset)))
                if len(selected) != scale or len({item.offset for _, item in selected}) != scale:
                    valid = False
                    break
                candidates.extend(selected)
                for slot in range(scale):
                    offset = offsets[binding.layer, binding.component] + slot * physical_bytes
                    writes[binding.layer].append((first + slot * binding.byte_strides[0], offset, physical_bytes))
            if not valid:
                continue
            object_id = request.group_hashes[group][index]
            if schema_qualified:
                # Equal token hashes and byte counts do not prove equal
                # payloads: HND/NHD, TP head ranges and MTP component counts
                # can differ. Qualify the content identity with the actual
                # shared format, as StoreLayout does upstream. The existing
                # backend key builder still owns model/group/rank naming.
                schema = (
                    size,
                    tuple((asdict(part.cache), part.offset) for _, part in candidates),
                )
                fingerprint = hashlib.sha256(
                    json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                object_id = f"pool-full-v1:{fingerprint}:{object_id}"
            key = builder.make_full_key(group, object_id, object_rank, stage)
            if key in objects:
                # Duplicate physical rows for one immutable prefix need only
                # one complete write; all equivalent pages are covered.
                covered.update(page_index for page_index, _ in candidates)
                continue
            objects[key] = size
            for layer, ranges in writes.items():
                layers[layer].extend(
                    LayerWriteRange(key, address, offset, length) for address, offset, length in ranges
                )
            refs.append(
                PoolObjectRef(
                    profile.profile_id,
                    group,
                    object_id,
                    object_rank,
                    stage,
                    size,
                    tuple(part for _, part in candidates),
                )
            )
            covered.update(page_index for page_index, _ in candidates)
    return objects, layers, refs, covered
