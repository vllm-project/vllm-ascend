from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import regex as re
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    get_layerwise_protocol,
)
from vllm_ascend.utils import get_kv_cache_tensor_layers

_NUM_SHARED_BUFFERS = "layerwise_num_shared_buffers"
_PREFETCH_LAYERS = "layerwise_prefetch_layers"
_INDEPENDENT_LAYERS = "layerwise_independent_layers"
_DEFAULT_MAX_PREFETCH_LAYERS = 8
_INDEXER_CACHE_SUFFIX = ".indexer.k_cache"


def _cache_component_name(layer_name: str) -> str:
    """Name a cache component independently of its physical layer number."""
    return layer_name.split(".self_attn.", 1)[-1]


def _is_deepseek_v4_spec(spec: KVCacheSpec) -> bool:
    return getattr(spec, "model_version", None) == "deepseek_v4"


def get_layerwise_physical_layer_index(layer_name: str, base_layers: int) -> int:
    match = re.search(
        r"(?:^|\.)mtp(?:\.layers)?\.(\d+)(?:\.|$)",
        layer_name,
    )
    if match:
        return base_layers + int(match.group(1))
    match = re.search(r"layers\.(\d+)", layer_name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", layer_name)
    return int(match.group(1)) if match else 0


@dataclass(frozen=True)
class LayerwiseCacheLayout:
    num_shared_buffers: int
    num_prefetch_layers: int
    independent_layers: list[int]
    prefetch_layer_map: dict[int, int]
    storage_indices: list[list[int]]
    has_layer_reuse: bool


@dataclass(frozen=True)
class NamedKVCacheSpec:
    layer_name: str
    spec: KVCacheSpec


@dataclass(frozen=True)
class LayerwiseLayerCacheSpecs:
    main: NamedKVCacheSpec
    indexer: NamedKVCacheSpec | None = None
    extra_main_specs: tuple[NamedKVCacheSpec, ...] = ()


@dataclass(frozen=True)
class LayerwiseReuseLayout:
    layer_cache_specs: dict[int, LayerwiseLayerCacheSpecs]
    buffer_slots: tuple[tuple[int, ...], ...]
    prefetch_layer_map: dict[int, int]
    independent_layers: list[int]
    num_prefetch_layers: int
    has_layer_reuse: bool


def get_layerwise_reuse_config(kv_transfer_config: Any) -> dict[str, Any] | None:
    """Return the extra config of the layerwise-reuse connector, if any.

    A connector opts into layerwise reuse when its backend carries a
    layerwise protocol and the protocol accepts the connector's extra
    config. Both checks resolve through the backend registry — the generic
    layer never names the protocol or the backend.
    """
    if kv_transfer_config is None:
        return None

    connector_name = getattr(kv_transfer_config, "kv_connector", None)
    root_extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    if connector_name in ("AscendStoreConnector", "MooncakeConnectorStoreV1"):
        connector_configs = [
            {
                "kv_connector": connector_name,
                "kv_connector_extra_config": root_extra_config,
            }
        ]
    elif connector_name == "MultiConnector":
        connector_configs = root_extra_config.get("connectors", [])
    else:
        return None

    for connector_config in connector_configs:
        if not isinstance(connector_config, dict):
            continue
        if connector_config.get("kv_connector") not in (
            "AscendStoreConnector",
            "MooncakeConnectorStoreV1",
        ):
            continue
        extra_config = connector_config.get("kv_connector_extra_config") or {}
        protocol = get_layerwise_protocol(str(extra_config.get("backend", "mooncake")))
        if protocol is None:
            continue
        layerwise_config = protocol.extract_layout_config(extra_config)
        if layerwise_config is not None:
            return layerwise_config
    return None


def _parse_int_config(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        return int(value)
    except (TypeError, ValueError) as err:
        raise TypeError(f"{name} must be an integer, got {value!r}") from err


def build_layerwise_cache_layout(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> LayerwiseCacheLayout:
    shared_buffers_value = extra_config.get(_NUM_SHARED_BUFFERS) if extra_config else None
    if shared_buffers_value is None:
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        num_shared_buffers = num_layers
    else:
        num_shared_buffers = _parse_int_config(shared_buffers_value, _NUM_SHARED_BUFFERS)
        if num_shared_buffers < 1:
            raise ValueError(f"{_NUM_SHARED_BUFFERS} must be at least 1")

    prefetch_value = extra_config.get(_PREFETCH_LAYERS) if extra_config else None
    if prefetch_value is None:
        num_prefetch_layers = min(num_shared_buffers, _DEFAULT_MAX_PREFETCH_LAYERS)
    else:
        num_prefetch_layers = _parse_int_config(prefetch_value, _PREFETCH_LAYERS)
        if num_prefetch_layers < 1:
            raise ValueError(f"{_PREFETCH_LAYERS} must be at least 1")

    independent_value = extra_config.get(_INDEPENDENT_LAYERS) if extra_config else None
    if independent_value is None:
        layer_indices = [0]
    elif isinstance(independent_value, str) and independent_value.strip().lower() == "all":
        layer_indices = list(range(num_layers))
    elif isinstance(independent_value, list):
        layer_indices = [_parse_int_config(index, _INDEPENDENT_LAYERS) for index in independent_value]
    else:
        raise TypeError(f"{_INDEPENDENT_LAYERS} must be a list of integers or 'all'")

    normalized_indices = set()
    for layer_index in layer_indices:
        if layer_index < 0:
            layer_index += num_layers
        if layer_index < 0 or layer_index >= num_layers:
            raise ValueError(
                f"{_INDEPENDENT_LAYERS} contains out-of-range layer index "
                f"{layer_index}; valid range is [0, {num_layers - 1}]"
            )
        normalized_indices.add(layer_index)
    independent_layers = sorted(normalized_indices)

    independent_layer_set = set(independent_layers)
    reused_layers = [index for index in range(num_layers) if index not in independent_layer_set]
    has_layer_reuse = len(reused_layers) > num_shared_buffers
    prefetch_layer_map = {
        reused_layers[next_index]: reused_layers[next_index - num_shared_buffers]
        for next_index in range(num_shared_buffers, len(reused_layers))
    }
    storage_indices = [[layer] for layer in independent_layers]
    for slot in range(num_shared_buffers):
        members = list(range(slot, len(reused_layers), num_shared_buffers))
        if members:
            storage_indices.append([reused_layers[index] for index in members])

    return LayerwiseCacheLayout(
        num_shared_buffers=num_shared_buffers,
        num_prefetch_layers=num_prefetch_layers,
        independent_layers=independent_layers,
        prefetch_layer_map=prefetch_layer_map,
        storage_indices=storage_indices,
        has_layer_reuse=has_layer_reuse,
    )


def get_layerwise_kv_cache_specs(
    kv_cache_config: KVCacheConfig,
) -> dict[str, KVCacheSpec]:
    """Expand group specs into a cache spec for every logical layer."""
    layer_specs: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        for layer_name in group.layer_names:
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                layer_specs[layer_name] = group_spec.kv_cache_specs[layer_name]
            else:
                layer_specs[layer_name] = group_spec
    return layer_specs


def build_layerwise_reuse_layout(
    layer_specs: dict[str, KVCacheSpec],
    base_layers: int,
    extra_config: dict[str, Any],
) -> LayerwiseReuseLayout:
    """Build reusable physical-layer slots by grouping layers on their main cache spec."""
    named_specs_by_layer: dict[int, list[NamedKVCacheSpec]] = {}
    for layer_name, layer_spec in layer_specs.items():
        physical_layer = get_layerwise_physical_layer_index(layer_name, base_layers)
        named_specs_by_layer.setdefault(physical_layer, []).append(NamedKVCacheSpec(layer_name, layer_spec))

    physical_layers = sorted(named_specs_by_layer)
    base_layout = build_layerwise_cache_layout(len(physical_layers), extra_config)
    independent_layers = [physical_layers[index] for index in base_layout.independent_layers]
    independent_layer_set = set(independent_layers)

    layer_cache_specs: dict[int, LayerwiseLayerCacheSpecs] = {}
    for physical_layer, named_specs in named_specs_by_layer.items():
        if len(named_specs) == 1:
            layer_cache_specs[physical_layer] = LayerwiseLayerCacheSpecs(main=named_specs[0])
            continue

        indexer_specs = [spec for spec in named_specs if spec.layer_name.endswith(_INDEXER_CACHE_SUFFIX)]
        main_specs = [spec for spec in named_specs if not spec.layer_name.endswith(_INDEXER_CACHE_SUFFIX)]
        if len(main_specs) < 1:
            raise ValueError(
                f"Physical layer {physical_layer} has no main cache spec; "
                f"got {[spec.layer_name for spec in named_specs]}."
            )
        # Select '.attn' as main spec, rest as extra
        main_spec = next((s for s in main_specs if s.layer_name.endswith(".attn")), main_specs[0])
        extra_specs = tuple(s for s in main_specs if s is not main_spec)
        indexer_spec = indexer_specs[0] if indexer_specs else None
        layer_cache_specs[physical_layer] = LayerwiseLayerCacheSpecs(
            main=main_spec,
            indexer=indexer_spec,
            extra_main_specs=extra_specs,
        )

    is_deepseek_v4 = any(_is_deepseek_v4_spec(spec) for spec in layer_specs.values())
    signature_buckets: list[tuple[Any, list[int]]] = []
    for physical_layer in physical_layers:
        if physical_layer in independent_layer_set:
            continue
        # V4 has MLA, SWA, compressor state and indexer cache components.
        # Reusing a slot across different component sets would either drop a
        # component or alias two live components of the same request.
        if is_deepseek_v4:
            signature = tuple(
                sorted(
                    (
                        (_cache_component_name(named.layer_name), named.spec)
                        for named in named_specs_by_layer[physical_layer]
                    ),
                    key=lambda component: component[0],
                )
            )
        else:
            # GLM/SFA can share the main component even if some layers lack
            # an indexer; the indexer is planned separately below.
            signature = layer_cache_specs[physical_layer].main.spec
        for bucket_signature, bucket_layers in signature_buckets:
            if signature == bucket_signature:
                bucket_layers.append(physical_layer)
                break
        else:
            signature_buckets.append((signature, [physical_layer]))

    buffer_slots: list[tuple[int, ...]] = [(layer,) for layer in independent_layers]
    prefetch_layer_map: dict[int, int] = {}
    for _, bucket_layers in signature_buckets:
        num_shared_buffers = min(base_layout.num_shared_buffers, len(bucket_layers))
        for buffer_index in range(num_shared_buffers):
            layers_sharing_buffer = tuple(bucket_layers[buffer_index::num_shared_buffers])
            buffer_slots.append(layers_sharing_buffer)
            for owner_index in range(1, len(layers_sharing_buffer)):
                prefetch_layer_map[layers_sharing_buffer[owner_index]] = layers_sharing_buffer[owner_index - 1]

    if prefetch_layer_map:
        unsupported_specs = [
            named_spec
            for named_specs in named_specs_by_layer.values()
            for named_spec in named_specs
            if not isinstance(named_spec.spec, AttentionSpec)
        ]
        if unsupported_specs:
            named_spec = unsupported_specs[0]
            raise NotImplementedError(
                "Layerwise KV cache reuse supports attention cache specs only; "
                f"{named_spec.layer_name} uses {type(named_spec.spec).__name__}."
            )

    return LayerwiseReuseLayout(
        layer_cache_specs=layer_cache_specs,
        buffer_slots=tuple(buffer_slots),
        prefetch_layer_map=prefetch_layer_map,
        independent_layers=independent_layers,
        num_prefetch_layers=base_layout.num_prefetch_layers,
        has_layer_reuse=bool(prefetch_layer_map),
    )


def apply_layerwise_kv_cache_plan(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> None:
    """Rewrite logical layer tensors to use shared physical KV buffers."""
    extra_config = get_layerwise_reuse_config(vllm_config.kv_transfer_config)
    if extra_config is None:
        return

    old_tensors = kv_cache_config.kv_cache_tensors
    if not old_tensors:
        return

    base_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    layer_specs = get_layerwise_kv_cache_specs(kv_cache_config)
    reuse_layout = build_layerwise_reuse_layout(
        layer_specs,
        base_layers,
        extra_config,
    )
    actual_layers = len(reuse_layout.layer_cache_specs)
    if not reuse_layout.has_layer_reuse:
        return
    if actual_layers < base_layers:
        logger.warning(
            "Layer reuse expected at least %d layers, got %d; skip tensor merge.",
            base_layers,
            actual_layers,
        )
        return
    if actual_layers > base_layers:
        logger.info(
            "Layer reuse includes %d base and %d MTP/spec-decode layer(s).",
            base_layers,
            actual_layers - base_layers,
        )

    # vLLM describes multiple contiguous layer regions in one backing
    # allocation. Reuse replaces that placement before any storage exists;
    # each new descriptor owns one physical slot whose layers alias it.
    seen_layers: set[str] = set()
    for tensor in old_tensors:
        for layer_idx, layer_name in enumerate(get_kv_cache_tensor_layers(tensor)):
            spec = layer_specs[layer_name]
            layer_size = kv_cache_config.num_blocks * spec.page_size_bytes
            start = tensor.offset + layer_idx * tensor.layer_stride
            if tensor.block_stride != spec.page_size_bytes:
                raise NotImplementedError("Layerwise KV cache reuse requires contiguous per-layer pages.")
            if start < 0 or start + layer_size > tensor.size:
                raise ValueError(f"Layerwise KV cache descriptor for {layer_name} exceeds its backing allocation.")
            if layer_name in seen_layers:
                raise ValueError(f"Duplicate layerwise KV cache descriptor for {layer_name}.")
            seen_layers.add(layer_name)
    if seen_layers != set(layer_specs):
        raise ValueError("Layerwise KV cache descriptors must cover every cache spec.")

    def _merge_specs(named_specs: list[NamedKVCacheSpec]) -> None:
        shared_by = [named_spec.layer_name for named_spec in named_specs]
        reference_spec = layer_specs[shared_by[0]]
        if any(layer_specs[layer_name] != reference_spec for layer_name in shared_by[1:]):
            raise ValueError(
                "Layers sharing layerwise KV buffers must have identical cache specs for every named cache spec."
            )
        new_tensors.append(
            KVCacheTensor(
                layers=shared_by,
                size=kv_cache_config.num_blocks * reference_spec.page_size_bytes,
                layer_stride=0,
                block_stride=reference_spec.page_size_bytes,
                offset=0,
            )
        )

    if any(_is_deepseek_v4_spec(spec) for spec in layer_specs.values()):
        # The V4 model runner materializes every descriptor from one backing.
        # Keep that contract while giving each distinct cache component its
        # own range within every reusable physical-layer slot. In particular,
        # do not discard the extra_main_specs (SWA and compressor state).
        layer_groups = {
            name: group_idx
            for group_idx, group in enumerate(kv_cache_config.kv_cache_groups)
            for name in group.layer_names
        }
        component_descriptors: list[tuple[list[str], KVCacheSpec]] = []
        for slot in reuse_layout.buffer_slots:
            shared_components: dict[tuple[int, str], list[str]] = {}
            for layer in slot:
                specs = reuse_layout.layer_cache_specs[layer]
                for named in (specs.main, *specs.extra_main_specs, *([specs.indexer] if specs.indexer else [])):
                    key = (layer_groups[named.layer_name], _cache_component_name(named.layer_name))
                    shared_components.setdefault(key, []).append(named.layer_name)
            for names in shared_components.values():
                spec = layer_specs[names[0]]
                if any(layer_specs[name] != spec for name in names[1:]):
                    raise ValueError("DeepSeek-V4 shared cache components must have identical specs.")
                component_descriptors.append((names, spec))

        backing_size = kv_cache_config.num_blocks * sum(spec.page_size_bytes for _, spec in component_descriptors)
        original_backing_size = max(tensor.size for tensor in old_tensors)
        if backing_size > original_backing_size:
            raise ValueError(
                "DeepSeek-V4 layerwise plan exceeds the original KV cache budget: "
                f"{backing_size} > {original_backing_size} bytes."
            )
        offset = 0
        v4_tensors: list[KVCacheTensor] = []
        for names, spec in component_descriptors:
            v4_tensors.append(
                KVCacheTensor(
                    layers=names,
                    size=backing_size,
                    layer_stride=0,
                    block_stride=spec.page_size_bytes,
                    offset=offset,
                )
            )
            offset += kv_cache_config.num_blocks * spec.page_size_bytes
        kv_cache_config.kv_cache_tensors = v4_tensors
        logger.info(
            "DeepSeek-V4 layerwise KV cache reuse planned %d components in %d physical slots (%d bytes).",
            len(v4_tensors),
            len(reuse_layout.buffer_slots),
            backing_size,
        )
        return

    new_tensors: list[KVCacheTensor] = []
    for slot in reuse_layout.buffer_slots:
        _merge_specs([reuse_layout.layer_cache_specs[layer].main for layer in slot])
        indexer_specs: list[NamedKVCacheSpec] = []
        for layer in slot:
            indexer = reuse_layout.layer_cache_specs[layer].indexer
            if indexer is not None:
                indexer_specs.append(indexer)
        if indexer_specs:
            _merge_specs(indexer_specs)
    kv_cache_config.kv_cache_tensors = new_tensors
    logger.info(
        "Layerwise KV cache reuse merged %d descriptors into %d descriptors using %d buffer assignments.",
        len(old_tensors),
        len(new_tensors),
        len(reuse_layout.buffer_slots),
    )
