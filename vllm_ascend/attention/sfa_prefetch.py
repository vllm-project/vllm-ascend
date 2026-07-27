from dataclasses import dataclass
from functools import lru_cache
from typing import Any


def get_layer_index(layer_name: str) -> int:
    """Extract the index from a model.layers.<index> module path."""
    parts = layer_name.split(".")
    for index, part in enumerate(parts):
        if part == "layers" and index + 1 < len(parts):
            try:
                return int(parts[index + 1])
            except ValueError:
                pass
    return -1


def get_layer_name_with_index(layer_name: str, target_index: int) -> str:
    """Return a layer path with its model.layers index replaced."""
    parts = layer_name.split(".")
    for index, part in enumerate(parts):
        if part == "layers" and index + 1 < len(parts):
            parts[index + 1] = str(target_index)
            return ".".join(parts)
    return layer_name


@dataclass(frozen=True)
class SfaPrefetchRole:
    target_layer_ids: tuple[int, ...] = ()
    buffer_index: int = -1

    @property
    def is_producer(self) -> bool:
        return bool(self.target_layer_ids)


@dataclass(frozen=True)
class SfaPrefetchPlan:
    topk_size: int
    max_prefetch_layers: int
    roles: tuple[SfaPrefetchRole, ...]

    def get_role(self, layer_id: int) -> SfaPrefetchRole:
        if 0 <= layer_id < len(self.roles):
            return self.roles[layer_id]
        return SfaPrefetchRole()


def _get_config_value(configs: tuple[Any, ...], name: str, default: Any) -> Any:
    for config in configs:
        value = getattr(config, name, None) if config is not None else None
        if value is not None:
            return value
    return default


@lru_cache(maxsize=32)
def _build_plan(
    layer_types: tuple[str, ...],
    topk_size: int,
    first_dense_layers: int,
    moe_layer_frequency: int,
    num_routed_experts: int,
) -> SfaPrefetchPlan:
    roles = [SfaPrefetchRole() for _ in layer_types]
    if (
        topk_size <= 0
        or num_routed_experts <= 0
        or moe_layer_frequency <= 0
    ):
        return SfaPrefetchPlan(topk_size, 0, tuple(roles))

    def is_moe_layer(layer_id: int) -> bool:
        return (
            layer_id >= first_dense_layers
            and (layer_id - first_dense_layers) % moe_layer_frequency == 0
        )

    max_prefetch_layers = 0
    for full_layer_id, layer_type in enumerate(layer_types):
        if layer_type != "full":
            continue
        shared_layer_ids = []
        next_layer_id = full_layer_id + 1
        while next_layer_id < len(layer_types) and layer_types[next_layer_id] == "shared":
            shared_layer_ids.append(next_layer_id)
            next_layer_id += 1
        group_layer_ids = (full_layer_id, *shared_layer_ids)
        producer_layer_id = next(
            (layer_id for layer_id in group_layer_ids if is_moe_layer(layer_id)),
            None,
        )
        if producer_layer_id is None:
            continue
        target_layer_ids = tuple(
            layer_id for layer_id in shared_layer_ids if layer_id > producer_layer_id
        )
        if not target_layer_ids:
            continue
        roles[producer_layer_id] = SfaPrefetchRole(
            target_layer_ids=target_layer_ids
        )
        for buffer_index, layer_id in enumerate(target_layer_ids):
            roles[layer_id] = SfaPrefetchRole(buffer_index=buffer_index)
        max_prefetch_layers = max(max_prefetch_layers, len(target_layer_ids))

    return SfaPrefetchPlan(topk_size, max_prefetch_layers, tuple(roles))


def build_sfa_prefetch_plan(
    hf_config: Any,
    hf_text_config: Any,
) -> SfaPrefetchPlan:
    """Build the per-layer grouped-prefetch roles from model structure."""
    configs = (hf_text_config, hf_config)
    indexer_types = _get_config_value(configs, "indexer_types", None)
    if indexer_types is None:
        return SfaPrefetchPlan(0, 0, ())
    layer_types = tuple(str(layer_type).lower() for layer_type in indexer_types)
    return _build_plan(
        layer_types,
        int(_get_config_value(configs, "index_topk", 0) or 0),
        int(_get_config_value(configs, "first_k_dense_replace", 0) or 0),
        int(_get_config_value(configs, "moe_layer_freq", 1) or 1),
        int(_get_config_value(configs, "n_routed_experts", 0) or 0),
    )
