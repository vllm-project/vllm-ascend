# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select the layerwise strategy from the public switch and model topology."""

from collections.abc import Mapping
from enum import Enum
from typing import Any


class KVPoolTransferMode(Enum):
    BULK = "bulk"
    PHYSICAL_LAYER = "physical_layer"
    CROSS_LAYER_KV = "cross_layer_kv"


def config_value(config: Any, name: str, default: Any = None) -> Any:
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def text_model_config(vllm_config: Any) -> Any:
    model_config = getattr(vllm_config, "model_config", None)
    config = getattr(model_config, "hf_text_config", None)
    if config is None:
        config = getattr(model_config, "hf_config", None)
    return config_value(config, "text_config", None) or config


def kv_source_layer_ids(hf_config: Any) -> tuple[int, ...]:
    num_layers = config_value(hf_config, "num_hidden_layers")
    source_ids = config_value(hf_config, "kv_source_layer_ids")
    if type(num_layers) is not int or num_layers <= 0 or not isinstance(source_ids, (list, tuple)):
        return ()
    if not source_ids or any(type(source) is not int or not 0 <= source < num_layers for source in source_ids):
        return ()
    if any(left >= right for left, right in zip(source_ids, source_ids[1:])):
        return ()
    return tuple(source_ids)


def resolve_kv_pool_transfer_mode(
    vllm_config: Any, use_layerwise: bool | None = None, *, extra_config: Mapping[str, Any] | None = None
) -> KVPoolTransferMode:
    """Use the same deterministic decision on scheduler and worker processes."""
    extra = vllm_config.kv_transfer_config.kv_connector_extra_config if extra_config is None else extra_config
    if use_layerwise is None:
        use_layerwise = extra.get("use_layerwise", False)
    if not use_layerwise:
        return KVPoolTransferMode.BULK
    # Shared-source scheduling extends Memcache GVA only. Mooncake retains
    # its existing block-key protocol regardless of the model's KV topology.
    if extra.get("backend", "mooncake").lower() != "memcache":
        return KVPoolTransferMode.PHYSICAL_LAYER
    config = text_model_config(vllm_config)
    sources = kv_source_layer_ids(config)
    declared_sources = config_value(config, "kv_source_layer_ids")
    if isinstance(declared_sources, (list, tuple)) and not declared_sources:
        return KVPoolTransferMode.PHYSICAL_LAYER
    if declared_sources is not None and not sources:
        raise ValueError("Invalid cross-layer KV topology: check num_hidden_layers and kv_source_layer_ids")
    if sources and len(sources) < config_value(config, "num_hidden_layers"):
        return KVPoolTransferMode.CROSS_LAYER_KV
    return KVPoolTransferMode.PHYSICAL_LAYER
