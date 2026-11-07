# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forward CrossLayerKV attention hooks through composite KV connectors."""

from collections.abc import Mapping
from typing import Any


def supports_cross_layer_kv_hooks(forward_context: Mapping[str, Any], num_layers: int) -> bool:
    """Require a registered consumer gate on every configured backbone layer."""
    if not isinstance(forward_context, Mapping) or type(num_layers) is not int or num_layers <= 0:
        return False
    layers = {
        layer.layer_idx
        for layer in forward_context.values()
        if getattr(layer, "supports_cross_layer_kv_transfer_hooks", False) is True
        and type(getattr(layer, "layer_idx", None)) is int
        and 0 <= layer.layer_idx < num_layers
    }
    return layers == set(range(num_layers))


def dispatch_cross_layer_kv_hook(connector: Any, method_name: str, layer_idx: int) -> None:
    """Invoke a connector hook, or visit the public MultiConnector children."""
    hook = getattr(connector, method_name, None)
    if callable(hook):
        hook(layer_idx)
        return
    for child in getattr(connector, "sub_connectors", ()):
        dispatch_cross_layer_kv_hook(child, method_name, layer_idx)
