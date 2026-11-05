# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forward CED attention hooks through composite KV connectors."""

from typing import Any


def dispatch_ced_hook(connector: Any, method_name: str, layer_idx: int) -> None:
    """Invoke a connector hook, or visit the public MultiConnector children."""
    hook = getattr(connector, method_name, None)
    if callable(hook):
        hook(layer_idx)
        return
    for child in getattr(connector, "sub_connectors", ()):
        dispatch_ced_hook(child, method_name, layer_idx)
