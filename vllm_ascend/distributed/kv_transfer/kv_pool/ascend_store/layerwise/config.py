# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration shared by the scheduler, worker and cache-buffer planner."""

from collections.abc import Mapping
from typing import Any

import regex as re

DEFAULT_KEY_PREFETCH_LAYERS = 2
MAX_DEFAULT_GVA_PREFETCH_LAYERS = 8
PREFETCH_LAYERS = "layerwise_prefetch_layers"


def parse_layerwise_prefetch_layers(extra_config: Mapping[str, Any], default: int) -> int:
    """Accept positive integers and losslessly parsed decimal integer strings.

    In particular, ``int(value)`` alone silently truncates floats and accepts
    booleans. Validate types before conversion so an invalid prefetch window
    cannot reach a transfer thread or leave layer zero waiting indefinitely.
    """
    value = extra_config.get(PREFETCH_LAYERS, default)
    if isinstance(value, str) and re.fullmatch(r"\+?[0-9]+", value.strip()):
        try:
            value = int(value.strip(), 10)
        except ValueError as exc:
            raise ValueError(f"{PREFETCH_LAYERS} must be a positive integer; got {value!r}") from exc
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{PREFETCH_LAYERS} must be a positive integer; got {value!r}")
    return value


def validate_layerwise_config(extra_config: Mapping[str, Any], backend_name: str, data_plane: str | None) -> None:
    """Validate before creating a backend client or either transfer thread."""
    if data_plane not in ("gva", "block_key"):
        raise ValueError(
            f"Backend {backend_name!r} does not support layerwise KV pooling: "
            f"unsupported layerwise_data_plane={data_plane!r}"
        )
    # The GVA default is resolved by the planner once the buffer count is known.
    # An explicit value is checked here on every path, including the scheduler.
    parse_layerwise_prefetch_layers(extra_config, DEFAULT_KEY_PREFETCH_LAYERS)
    if data_plane != "gva":
        conflicting = [
            name for name in ("layerwise_num_shared_buffers", "layerwise_independent_layers") if name in extra_config
        ]
        if conflicting:
            raise ValueError(
                f"backend={backend_name!r} does not support layerwise NPU buffer reuse; "
                f"GVA-only options: {', '.join(conflicting)}"
            )
