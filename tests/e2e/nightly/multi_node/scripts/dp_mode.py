# SPDX-License-Identifier: Apache-2.0
"""Resolve the DP launch mode declared by a multi-node test case."""

import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

SUPPORTED_DP_LOAD_BALANCING = {"internal", "external"}


def resolve_config_path(
    yaml_path: str | None = None,
    config_base_path: str | None = None,
) -> Path:
    """Resolve a case config in the same way as the existing config loaders."""
    raw_path = yaml_path or os.getenv("CONFIG_YAML_PATH")
    if not raw_path:
        raise ValueError("CONFIG_YAML_PATH is required")

    path = Path(raw_path)
    if not path.is_absolute() and not path.exists():
        base_path = config_base_path or os.getenv("CONFIG_BASE_PATH")
        if base_path:
            path = Path(base_path) / path
    return path


def resolve_dp_load_balancing(
    yaml_path: str | None = None,
    config_base_path: str | None = None,
) -> str:
    """Read ``dp_load_balancing`` from the case YAML.

    The path fallback only supports configs from before the field was added.
    New and migrated configs must declare the field explicitly.
    """
    path = resolve_config_path(yaml_path, config_base_path)
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise TypeError(f"multi-node config must be a mapping: {path}")

    mode = config.get("dp_load_balancing")
    if mode is None:
        normalized_path = path.as_posix()
        mode = "external" if "/external_dp/" in f"/{normalized_path}" else "internal"
        logger.warning(
            "%s does not declare dp_load_balancing; using legacy path fallback: %s",
            path,
            mode,
        )

    mode = str(mode).lower()
    if mode not in SUPPORTED_DP_LOAD_BALANCING:
        supported = ", ".join(sorted(SUPPORTED_DP_LOAD_BALANCING))
        raise ValueError(f"Unsupported dp_load_balancing={mode!r} in {path}; expected one of: {supported}")
    return mode
