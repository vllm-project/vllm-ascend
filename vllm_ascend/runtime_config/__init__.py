"""Runtime control plane (``runtime_config.json``)."""

from vllm_ascend.runtime_config.config import RuntimeConfig, resolve_runtime_config_path, resolve_runtime_report_dir
from vllm_ascend.runtime_config.from_additional_config import (
    ADDITIONAL_CONFIG_STRIP_KEYS,
    RuntimeConfigBootstrap,
    build_runtime_config_from_additional,
)

__all__ = [
    "ADDITIONAL_CONFIG_STRIP_KEYS",
    "RuntimeConfig",
    "RuntimeConfigBootstrap",
    "build_runtime_config_from_additional",
    "resolve_runtime_config_path",
    "resolve_runtime_report_dir",
]
