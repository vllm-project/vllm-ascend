"""Runtime control plane (``runtime_config.json``)."""
from vllm_ascend.runtime_config.config import RuntimeConfig, resolve_runtime_config_path, resolve_runtime_report_dir

__all__ = ["RuntimeConfig", "resolve_runtime_config_path", "resolve_runtime_report_dir"]
