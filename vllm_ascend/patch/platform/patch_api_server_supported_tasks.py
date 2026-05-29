from __future__ import annotations

import json
from typing import Any

from vllm.logger import logger
from vllm.tasks import SupportedTask

try:
    from vllm.entrypoints.openai import api_server as _api_server
except Exception:
    _api_server = None


def _is_mtp_deepseek_v4(args: Any) -> bool:
    spec_cfg = getattr(args, "speculative_config", None)
    if isinstance(spec_cfg, str):
        try:
            spec_cfg = json.loads(spec_cfg)
        except Exception:
            spec_cfg = None
    method = None
    if isinstance(spec_cfg, dict):
        method = spec_cfg.get("method")
    model_tag = str(getattr(args, "model_tag", "") or getattr(args, "model", ""))
    return method == "mtp" and "deepseek-v4" in model_tag.lower()


if _api_server is not None and hasattr(_api_server, "build_app"):
    _orig_build_app = _api_server.build_app

    def _build_app_with_task_fallback(args: Any, supported_tasks: Any, model_config: Any):
        if not supported_tasks and _is_mtp_deepseek_v4(args):
            logger.warning_once(
                "Recovered empty supported_tasks in API server for MTP DeepSeek-V4; force enabling generate task."
            )
            supported_tasks = (SupportedTask.GENERATE,)
        return _orig_build_app(args, supported_tasks, model_config)

    _api_server.build_app = _build_app_with_task_fallback
