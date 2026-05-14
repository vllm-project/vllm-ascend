from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def maybe_report_v1_usage_stats(vllm_config: Any) -> None:
    """Report vLLM V1 usage stats if the feature is available.

    vLLM reports anonymized usage statistics by default (users can opt out
    following upstream documentation). vLLM Ascend should behave consistently
    with upstream V1 workers by reporting once on rank0 during initialization.

    This helper is intentionally best-effort and must never crash the worker.
    """
    try:
        from vllm.v1.utils import report_usage_stats  # type: ignore
    except Exception:
        return

    try:
        report_usage_stats(vllm_config)
    except Exception as e:
        logger.debug("Failed to report v1 usage stats: %s", e)

