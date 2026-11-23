# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MS Service Metric integration owned by vLLM Ascend.

Layout:
- ``config/``: one YAML file per feature, auto-loaded by the provider.
- ``handlers/``: one handler module per feature plus shared helpers in
  ``handlers/common.py``; YAML configs reference handlers as
  ``vllm_ascend.observability.ms_metrics.handlers:function_name``.
"""

from vllm_ascend.observability.ms_metrics.provider import get_metric_provider

__all__ = ["get_metric_provider"]
