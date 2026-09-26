# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observability integrations owned by vLLM Ascend.

Layout:
- ``metrics`` — MS Service Metric provider / handlers / YAML
- ``runtime_config`` — Runtime Guard JSON control plane
- ``runtime_guard`` — detect / report / dump_kv orchestration
"""

from vllm_ascend.observability.metrics import get_metric_provider

__all__ = ["get_metric_provider"]
