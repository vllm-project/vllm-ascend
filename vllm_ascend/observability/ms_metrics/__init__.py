# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MS Service Metric provider for vLLM Ascend."""

from .provider import get_metric_provider

__all__ = ["get_metric_provider"]
