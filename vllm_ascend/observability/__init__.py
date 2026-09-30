# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observability integrations owned by vLLM Ascend."""

from .provider import get_metric_provider

__all__ = ["get_metric_provider"]
