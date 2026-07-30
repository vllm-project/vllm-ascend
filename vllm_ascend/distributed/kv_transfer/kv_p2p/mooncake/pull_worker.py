# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Worker-side implementation entry point for Mooncake pull transfers."""

from .base_worker import MooncakeBaseConnectorWorker


class MooncakePullConnectorWorker(MooncakeBaseConnectorWorker):
    """Worker-side Mooncake pull connector implementation."""

    pass


__all__ = ["MooncakePullConnectorWorker"]
