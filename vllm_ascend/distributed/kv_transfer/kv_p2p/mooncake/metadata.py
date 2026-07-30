# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Metadata types for Mooncake KV transfer connectors."""

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)


class MooncakeConnectorMetadata(KVConnectorMetadata):
    """Scheduler-to-worker metadata for Mooncake KV transfers."""

    pass


__all__ = ["MooncakeConnectorMetadata"]
