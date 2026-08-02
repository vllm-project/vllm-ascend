# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Metadata types for Mooncake KV transfer connectors."""

from dataclasses import dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


@dataclass
class ReqMeta:
    """Request metadata required by the initial Mooncake pull path."""

    local_block_ids: BlockIds
    num_external_tokens: int
    num_computed_tokens: int
    remote_block_ids: BlockIds
    remote_host: str
    remote_port: int
    remote_engine_id: str
    remote_request_id: str
    num_prompt_blocks: int
    remote_block_size: int


class MooncakeConnectorMetadata(KVConnectorMetadata):
    """Scheduler-to-worker metadata for Mooncake KV transfers."""

    def __init__(self) -> None:
        self.requests: dict[str, ReqMeta] = {}
        self.requests_to_send: dict[str, float] = {}
        self.reqs_in_batch: set[str] = set()

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        num_external_tokens: int,
        kv_transfer_params: dict[str, Any],
    ) -> None:
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            num_external_tokens=num_external_tokens,
            num_computed_tokens=kv_transfer_params.get("num_computed_tokens", 0),
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_request_id=kv_transfer_params["remote_request_id"],
            num_prompt_blocks=kv_transfer_params.get("num_prompt_blocks", 0),
            remote_block_size=kv_transfer_params.get("remote_block_size", 0),
        )


__all__ = ["MooncakeConnectorMetadata", "ReqMeta"]
