# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Scheduler-to-worker messages for whole-request Mooncake push."""

from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata


@dataclass(frozen=True)
class PushReceive:
    request_id: str
    transfer_id: str
    block_ids: BlockIds
    num_tokens: int
    producer_host: str
    producer_port: int
    producer_engine_id: str
    producer_tp_size: int


@dataclass(frozen=True)
class PushSource:
    request_id: str
    transfer_id: str
    block_ids: BlockIds
    num_tokens: int
    delayed_free: bool
    cancelled: bool = False


class MooncakePushMetadata(MooncakeConnectorMetadata):
    def __init__(self) -> None:
        super().__init__()
        self.push_receives: list[PushReceive] = []
        self.push_sources: list[PushSource] = []
        self.push_cancellations: list[str] = []
