# SPDX-License-Identifier: Apache-2.0
"""Local SFA row reservations; the cross-host handshake stays Mooncake's."""

from dataclasses import dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata, ReqMeta


@dataclass
class MooncakeSFAReqMeta(ReqMeta):
    copy_sfa_slot: int | None = None


class MooncakeSFAConnectorMetadata(MooncakeConnectorMetadata):
    def __init__(self, bindings: dict[str, int] | None = None) -> None:
        super().__init__()
        self.copy_sfa_slots = dict(bindings or {})

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        local_num_prompt_tokens: int,
        num_external_tokens: int,
        kv_transfer_params: dict[str, Any],
        local_full_block_ids: BlockIds | None = None,
        copy_sfa_slot: int | None = None,
    ) -> None:
        super().add_new_req(
            request_id,
            local_block_ids,
            local_num_prompt_tokens,
            num_external_tokens,
            kv_transfer_params,
            local_full_block_ids,
        )
        self.requests[request_id] = MooncakeSFAReqMeta(
            **vars(self.requests[request_id]),
            copy_sfa_slot=self.copy_sfa_slots.get(request_id) if copy_sfa_slot is None else copy_sfa_slot,
        )
