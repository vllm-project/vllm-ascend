# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Interfaces for future Mooncake layerwise Push worker tasks."""

from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata

if TYPE_CHECKING:
    import torch


class MooncakePushConnectorWorker(MooncakeBaseConnectorWorker):
    """Worker interfaces; control flow belongs to future threads or coroutines.

    Bootstrap and static metadata publication remain scheduler-owned.
    """

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor | list[torch.Tensor]"]) -> None:
        """Register KV storage only; control-task startup is deferred."""
        super().register_kv_caches(kv_caches)

    def start_load_kv(self, metadata: MooncakeConnectorMetadata) -> None:
        """Pass D allocations to the future control task for ready publication."""
        raise NotImplementedError

    def get_transfer_results(self) -> KVConnectorTransferResults:
        """Collect actual transfer outcomes; a Ready ACK is not KV completion."""
        raise NotImplementedError

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Collect load errors from the future transfer task."""
        raise NotImplementedError
