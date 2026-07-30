# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Scheduler-side logic for Mooncake pull transfers."""

import math
import time
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import (
    MooncakeBaseConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class MooncakePullConnectorScheduler(MooncakeBaseConnectorScheduler):
    """Scheduler-side Mooncake pull connector implementation."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # Requests waiting for the worker to start a READ transfer.
        self._reqs_need_recv: dict[str, tuple[Request, BlockIds, BlockIds, int]] = {}
        # Producer requests whose blocks must remain allocated until read.
        self._reqs_need_send: dict[str, float] = {}
        self._reqs_in_batch: set[str] = set()

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        """Return prompt tokens that will be loaded from a remote producer."""
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector get_num_new_matched_tokens: num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            token_ids = request.prompt_token_ids or []
            actual = self._state_prefill_token_count(len(token_ids))
            params["num_computed_tokens"] = num_computed_tokens
            count = max(actual - num_computed_tokens, 0)
            if count > 0:
                return count, True

        if params is not None and params.get("do_remote_decode") and self.need_truncate:
            self._truncate_request_for_prefill(request)

        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if params is not None and (params.get("do_remote_prefill", False) or params.get("do_remote_decode", False)):
            self._reqs_in_batch.add(request.request_id)

        if params is None or not params.get("do_remote_prefill"):
            return

        if params.get("remote_block_ids"):
            required_remote_fields = (
                "remote_engine_id",
                "remote_host",
                "remote_port",
                "remote_request_id",
            )
            if all(field in params for field in required_remote_fields):
                local_block_ids = blocks.get_unhashed_block_ids_all_groups() if num_external_tokens > 0 else []
                local_full_block_ids = blocks.get_block_ids() if num_external_tokens > 0 else tuple()
                self._reqs_need_recv[request.request_id] = (
                    request,
                    local_block_ids,
                    local_full_block_ids,
                    num_external_tokens,
                )
            else:
                logger.warning("Got invalid KVTransferParams. params=%s.", params)
        else:
            assert num_external_tokens == 0

        # Only trigger one transfer for a request.
        params["do_remote_prefill"] = False

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        for (
            req_id,
            (req, block_ids, full_block_ids, num_external_tokens),
        ) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                local_full_block_ids=full_block_ids,
                num_external_tokens=num_external_tokens,
                kv_transfer_params=req.kv_transfer_params,
            )

        self._reqs_need_recv.clear()
        meta.requests_to_send = self._reqs_need_send
        self._reqs_need_send = {}
        meta.reqs_in_batch = self._reqs_in_batch
        self._reqs_in_batch = set()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Expose completed producer blocks for a remote READ transfer."""
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished: request_status=%s, kv_transfer_params=%s",
            request.status,
            params,
        )

        if (
            params is None
            or not params.get("do_remote_decode")
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            return False, None

        prompt_token_ids = request.prompt_token_ids or []
        prompt_len = len(prompt_token_ids)
        num_prompt_blocks = math.ceil(prompt_len / self.block_size)
        computed_block_ids = self._get_transfer_block_ids(block_ids, prompt_len)
        num_computed_blocks = sum(len(group_block_ids) for group_block_ids in computed_block_ids)
        delay_free_blocks = num_computed_blocks > 0
        if delay_free_blocks:
            logger.info(
                "Delaying free of %d blocks for request %s",
                num_computed_blocks,
                request.request_id,
            )
            self._reqs_need_send[request.request_id] = time.time()

        return delay_free_blocks, {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_block_ids": computed_block_ids,
            "remote_engine_id": self.engine_id,
            "remote_request_id": request.request_id,
            "remote_host": self.side_channel_host,
            "remote_port": self.side_channel_port,
            "remote_pcp_size": self.pcp_size,
            "remote_dcp_size": self.dcp_size,
            "remote_ptp_size": self.tp_size,
            "last_token_id": request.output_token_ids[-1],
            "remote_multi_nodes_meta_mapping": (self.multi_nodes_meta_mapping),
            "num_prompt_blocks": num_prompt_blocks,
            "remote_block_size": self.block_size,
        }


__all__ = ["MooncakePullConnectorScheduler"]
