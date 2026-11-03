# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Whole-request push scheduling; no per-layer attention hooks."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from uuid import UUID

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorHandshakeMetadata
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import MooncakeBaseConnectorScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_config import PushConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_metadata import (
    MooncakePushMetadata,
    PushReceive,
    PushSource,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


def transfer_id(request: "Request") -> str:
    params = request.kv_transfer_params or {}
    value = params.get("push_transfer_id")
    if not isinstance(value, str):
        raise ValueError("Mooncake push requires a unique push_transfer_id UUID for each request attempt")
    return str(UUID(value))


class MooncakePushConnectorScheduler(MooncakeBaseConnectorScheduler):
    def __init__(self, config: "VllmConfig", engine_id: str, cache: "KVCacheConfig") -> None:
        super().__init__(config, engine_id, cache)
        self.push_config = PushConfig.from_config(config, cache)
        self._new_receives: dict[str, PushReceive] = {}
        self._receiving: set[str] = set()
        self._sending: set[str] = set()
        self._new_sources: list[PushSource] = []
        self._cancellations: list[str] = []

    def shutdown(self) -> None:
        self._new_receives.clear()
        self._new_sources.clear()
        self._cancellations.clear()
        self._receiving.clear()
        self._sending.clear()

    def set_xfer_handshake_metadata_from_workers(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        # Push exchanges static metadata directly between paired workers.
        pass

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        params = request.kv_transfer_params or {}
        if not params.get("do_remote_prefill"):
            return 0, False
        transfer_id(request)
        if params.get("remote_tp_size") != self.tp_size:
            raise ValueError("Mooncake push initially requires equal producer and consumer TP sizes")
        # D computes the final prompt token and samples the first output token.
        # P's sampled token is discarded, so D can be submitted before P returns.
        count = max(request.num_prompt_tokens - 1 - num_computed_tokens, 0)
        return count, count > 0

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int) -> None:
        params = request.kv_transfer_params or {}
        if not params.get("do_remote_prefill"):
            return
        if num_external_tokens and request.num_computed_tokens:
            raise ValueError("Mooncake push does not initially support local prefix hits")
        key = transfer_id(request)
        if request.request_id in self._receiving:
            raise ValueError("A push receive is already registered for this request")
        num_tokens = max(request.num_prompt_tokens - 1, 0)
        receive = PushReceive(
            request_id=request.request_id,
            transfer_id=key,
            block_ids=self._get_transfer_block_ids(blocks.get_block_ids(), num_tokens),
            num_tokens=num_tokens,
            producer_host=params["remote_host"],
            producer_port=int(params["remote_port"]),
            producer_engine_id=params["remote_engine_id"],
            producer_tp_size=int(params["remote_tp_size"]),
        )
        self._new_receives[request.request_id] = receive
        if num_external_tokens:
            self._receiving.add(request.request_id)
        params["do_remote_prefill"] = False

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> MooncakePushMetadata:
        meta = MooncakePushMetadata()
        meta.push_receives = list(self._new_receives.values())
        meta.push_sources, self._new_sources = self._new_sources, []
        meta.push_cancellations, self._cancellations = self._cancellations, []
        self._new_receives.clear()
        return meta

    def request_finished(self, request: "Request", block_ids: BlockIds) -> tuple[bool, dict[str, Any] | None]:
        request_id = request.request_id
        if request_id in self._receiving:
            if request_id in self._new_receives:
                # Registration was never handed to workers: no remote WRITE is possible.
                self._new_receives.pop(request_id)
                self._receiving.remove(request_id)
                return False, None
            self._cancellations.append(request_id)
            # Keep D's blocks until the worker gets a terminal response from P.
            return True, None
        params = request.kv_transfer_params or {}
        if not params.get("do_remote_decode"):
            return False, None
        key = transfer_id(request)
        cancelled = request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        num_tokens = max(request.num_prompt_tokens - 1, 0)
        source_blocks = self._get_transfer_block_ids(block_ids, num_tokens) if not cancelled else tuple()
        delayed_free = any(source_blocks)
        self._new_sources.append(PushSource(request_id, key, source_blocks, num_tokens, delayed_free, cancelled))
        if delayed_free:
            self._sending.add(request_id)
        return delayed_free, None

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        self._sending.difference_update(connector_output.finished_sending or ())
        self._receiving.difference_update(connector_output.finished_recving or ())

    def has_pending_push_work(self) -> bool:
        return bool(self._new_receives or self._new_sources or self._cancellations or self._sending or self._receiving)
