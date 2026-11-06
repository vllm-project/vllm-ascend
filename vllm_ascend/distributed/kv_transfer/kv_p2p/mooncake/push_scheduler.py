# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Layerwise proxy dispatch and scheduler metadata service for Mooncake Push."""

import threading
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import msgspec
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorHandshakeMetadata
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import (
    GET_META_MSG,
    MooncakeBaseConnectorScheduler,
    MooncakeMetadataThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
    MooncakeTransferMetadataGroups,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import MooncakeMetaserverClient

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


class MooncakePushMetadataThread(MooncakeMetadataThread):
    """D metadata service with startup confirmation and engine validation."""

    def start(self) -> None:
        super().start()
        if not self.ready_event.wait(timeout=10):
            self.shutdown()
            raise TimeoutError("Timed out starting D scheduler metadata thread")
        if self.error is not None:
            raise RuntimeError("Mooncake scheduler metadata thread failed") from self.error

    def _handle_message(self, message: tuple) -> bytes | None:
        if message != (GET_META_MSG, self.engine_id):
            return msgspec.msgpack.encode({"error": "Expected metadata query for this engine"})
        return super()._handle_message(message)


class MooncakePushConnectorScheduler(MooncakeBaseConnectorScheduler):
    """Publish D allocations through the layerwise proxy; track P sends."""

    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig") -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self._metadata_thread: MooncakePushMetadataThread | None = None
        self._reqs_pending_recv: dict[str, tuple[Request, BlockIds, int]] = {}
        self._reqs_need_send_layerwise: dict[str, tuple[Request, BlockIds]] = {}
        self._reqs_sent_to_worker: set[str] = set()
        self._reqs_need_send: dict[str, float] = {}
        self._reqs_pending_send: dict[str, float] = {}
        self._reqs_aborted: dict[str, dict[str, Any]] = {}
        self._remote_worker_addresses: tuple[tuple[int, int, int, str], ...] = ()
        self.metaserver_client = MooncakeMetaserverClient() if self.kv_transfer_config.is_kv_consumer else None

    def shutdown(self) -> None:
        try:
            if self.metaserver_client is not None:
                self.metaserver_client.shutdown()
        finally:
            self.metaserver_client = None
            if self._metadata_thread is not None:
                try:
                    self._metadata_thread.shutdown()
                finally:
                    self._metadata_thread = None

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int) -> None:
        """Track P source allocations or queue D receives for worker metadata."""
        params = request.kv_transfer_params
        if self.kv_transfer_config.is_kv_producer:
            if not params or not params.get("do_remote_decode"):
                return
            if params["remote_num_external_tokens"] == 0:
                return
            self._reqs_need_send_layerwise[request.request_id] = (
                request,
                tuple(list(group) for group in blocks.get_block_ids()),
            )
        else:
            if not params or not params.get("do_remote_prefill"):
                return
            if num_external_tokens > 0:
                self._reqs_pending_recv[request.request_id] = (
                    request,
                    tuple(list(group) for group in blocks.get_block_ids()),
                    num_external_tokens,
                )
            params["do_remote_prefill"] = False
            # The proxy matches external IDs; workers retain the full D ID.
            external_request_id = request.request_id[:-9]
            message = dict(
                request_id=external_request_id,
                remote_request_id=request.request_id,
                token_ids=[],
                do_remote_prefill=False,
                do_remote_decode=True,
                remote_block_ids=tuple(
                    tuple(group)
                    for group in self._get_transfer_block_ids(blocks.get_block_ids(), request.num_prompt_tokens)
                ),
                remote_block_size=self.group_block_size,
                remote_engine_id=self.engine_id,
                remote_host=self.side_channel_host,
                remote_port=self.side_channel_port,
                remote_tp_size=self.tp_size,
                remote_pp_size=self.pp_size,
                remote_pcp_size=self.pcp_size,
                remote_dcp_size=self.dcp_size,
                remote_num_computed_tokens=params["num_computed_tokens"],
                remote_num_external_tokens=num_external_tokens,
                remote_worker_addresses=self._remote_worker_addresses,
            )
            assert self.metaserver_client is not None
            self.metaserver_client.submit(request.request_id, params.get("metaserver"), message)

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> MooncakeConnectorMetadata:
        meta = MooncakeConnectorMetadata()
        if self.kv_transfer_config.is_kv_producer:
            # TODO: propagate P preemption to workers and retire the old transfer
            # before core reuses its blocks; resumed requests need fresh progress.
            # Updating resumed block IDs alone does not provide that handshake.
            # Ordinary request cancellation is handled by request_finished below.
            # The current deployment assumes P requests are not preempted.
            cached_reqs = scheduler_output.scheduled_cached_reqs
            num_computed_tokens = dict(zip(cached_reqs.req_ids, cached_reqs.num_computed_tokens))
            for req in scheduler_output.scheduled_new_reqs:
                num_computed_tokens[req.req_id] = req.num_computed_tokens
            for req_id, new_blocks in zip(cached_reqs.req_ids, cached_reqs.new_block_ids):
                if req_id not in self._reqs_need_send_layerwise:
                    continue
                request, block_ids = self._reqs_need_send_layerwise[req_id]
                if new_blocks is not None:
                    if req_id in cached_reqs.resumed_req_ids:
                        block_ids = tuple(list(group) for group in new_blocks)
                    else:
                        block_ids = tuple(old + new for old, new in zip(block_ids, new_blocks, strict=True))
                    self._reqs_need_send_layerwise[req_id] = request, block_ids
            for req_id, num_scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
                if req_id not in self._reqs_need_send_layerwise or req_id in self._reqs_need_send:
                    continue
                request, block_ids = self._reqs_need_send_layerwise[req_id]
                local_num_computed_tokens = num_computed_tokens[req_id]
                if local_num_computed_tokens >= request.num_prompt_tokens:
                    continue
                meta.add_new_push_send_req(
                    request_id=req_id,
                    # Match D's transfer table: the final Mamba block must be
                    # the prompt state, not a speculative lookahead block.
                    local_block_ids=self._get_transfer_block_ids(block_ids, request.num_prompt_tokens),
                    num_prompt_tokens=request.num_prompt_tokens,
                    num_computed_tokens=local_num_computed_tokens,
                    num_scheduled_tokens=min(
                        num_scheduled_tokens, request.num_prompt_tokens - local_num_computed_tokens
                    ),
                    kv_transfer_params=request.kv_transfer_params,
                )
                self._reqs_sent_to_worker.add(req_id)
            # These notifications must also be delivered in connector-only steps.
            meta.requests_to_send = self._reqs_pending_send
            self._reqs_pending_send = {}
            meta.push_aborted_requests = self._reqs_aborted
            self._reqs_aborted = {}
            # P layer hooks need request state before forward, including steps
            # that only drain outstanding transfers or process cancellations.
            if self._reqs_need_send_layerwise or meta.requests_to_send or meta.push_aborted_requests:
                scheduler_output.has_sync_kv_loads = True

        else:
            # Poll without waiting, including connector-only steps. Workers
            # decide when failed receives are safe to release.
            if self.metaserver_client is not None:
                meta.push_failed_requests = self.metaserver_client.get_failed_requests()
            for req_id, (req, block_ids, num_external_tokens) in self._reqs_pending_recv.items():
                assert req.kv_transfer_params is not None
                meta.add_new_push_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    num_external_tokens=num_external_tokens,
                    kv_transfer_params=req.kv_transfer_params,
                )
            self._reqs_pending_recv.clear()
        return meta

    def request_finished(self, request: "Request", block_ids: BlockIds) -> tuple[bool, dict[str, Any] | None]:
        if not self.kv_transfer_config.is_kv_producer:
            # Core pins cancelled D receives until finished_recving.
            return False, None
        req_id = request.request_id
        params = request.kv_transfer_params
        if not params or not params.get("do_remote_decode"):
            return False, None
        normal_finish = request.status in (RequestStatus.FINISHED_LENGTH_CAPPED, RequestStatus.FINISHED_STOPPED)
        if not normal_finish:
            # Notify workers even if cancellation precedes the first dispatch.
            self._reqs_aborted[req_id] = dict(
                remote_request_id=params["remote_request_id"],
                remote_engine_id=params["remote_engine_id"],
                remote_worker_addresses=tuple(tuple(address) for address in params["remote_worker_addresses"]),
            )
        if req_id not in self._reqs_need_send_layerwise:
            return False, None
        # Normal prefill output is returned only after the worker's send barrier.
        # Transfer failures are reported to D through the worker's DONE status.
        if normal_finish or req_id not in self._reqs_sent_to_worker:
            if normal_finish and req_id in self._reqs_sent_to_worker:
                # Retire worker bookkeeping on the next step; KV is already safe to free.
                self._reqs_pending_send[req_id] = time.monotonic()
            self._reqs_sent_to_worker.discard(req_id)
            self._reqs_need_send_layerwise.pop(req_id)
            return False, None
        # Includes cancellation: submitted WRITEs may still read P blocks.
        self._reqs_need_send[req_id] = time.monotonic()
        self._reqs_pending_send[req_id] = self._reqs_need_send[req_id]
        return True, None

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        if not self.kv_transfer_config.is_kv_producer:
            return
        finished_sending: set[str] = set()
        for req_id in connector_output.finished_sending or ():
            if req_id in self._reqs_need_send:
                finished_sending.add(req_id)
                self._reqs_need_send.pop(req_id)
                self._reqs_sent_to_worker.discard(req_id)
                self._reqs_need_send_layerwise.pop(req_id)
                self._reqs_pending_send.pop(req_id, None)
                self._reqs_aborted.pop(req_id, None)
        # Normal requests were already freed; only delayed cancellations need
        # finished_sending forwarded to core.
        connector_output.finished_sending = finished_sending or None

    def set_xfer_handshake_metadata_from_workers(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        """Serve D cache layouts and publish its worker completion endpoints."""
        if self.kv_transfer_config.is_kv_producer:
            return
        use_kv_pp = self.ascend_config.kvpp_config.size > 1
        if use_kv_pp and self.dcp_size != 1:
            raise ValueError("Mooncake KVPP cannot be combined with DCP")
        thread = MooncakePushMetadataThread(
            self.side_channel_host,
            self.side_channel_port,
            self.engine_id,
            metadata,
            self.tp_size,
            self.pp_size,
            self.pcp_size,
            self.dcp_size,
            use_kv_pp,
            threading.Event(),
        )
        if self._metadata_thread is not None:
            if self._metadata_thread.error is not None:
                raise RuntimeError("Mooncake scheduler metadata thread failed") from self._metadata_thread.error
            if self._metadata_thread.encoded_metadata != thread.encoded_metadata:
                raise RuntimeError("Replacing active D worker metadata is not supported")
            return
        try:
            thread.start()
        except Exception:
            thread.shutdown()
            raise
        self._metadata_thread = thread
        groups = msgspec.msgpack.decode(thread.encoded_metadata, type=MooncakeTransferMetadataGroups)
        self._remote_worker_addresses = tuple(
            (pp_rank, pcp_rank, tp_rank, make_zmq_path("tcp", worker.local_ip, worker.handshake_port))
            for pp_rank, pp in groups.metadata_by_pp_rank.items()
            for pcp_rank, pcp in pp.metadata_by_pcp_rank.items()
            for tp_rank, worker in pcp.metadata_by_tp_rank.items()
        )
