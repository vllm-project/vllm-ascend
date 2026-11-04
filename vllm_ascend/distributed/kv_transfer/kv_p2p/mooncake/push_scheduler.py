# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Scheduler-owned bootstrap service for Mooncake layerwise push transfers."""

import threading
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import msgspec
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorHandshakeMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    get_mooncake_bootstrap_addr,
)
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import (
    GET_META_MSG,
    MooncakeBaseConnectorScheduler,
    MooncakeMetadataThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.bootstrap_server import (
    AscendMooncakeBootstrapServer,
    register_workers_with_bootstrap,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
)

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
    """Own the P bootstrap service in EngineCore's scheduler process.

    Register P endpoints and retain source blocks until workers finish sends.
    D serves static worker metadata on a background thread.
    """

    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig") -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self.bootstrap_server: AscendMooncakeBootstrapServer | None = None
        self._metadata_thread: MooncakePushMetadataThread | None = None
        self._reqs_pending_recv: dict[str, tuple[Request, BlockIds, int]] = {}
        self._reqs_need_send_layerwise: dict[str, tuple[Request, BlockIds]] = {}
        self._reqs_sent_to_worker: set[str] = set()
        self._reqs_need_send: dict[str, float] = {}
        self._reqs_finished_sending: set[str] = set()
        self._reqs_pending_send: dict[str, float] = {}
        self._reqs_aborted: dict[str, str] = {}
        self._registered_worker_metadata: bytes | None = None
        self.start_bootstrap_server()

    def _should_launch_bootstrap_server(self) -> bool:
        """Elect one P scheduler without querying worker TP/PP process groups."""
        if not self.kv_transfer_config.is_kv_producer:
            return False
        if self.kv_transfer_config.get_from_extra_config("bootstrap_server_address", None) is not None:
            return False
        parallel_config = self.vllm_config.parallel_config
        if parallel_config.local_engines_only:
            return parallel_config.data_parallel_rank_local == 0
        return parallel_config.data_parallel_index == 0

    def start_bootstrap_server(self) -> None:
        """Start the upstream HTTP registry in a scheduler-owned thread."""
        if not self._should_launch_bootstrap_server() or self.bootstrap_server is not None:
            return
        _, port = get_mooncake_bootstrap_addr(self.vllm_config)
        server = AscendMooncakeBootstrapServer("0.0.0.0", port)
        try:
            server.start()
        except Exception:
            server.shutdown()
            raise
        self.bootstrap_server = server

    def shutdown(self) -> None:
        """Called through the connector when EngineCore shuts down its scheduler."""
        if self._metadata_thread is not None:
            self._metadata_thread.shutdown()
            self._metadata_thread = None
        if self.bootstrap_server is not None:
            self.bootstrap_server.shutdown()
            self.bootstrap_server = None

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int) -> None:
        """Track P source allocations or queue D receives for worker metadata."""
        params = request.kv_transfer_params
        if self.kv_transfer_config.is_kv_producer:
            if not params or not params.get("do_remote_decode"):
                return
            if not params.get("transfer_id"):
                raise ValueError("Missing Push transfer parameter: transfer_id")
            if request.request_id in self._reqs_sent_to_worker:
                raise RuntimeError("Preempting an active Mooncake Push transfer is not supported")
            self._reqs_need_send_layerwise[request.request_id] = (
                request,
                tuple(list(group) for group in blocks.get_block_ids()),
            )
        else:
            if not params or not params.get("do_remote_prefill"):
                return
            if num_external_tokens > 0:
                if self._metadata_thread is None:
                    raise RuntimeError("D scheduler metadata service is not ready")
                self._reqs_pending_recv[request.request_id] = (
                    request,
                    tuple(list(group) for group in blocks.get_block_ids()),
                    num_external_tokens,
                )
            params["do_remote_prefill"] = False

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> MooncakeConnectorMetadata:
        meta = MooncakeConnectorMetadata()
        if self.kv_transfer_config.is_kv_producer:
            # Core can recycle preempted blocks before the worker receives this
            # output. Until WRITE lifetime fencing is implemented, fail before
            # dispatching a new model step rather than reuse a stale source table.
            if set(scheduler_output.preempted_req_ids or ()) & self._reqs_sent_to_worker:
                raise RuntimeError("Preempting an active Mooncake Push transfer is not supported")
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
                assert request.kv_transfer_params is not None
                meta.add_new_push_send_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
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

        else:
            for req_id, (req, block_ids, num_external_tokens) in self._reqs_pending_recv.items():
                assert req.kv_transfer_params is not None
                meta.add_new_push_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    num_external_tokens=num_external_tokens,
                    kv_transfer_params=req.kv_transfer_params,
                    scheduler_host=self.side_channel_host,
                    scheduler_port=self.side_channel_port,
                )
            self._reqs_pending_recv.clear()
        return meta

    def request_finished(self, request: "Request", block_ids: BlockIds) -> tuple[bool, dict[str, Any] | None]:
        if not self.kv_transfer_config.is_kv_producer:
            # Core pins cancelled D receives until finished_recving.
            return False, None
        req_id = request.request_id
        params = request.kv_transfer_params
        if not params or not params.get("do_remote_decode") or not params.get("transfer_id"):
            return False, None
        normal_finish = request.status in (RequestStatus.FINISHED_LENGTH_CAPPED, RequestStatus.FINISHED_STOPPED)
        if not normal_finish:
            # Ready may arrive before the first source metadata. The worker
            # needs transfer_id to reject that pending handoff even on abort.
            self._reqs_aborted[req_id] = params["transfer_id"]
        if req_id not in self._reqs_need_send_layerwise:
            return False, None
        # The Mooncake proxy already provides D's transfer_id, P engine and
        # externally reachable bootstrap endpoint. Do not return the local
        # registration URL (which can be loopback in external-LB mode).
        if req_id not in self._reqs_sent_to_worker or req_id in self._reqs_finished_sending:
            self._reqs_finished_sending.discard(req_id)
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
            elif req_id in self._reqs_need_send_layerwise:
                # Do not let core free a still-running request when transfer
                # completion arrives before request_finished (e.g. async PP).
                self._reqs_finished_sending.add(req_id)
        connector_output.finished_sending = finished_sending or None

    def set_xfer_handshake_metadata_from_workers(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        """Validate worker topology, then register P endpoints or serve D metadata."""
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
        if self.kv_transfer_config.is_kv_producer:
            if self._registered_worker_metadata is not None:
                if thread.encoded_metadata != self._registered_worker_metadata:
                    raise RuntimeError("Replacing registered P worker metadata is not supported")
                return
            host, port = get_mooncake_bootstrap_addr(self.vllm_config)
            register_workers_with_bootstrap(
                host=host,
                port=port,
                engine_id=self.engine_id,
                dp_rank=self.vllm_config.parallel_config.data_parallel_index,
                tp_size=self.tp_size,
                pp_size=self.pp_size,
                pcp_size=self.pcp_size,
                dcp_size=self.dcp_size,
                encoded_metadata=thread.encoded_metadata,
            )
            self._registered_worker_metadata = thread.encoded_metadata
            return
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
