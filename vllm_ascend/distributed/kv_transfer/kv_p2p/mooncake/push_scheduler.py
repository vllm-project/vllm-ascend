# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Scheduler-owned bootstrap service for Mooncake layerwise push transfers."""

import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import msgspec
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorHandshakeMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    get_mooncake_bootstrap_addr,
)

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import (
    GET_META_MSG,
    MooncakeBaseConnectorScheduler,
    MooncakeMetadataThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.bootstrap_server import AscendMooncakeBootstrapServer
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

    D serves static worker metadata on a background thread. Worker endpoint
    registration and layerwise transfer scheduling remain unimplemented.
    """

    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig") -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self.bootstrap_server: AscendMooncakeBootstrapServer | None = None
        self._metadata_thread: MooncakePushMetadataThread | None = None
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
        """Process allocated blocks for Push requests; implementation is deferred."""
        raise NotImplementedError

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> MooncakeConnectorMetadata:
        """Build scheduler-to-worker metadata; Push request population is deferred."""
        raise NotImplementedError

    def request_finished(self, request: "Request", block_ids: BlockIds) -> tuple[bool, dict[str, Any] | None]:
        """Publish final P blocks and retain them until worker transfers settle.

        Return delayed-free ownership and proxy-compatible transfer params;
        cancellation must not release allocations used by in-flight WRITEs.
        """
        raise NotImplementedError

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        """Consume aggregated worker outcomes for receive and delayed-free state.

        Preserve failure information so D cannot run with incomplete KV.
        """
        raise NotImplementedError

    def set_xfer_handshake_metadata_from_workers(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        """Publish D worker metadata after complete topology validation."""
        if self.kv_transfer_config.is_kv_producer:
            # P bootstrap registration is a separate, deferred control path.
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
