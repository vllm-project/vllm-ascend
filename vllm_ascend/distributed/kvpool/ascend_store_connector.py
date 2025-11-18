import threading
from typing import Any, Optional

import torch
import zmq
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.forward_context import ForwardContext
from vllm.utils import make_zmq_socket
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder

from vllm_ascend.distributed.kvpool.pool_worker import KVPoolWorker
from vllm_ascend.distributed.kvpool.pool_scheduler import KVPoolScheduler, get_zmq_rpc_path_lookup


class AscendStoreConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self.kv_role = vllm_config.kv_transfer_config.kv_role

        self.use_layerwise = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "use_layerwise", False)

        self.kv_caches: dict[str, torch.Tensor] = {}

        self._block_size = vllm_config.cache_config.block_size

        self.sended_but_unfinished_reqs: set[str] = set()

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = KVPoolScheduler(
                vllm_config, self.use_layerwise)
        else:
            self.connector_worker = KVPoolWorker(
                vllm_config,
                self.use_layerwise,
            )

            assert self.connector_worker is not None
            if vllm_config.parallel_config.rank == 0:
                self.lookup_server = LookupKeyServer(
                    self.connector_worker, vllm_config, self.use_layerwise)

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        assert self.connector_worker is not None
        self.connector_worker.start_load_kv(self._get_connector_metadata())

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self.use_layerwise:
            return
        self.connector_worker.wait_for_layer_load()

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        if not self.use_layerwise:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        self.connector_worker.save_kv_layer(self._get_connector_metadata())

    def wait_for_save(self):
        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return

        if self.use_layerwise:
        #     self.connector_worker.wait_layer_transfer_finish()
            return

        self.connector_worker.wait_for_save(self._get_connector_metadata())

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        meta = self._get_connector_metadata()
        done_sending, done_recving = self.connector_worker.get_finished()
        sended_and_finished: set[str] = set()
        for item in list(self.sended_but_unfinished_reqs):
            if item not in meta.unfinished_request_ids:
                sended_and_finished.add(item)
                self.sended_but_unfinished_reqs.remove(item)
        for item in done_sending:
            if item in meta.unfinished_request_ids:
                self.sended_but_unfinished_reqs.add(item)
            else:
                sended_and_finished.add(item)

        return sended_and_finished, done_recving


class LookupKeyServer:

    def __init__(
        self,
        pool_worker: KVPoolWorker,
        vllm_config: "VllmConfig",
        use_layerwise: bool,
    ):
        self.decoder = MsgpackDecoder(torch.Tensor)
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

        self.pool_worker = pool_worker
        self.running = True

        def process_request():
            while self.running:
                frames = self.socket.recv_multipart(copy=False)
                token_ids = self.decoder.decode(frames)
                result = self.pool_worker.lookup_scheduler(
                    token_ids, use_layerwise)
                response = result.to_bytes(4, "big")
                self.socket.send(response)

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        self.socket.close(linger=0)
        # TODO: close the thread!
