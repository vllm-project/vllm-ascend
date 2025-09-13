import threading
from enum import Enum
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Dict, List, Tuple, Union
import msgspec
import torch
import zmq
import threading

from concurrent.futures import Future

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group)
from vllm.utils import logger
from vllm.utils import make_zmq_path, make_zmq_socket, round_down, get_ip,cdiv
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.core.sched.output import SchedulerOutput
from vllm_ascend.distributed.memcache.memcache_engine import MemcacheEngine
from vllm.v1.request import Request
from vllm.forward_context import ForwardContext
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm_ascend.distributed.memcache.config_data import MemcacheConnectorMetadata,RequestTracker,LoadSpec,ReqMeta



class MemcacheConnectorV1(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self.kv_role = vllm_config.kv_transfer_config.kv_role

        self.use_layerwise=vllm_config.kv_transfer_config.kv_connector_extra_config.get("use_layerwise", False)

        self.kv_caches: dict[str, torch.Tensor] = {}

        self._block_size = vllm_config.cache_config.block_size

        self.skip_last_n_tokens = vllm_config.kv_transfer_config.get_from_extra_config(
            "skip_last_n_tokens", 1
        )

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MemcacheConnectorV1Scheduler(vllm_config, self.skip_last_n_tokens, self.use_layerwise) 
        else:
            self.connector_worker = MemcacheEngine(
                vllm_config,
                self.use_layerwise,
                self.skip_last_n_tokens,
            )

            assert self.connector_worker is not None
            if vllm_config.parallel_config.rank == 0:
                self.lookup_server = MemcacheLookupServer(
                    self.connector_worker, vllm_config, self.use_layerwise
                )
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
        attn_metadata = forward_context.attn_metadata
        # if attn_metadata is None:
        #     logger.warning("In connector.start_load_kv, but the attn_metadata is None")
        #     return
        assert self.connector_worker is not None
        assert isinstance(self._get_connector_metadata(), MemcacheConnectorMetadata)
        self.connector_worker.start_load_kv(self._get_connector_metadata())
        
    def wait_for_layer_load(self, layer_name: str) -> None:
        """MemcacheConnector does not do layerwise saving."""
        if not self.use_layerwise:
            return
        self.connector_worker.wait_for_layer_load()

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """MemcacheConnector does not save explicitly."""
        if not self.use_layerwise:
            return
        
        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        self.connector_worker.save_kv_layer(self._get_connector_metadata())

    def wait_for_save(self):
        """MemcacheConnector does not save explicitly."""
        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return

        if self.use_layerwise:
            self.connector_worker.wait_layer_transfer_finish()
            return
        
        self.connector_worker.wait_for_save(self._get_connector_metadata())
        # time.sleep(1)

    def get_finished(self,
                        finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
            """Get the finished recving and sending requests."""
            assert self.connector_worker is not None
            return self.connector_worker.get_finished()


def get_zmq_rpc_path_memcache(
    vllm_config: Optional["VllmConfig"] = None,
) -> str:
    base_url = envs.VLLM_RPC_BASE_PATH
    # Default to 0 if not configured
    rpc_port = 0
    if vllm_config is not None:
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "memcache_rpc_port", 0
        )
    logger.debug("Base URL: %s, RPC Port: %s", base_url, rpc_port)
    return f"ipc://{base_url}/memcache_rpc_port_{rpc_port}"


class MemcacheConnectorV1Scheduler:
    def __init__(self, vllm_config: "VllmConfig", skip_last_n_tokens, use_layerwise):
        self.client=MemcacheLookupClient(vllm_config)
        self.use_layerwise=use_layerwise
        self.kv_role = vllm_config.kv_transfer_config.kv_role
                # request_id -> (vllm cached tokes, memcache cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}
        self.skip_last_n_tokens = skip_last_n_tokens
        self._block_size = vllm_config.cache_config.block_size
                # request_id -> full_token_ids
        self._request_trackers: dict[str, RequestTracker] = {}
                # Whether to discard partial chunks
        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", False
            )
        )
        self._unfinished_requests: dict[str, Request] = {}
    
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self._discard_partial_chunks:
            token_block_end = len(request.prompt_token_ids) // self._block_size * self._block_size
            token_ids = torch.tensor(request.prompt_token_ids[:token_block_end])
        else:
            token_ids = torch.tensor(request.prompt_token_ids)
        num_external_hit_tokens = self.client.lookup(token_ids)

        if num_external_hit_tokens == request.num_tokens:
            num_external_hit_tokens -= 1

        need_to_allocate = num_external_hit_tokens - num_computed_tokens

        logger.info(
            "Reqid: %s, Total tokens %d, memcache hit tokens: %d, need to load: %d",
            request.request_id,
            request.num_tokens,
            num_external_hit_tokens,
            need_to_allocate,
        )
        
        if need_to_allocate <= 0:
            return 0, False
        
        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            memcache_cached_tokens=num_external_hit_tokens,
            can_load=False,
        )

        return need_to_allocate, not self.use_layerwise

    def update_state_after_alloc(self, request: "Request", blocks:"KVCacheBlocks", num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """
        local_block_ids=[]
        if num_external_tokens > 0:
            local_block_ids = blocks.get_block_ids()[0]

        self._unfinished_requests[request.request_id] = (
                        request, local_block_ids)
        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            return

        assert (
            num_external_tokens > 0
            and num_external_tokens
            == self.load_specs[request.request_id].memcache_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
        ), (
            f"Mismatch in number of tokens: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].memcache_cached_tokens} - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens}"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        force_skip_save = self.kv_role == "kv_consumer"
        
        meta = MemcacheConnectorMetadata()

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
        
        for request in scheduler_output.scheduled_new_reqs:
            # Right now, we only load KV for new requests
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            request_tracker = RequestTracker.from_new_request(
                request, num_tokens_to_compute
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                load_spec=load_spec,
                skip_save=force_skip_save,
                is_last_chunk=len(request_tracker.token_ids)>=len(request.prompt_token_ids),
                discard_partial_chunks=self._discard_partial_chunks,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        if isinstance(cached_reqs, list):
            for i, req in enumerate(cached_reqs):
                request_tracker = self._request_trackers[req.req_id]
                request_tracker.update(req.new_token_ids, req.new_block_ids)

                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    load_spec=None,
                    skip_save=force_skip_save,
                    is_last_chunk=len(request_tracker.token_ids)>=len(req.prompt_token_ids),
                    discard_partial_chunks=self._discard_partial_chunks,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)
        else:
            for i, req_id in enumerate(cached_reqs.req_ids):
                request_tracker = self._request_trackers[req_id]
                num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
                req_tuple = self._unfinished_requests.get(req_id)
                if req_tuple:  
                    request = req_tuple[0]
                    num_current_tokens = len(request_tracker.token_ids)
                    new_token_ids = request.all_token_ids[
                        num_current_tokens : num_current_tokens + num_new_tokens
                    ]
                else:
                    raise ValueError(
                        f"Request {req_id} is not in _unfinished_requests, "
                        f"but it is scheduled to be cached"
                    )
                new_block_ids = cached_reqs.new_block_ids[i]
                if not new_block_ids:
                    continue
                request_tracker.update(new_token_ids, new_block_ids)
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    load_spec=None,
                    skip_save=force_skip_save,
                    is_last_chunk=len(request_tracker.token_ids)>=len(request.prompt_token_ids),
                    discard_partial_chunks=self._discard_partial_chunks,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)

        request_ids = [req.req_id for req in scheduler_output.scheduled_new_reqs]
        for request_id, (request,block_ids) in self._unfinished_requests.items():
            if not request_id in request_ids and not request_id in cached_reqs.req_ids:
                load_spec = self.load_specs.pop(request_id, None)
                if not load_spec:
                    continue
                num_tokens_to_compute = load_spec.memcache_cached_tokens
                if (num_tokens_to_compute % self._block_size != 0) and (num_tokens_to_compute == len(request.prompt_token_ids) - 1):
                    num_tokens_to_compute = num_tokens_to_compute + 1
                request_tracker = RequestTracker(
                    req_id=request_id,
                    token_ids=request.prompt_token_ids[:num_tokens_to_compute].copy(),
                    allocated_block_ids=block_ids,
                    num_saved_tokens=0,
                )

                self._request_trackers[request_id] = request_tracker
 
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    load_spec=load_spec,
                    skip_save=None,
                    discard_partial_chunks=self._discard_partial_chunks,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        if self.kv_role == "kv_consumer":
            return False, None
        delay_free_blocks = len(block_ids) > 0
        if delay_free_blocks:
            logger.info("Delaying free of %d blocks for request %s",
                        len(block_ids), request.request_id)
        return delay_free_blocks, None


class MemcacheLookupClient:
    def __init__(self, vllm_config: "VllmConfig"):
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_memcache(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )

    def lookup(self, token_ids: torch.Tensor) -> int:
        request = self.encoder.encode(token_ids)
        self.socket.send_multipart(request, copy=False)
        resp = self.socket.recv()
        result = int.from_bytes(resp, "big")
        return result

    def close(self):
        self.socket.close(linger=0)


class MemcacheLookupServer:
    def __init__(
        self,
        memcache_engine: MemcacheEngine,
        vllm_config: "VllmConfig",
        use_layerwise: bool,
    ):
        self.decoder = MsgpackDecoder(torch.Tensor)
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_memcache(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

        self.memcache_engine = memcache_engine
        self.running = True

        def process_request():
            while self.running:
                frames = self.socket.recv_multipart(copy=False)
                token_ids = self.decoder.decode(frames)
                result = self.memcache_engine.lookup(token_ids, use_layerwise)
                response = result.to_bytes(4, "big")
                self.socket.send(response)

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        self.socket.close(linger=0)
        # TODO: close the thread!
