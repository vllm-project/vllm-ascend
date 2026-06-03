import threading
from typing import Any, Optional

import numpy as np
import torch
import torch_npu
import vllm.envs as envs
import zmq
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.forward_context import ForwardContext
from vllm.utils import logger, make_zmq_socket
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_ascend.attention.hstu_attention_v1 import RequestStage
from vllm_ascend.distributed.mooncake_user.backend import backend_map
from vllm_ascend.distributed.mooncake_user.mooncake_user_engine import \
    MooncakeEngine
from vllm_ascend.distributed.mooncake_user.user_config_data import (
    LoadSpec, MooncakeConnectorMetadata, MooncakeUserKey, ReqMeta,
    RequestTracker)


class MooncakeUserStoreConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        # 避免首次运行找不到报错
        self._connector_metadata = MooncakeConnectorMetadata(None)
        self.kv_role = vllm_config.kv_transfer_config.kv_role

        self.use_layerwise = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "use_layerwise", False)

        self.kv_caches: dict[str, torch.Tensor] = {}

        self._block_size = vllm_config.cache_config.block_size

        self.sended_but_unfinished_reqs: set[str] = set()

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakeStoreConnectorV1Scheduler(
                vllm_config, self.use_layerwise)
        else:
            self.connector_worker = MooncakeEngine(
                vllm_config,
                self.use_layerwise,
            )

            assert self.connector_worker is not None
            if vllm_config.parallel_config.rank == 0:
                self.lookup_server = MooncakeLookupServer(
                    self.connector_worker, vllm_config, self.use_layerwise)

        num_layers = vllm_config.model_config.hf_config.hstu_config.num_layers
        self._onload_history_kv_events = [
            torch_npu.npu.Event() for _ in range(num_layers)
        ]
        self._onload_stream = torch_npu.npu.Stream()

        # offload stream
        self._offload_history_kv_events = [
            torch_npu.npu.Event() for _ in range(num_layers)
        ]
        self._offload_stream = torch_npu.npu.Stream()

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
        assert isinstance(self._get_connector_metadata(),
                          MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._get_connector_metadata())

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeStoreConnector does not do layerwise saving."""
        if not self.use_layerwise:
            return
        self.connector_worker.wait_for_layer_load()

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """MooncakeStoreConnector does not save explicitly."""
        # breakpoint()
        if not self.use_layerwise:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        self.connector_worker.save_kv_layer(self._get_connector_metadata())

    def save_kv_cache_by_layer(self, layer_id: int, kv_cache: torch.Tensor,
                               attn_metadata: "AttentionMetadata",
                               **kwargs) -> None:
        self.connector_worker.save_kv_cache_by_layer(
            layer_id, kv_cache, attn_metadata, self._get_connector_metadata())

    def start_load_kv_cache_by_layer(self, forward_context: "ForwardContext",
                                     layer_name: str, layer_id: int,
                                     **kwargs) -> None:
        self.connector_worker.start_load_kv_cache_by_layer(
            forward_context, layer_name, layer_id,
            self._get_connector_metadata())

    def wait_for_save(self):
        """MooncakeStoreConnector does not save explicitly."""
        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return

        if self.use_layerwise:
            self.connector_worker.wait_layer_transfer_finish()
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


def get_zmq_rpc_path_mooncake(
    vllm_config: Optional["VllmConfig"] = None, ) -> str:
    base_url = envs.VLLM_RPC_BASE_PATH
    # Default to 0 if not configured
    rpc_port = 0
    if vllm_config is not None:
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "mooncake_rpc_port", 0)
    logger.debug("Base URL: %s, RPC Port: %s", base_url, rpc_port)
    return f"ipc://{base_url}/mooncake_rpc_port_{rpc_port}"


class MooncakeStoreConnectorV1Scheduler:

    def __init__(self, vllm_config: "VllmConfig", use_layerwise):
        self.client = MooncakeLookupClient(vllm_config)
        self.use_layerwise = use_layerwise
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.consumer_is_to_load = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_load", False)
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "load_async", False)
        # request_id -> (vllm cached tokes, mooncake cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}
        self._request_history_lens: dict[str, int] = {}
        self._block_size = vllm_config.cache_config.block_size
        # request_id -> full_token_ids
        self._request_trackers: dict[str, RequestTracker] = {}

        self._unfinished_requests: dict[str, tuple[Request, list[int]]] = {}
        self._unfinished_request_ids: set[str] = set()

        # 多加一个scheduler侧的m_store，会不会有冲突？
        self.backend = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "backend", "mooncake")
        self.m_store = backend_map.get(self.backend.lower())(
            vllm_config.parallel_config)
        self.history_token_ids = torch.zeros(
            (vllm_config.scheduler_config.max_model_len),
            dtype=torch.int64).to(vllm_config.device_config.device)
        history_addr = self.history_token_ids.data_ptr()
        history_size = self.history_token_ids.element_size() * np.prod(
            self.history_token_ids.shape)
        self.m_store.register_buffer(history_addr, history_size)

        self.model_name = vllm_config.model_config.model
        self.world_size = vllm_config.parallel_config.world_size
        self.worker_id = vllm_config.parallel_config.rank

    def get_uid_key(self, uid: int) -> MooncakeUserKey:
        return MooncakeUserKey(
            uid=uid,
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            value_type='token_id',
        )

    def put_history_token_ids(self, uid: int, token_ids_array: torch.Tensor):
        key = self.get_uid_key(uid)

        if self.m_store.exists(key.to_string()):
            self.m_store.remove(key.to_string())

        self.history_token_ids[:token_ids_array.shape[0]].copy_(
            token_ids_array)
        history_addr = self.history_token_ids.data_ptr()
        history_size = self.history_token_ids.element_size() * np.prod(
            token_ids_array.shape)
        self.m_store.put_from(key.to_string(), history_addr, history_size)

    def get_history_token_ids(self, uid: int):
        key = self.get_uid_key(uid)

        if not self.m_store.exists(key.to_string()):
            return []
        history_buffer = self.m_store.get_buffer(key.to_string())
        history_token_ids = np.frombuffer(history_buffer,
                                          dtype=np.int64).tolist()
        return history_token_ids

    def _normalize_uid(self, uid):
        return uid[0] if isinstance(uid, list) else uid

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
        if self.kv_role == "kv_producer":
            return 0, False

        extra_args = request.sampling_params.extra_args or {}
        uid = extra_args.get("uid")
        if uid is None:
            return 0, False
        uid = self._normalize_uid(uid)

        history_token_ids = self.get_history_token_ids(uid)
        num_external_hit_tokens = len(history_token_ids)
        if (not request.sampling_params.extra_args.get("request_stage")
                == RequestStage.Decode.value
                and num_external_hit_tokens >= request.num_prompt_tokens):
            # If external history covers the entire prompt (or exceeds it), force
            # recompute for prefill/PdMerge.  Decode skips this because the prompt
            # only contains candidates, so history_len >= candidate_len is expected.
            num_external_hit_tokens = 0
        need_to_allocate = num_external_hit_tokens

        logger.info(
            "Reqid: %s, user id %d, local hit tokens: %d, mooncake hit tokens: %d, need to load: %d",
            request.request_id,
            uid,
            num_computed_tokens,
            num_external_hit_tokens,
            need_to_allocate,
        )

        if need_to_allocate <= 0:
            return 0, False

        self._request_history_lens[
            request.request_id] = num_external_hit_tokens

        self.load_specs[request.request_id] = LoadSpec(
            mooncake_cached_tokens=num_external_hit_tokens,
            can_load=False,
            history_len=num_external_hit_tokens,
            history_token_ids=history_token_ids,
        )

        return need_to_allocate, self.load_async

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """
        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            return

        self.load_specs[request.request_id].can_load = True

    def get_history_len(self, request_id: str) -> Optional[int]:
        return self._request_history_lens.get(request_id)

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self._unfinished_request_ids.discard(finished_req_id)
            self._request_history_lens.pop(finished_req_id, None)

        meta = MooncakeConnectorMetadata(self._unfinished_request_ids)

        for request in scheduler_output.scheduled_new_reqs:
            # Right now, we only load KV for new requests
            load_spec = self.load_specs.pop(request.req_id, None)

            if load_spec is None:
                history_idx = -1 * request.sampling_params.extra_args[
                    'candidate_num'][0]
                if history_idx < 0:
                    history_token_ids = request.prompt_token_ids[:history_idx]
                    candidate_ids = request.prompt_token_ids[history_idx:]
                else:
                    history_token_ids = request.prompt_token_ids
                    candidate_ids = []
            else:
                history_token_ids = load_spec.history_token_ids
                candidate_ids = request.prompt_token_ids

            request_tracker = RequestTracker.from_new_request(
                request, history_token_ids, candidate_ids)

            if load_spec is None:
                uid = request_tracker.uid
                token_ids_array = torch.tensor(request_tracker.token_ids,
                                               dtype=torch.int64)
                self.put_history_token_ids(uid, token_ids_array)

            self._request_trackers[request.req_id] = request_tracker
            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                load_spec=load_spec,
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
        tracker = self._request_trackers.get(request.request_id)
        if tracker is not None and tracker.num_saved_tokens <= 0:
            return False, None
        delay_free_blocks = len(block_ids) > 0
        if delay_free_blocks:
            logger.info("Delaying free of %d blocks for request %s",
                        len(block_ids), request.request_id)
        return delay_free_blocks, None


class MooncakeLookupClient:

    def __init__(self, vllm_config: "VllmConfig"):
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_mooncake(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )

    def lookup(self, uid: int) -> int:
        request = self.encoder.encode(uid)
        self.socket.send_multipart(request, copy=False)
        resp = self.socket.recv()
        result = int.from_bytes(resp, "big")
        return result

    def close(self):
        self.socket.close(linger=0)


class MooncakeLookupServer:

    def __init__(
        self,
        mooncake_engine: MooncakeEngine,
        vllm_config: "VllmConfig",
        use_layerwise: bool,
    ):
        self.decoder = MsgpackDecoder(torch.Tensor)
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_mooncake(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

        self.mooncake_engine = mooncake_engine
        self.running = True

        def process_request():
            while self.running:
                frames = self.socket.recv_multipart(copy=False)
                uid = self.decoder.decode(frames)
                result = self.mooncake_engine.lookup_scheduler(
                    uid, use_layerwise)
                response = result.to_bytes(4, "big", signed=True)
                self.socket.send(response)

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        self.running = False
        self.socket.close(linger=0)
