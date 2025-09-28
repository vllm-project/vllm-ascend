import contextlib
import json
import math
import os
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Tuple

import llm_datadist  # type: ignore
import msgspec
import torch
import zmq
from llm_datadist import (BlocksCacheKey, CacheDesc, LLMConfig, LLMDataDist,
                          LLMException, LLMRole)
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import get_tp_group, get_world_group
from vllm.forward_context import ForwardContext
from vllm.utils import get_ip, logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus

import vllm_ascend.envs as envs_ascend
from vllm_ascend.utils import AscendSocVersion, get_ascend_soc_version
from vllm_ascend.utils import context_parallel_enable, sequence_parallel_enable
if context_parallel_enable():
    from vllm.distributed.parallel_state import get_context_model_parallel_rank

TORCH_DTYPE_TO_NPU_DTYPE = {
    torch.half: llm_datadist.DataType.DT_FLOAT16,
    torch.float16: llm_datadist.DataType.DT_FLOAT16,
    torch.bfloat16: llm_datadist.DataType.DT_BF16,
    torch.float: llm_datadist.DataType.DT_FLOAT,
    torch.float32: llm_datadist.DataType.DT_FLOAT,
    torch.int8: llm_datadist.DataType.DT_INT8,
    torch.int64: llm_datadist.DataType.DT_INT64,
    torch.int32: llm_datadist.DataType.DT_INT32
}


class LLMDataDistCMgrEvent(Enum):
    ReqForMetadata = 0
    ReqForFinished = 1


class LLMDataDistCMgrAgentMetadata(msgspec.Struct):
    super_pod_id: str
    server_id: str
    device_id: str
    device_ip: str
    super_device_id: str
    cluster_id: int


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: str
    engine_id: str
    remote_tp_size: str
    remote_cp_size: str
    remote_sp_size: str


class LLMDataDistCMgrConnectorMetadata(KVConnectorMetadata):

    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}

    def add_new_req(self, request_id: str, local_block_ids: list[int],
                    kv_transfer_params: dict[str, Any]):
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            engine_id=kv_transfer_params["remote_engine_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            remote_tp_size=kv_transfer_params["remote_tp_size"],
            remote_cp_size=kv_transfer_params["remote_cp_size"],
            remote_sp_size=kv_transfer_params["remote_sp_size"],
        )


class LLMDataDistCMgrConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        assert vllm_config.kv_transfer_config is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: Optional[
                LLMDataDistCMgrConnectorScheduler] = LLMDataDistCMgrConnectorScheduler(
                    vllm_config, self.engine_id)
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = LLMDataDistCMgrConnectorWorker(vllm_config)

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
    def register_kv_caches(
            self,
            kv_caches: dict[
                str,  # type: ignore[override]
                Tuple[torch.Tensor]]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata,
                          LLMDataDistCMgrConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """LLMDataDistCMgrConnector does not do layerwise saving, the load is in blocking manager."""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata, **kwargs) -> None:
        """LLMDataDistCMgrConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        """LLMDataDistCMgrConnector does not save explicitly."""
        pass


class LLMDataDistCMgrConnectorScheduler():

    def __init__(self, vllm_config: VllmConfig, engine_id: Optional[str]):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id = engine_id
        self.local_ip = get_ip()
        # Can not retrieve the parallel config since it is not initialized.
        self.local_dp_rank = None
        self.tp_size = None
        dp_rank_local = self.vllm_config.parallel_config.data_parallel_rank_local
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        self.cp_size = self.vllm_config.parallel_config.context_parallel_size if context_parallel_enable() else 1
        self.sp_size = tp_size if (sequence_parallel_enable() and self.vllm_config.parallel_config.enable_sequence_parallel) else 1

        self.port = dp_rank_local * self.cp_size * tp_size + envs_ascend.VLLM_ASCEND_LLMDD_RPC_PORT if dp_rank_local is not None else tp_size + envs_ascend.VLLM_ASCEND_LLMDD_RPC_PORT

        self._reqs_need_recv: dict[str, tuple[Request, list[int]]] = {}

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the 
            external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
            asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            f"LLMDataDistCMgrConnector get_num_new_matched_tokens: num_computed_tokens={num_computed_tokens}, kv_transfer_params={params}"
        )

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert num_computed_tokens % self.block_size == 0
            # Note: We use the full token count as transmit data here.
            count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
            return count, count > 0

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks,
                                 num_externel_tokens: int):
        params = request.kv_transfer_params
        logger.debug(
            f"LLMDataDistCMgrConnector update states num_externel_tokens: {num_externel_tokens} kv_transfer_params: {params}"
        )
        if params is not None and params.get("do_remote_prefill"):
            if params.get("remote_block_ids"):
                if all(p in params for p in ("remote_engine_id", "remote_host",
                                             "remote_port", "remote_tp_size")):
                    self._reqs_need_recv[request.request_id] = (
                        request, blocks.get_unhashed_block_ids())
                else:
                    logger.warning("" \
                    f"Invalid KVTransferParams {params}, This request will be discard")
            else:
                assert num_externel_tokens == 0
            params["do_remote_prefill"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = LLMDataDistCMgrConnectorMetadata()

        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(request_id=req_id,
                             local_block_ids=block_ids,
                             kv_transfer_params=req.kv_transfer_params)
        self._reqs_need_recv.clear()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:

        params = request.kv_transfer_params
        logger.debug(
            "LLMDataDistCMgrConnector request_finished, request_status=%s, "
            "kv_transfer_params=%s", request.status, params)

        if (params is None or not params.get("do_remote_decode")
                or request.status != RequestStatus.FINISHED_LENGTH_CAPPED):
            return False, None

        # note: NIXL transfer the full block only, but I don't see any reason to do that, so here
        # we just transfer any data that computed from prefill node
        # note: there might be some issue on this, check it if there is any unexpected result
        computed_block_ids = block_ids
        delay_free_blocks = len(computed_block_ids) > 0
        if delay_free_blocks:
            logger.info("Delaying free of %d blocks for request %s",
                        len(computed_block_ids), request.request_id)
        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            remote_host=self.local_ip,
            remote_port=self.port,
            remote_tp_size=str(
                self.vllm_config.parallel_config.tensor_parallel_size),
            remote_cp_size=str(self.cp_size),
            remote_sp_size=str(self.sp_size),
        )


class LLMDataDistCMgrConnectorWorker():
    """
  Implementation of Worker side methods
  """

    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.kv_transfer_config is not None
        logger.info("Initialize the LLMDataDistCMgrConnectorWorker")
        # we assume the local node only contains dp and tp, and tp will not communicate inter-node.
        # for any scenario beyond this scope, the functionality of this connector is not guaranteed.
        self.local_rank_on_node = get_world_group().rank % (
            vllm_config.parallel_config.data_parallel_size_local *
            vllm_config.parallel_config.tensor_parallel_size)
        self.local_rank = get_world_group().local_rank
        self.local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.tp_rank = get_tp_group().rank_in_group
        self.rank = get_world_group().rank
        self.cp_size = vllm_config.parallel_config.context_parallel_size if context_parallel_enable() else 1
        self.cp_rank = get_context_model_parallel_rank() if context_parallel_enable() else 0
        self.sp_size = self.tp_size if (sequence_parallel_enable() and vllm_config.parallel_config.enable_sequence_parallel) else 1
        self.local_ip = get_ip()
        self.kv_transfer_config: KVTransferConfig = vllm_config.kv_transfer_config
        self.local_agent_metadata: Optional[
            LLMDataDistCMgrAgentMetadata] = None
        self.vllm_config = vllm_config
        self.executor = ThreadPoolExecutor(1)
        self.thread_lock = threading.Lock()

        self.llm_datadist_role = None
        self.llm_datadist_remote_role = None
        if self.kv_transfer_config.kv_role == "kv_producer":
            self.llm_datadist_role = LLMRole.PROMPT
            self.llm_datadist_remote_role = LLMRole.DECODER
        elif self.kv_transfer_config.kv_role == "kv_consumer":
            self.llm_datadist_role = LLMRole.DECODER
            self.llm_datadist_remote_role = LLMRole.PROMPT
        else:
            raise RuntimeError(
                f"LLMDataDistWorker: Receive unexpected kv role in LLMDataDistWorker, this worker now only support kv_producer and kv_consumer, but receiving {vllm_config.kv_transfer_config.kv_role}"
            )

        # linked_cluster record the cluster that already build the connection its format should be {"cluster_id": "comm_name"}
        self.linked_cluster: dict[Any, Any] = {}
        self.prefill_device_list: list[tuple[int, int]] = []
        self.decode_device_list: list[tuple[int, int]] = []
        global_rank_table = self.read_offline_rank_table()
        self.local_agent_metadata = self.read_agent_metadata(global_rank_table)
        self.llm_datadist = LLMDataDist(self.llm_datadist_role,
                                        self.local_agent_metadata.cluster_id)
        self.init_llm_datadist()
        self.finished_reqs: set[str] = set()
        self.soc_info = get_ascend_soc_version()
        # Set hccl deterministic for model execute
        os.environ["HCCL_DETERMINISTIC"] = "true"
        self.done_receiving_counts: defaultdict[str,
                                                set[int]] = defaultdict(set)

    def listen_for_agent_metadata_req(self, event: threading.Event):
        assert self.local_agent_metadata is not None
        port = envs_ascend.VLLM_ASCEND_LLMDD_RPC_PORT + self.local_dp_rank * self.cp_size * self.tp_size + self.cp_rank * self.tp_size + self.tp_rank \
            if self.local_dp_rank is not None else envs_ascend.VLLM_ASCEND_LLMDD_RPC_PORT + self.tp_size + self.tp_rank
        url = f"tcp://{envs_ascend.VLLM_ASCEND_LLMDD_RPC_IP}:{port}"
        msg_encoder = msgspec.msgpack.Encoder()
        msg_decoder = msgspec.msgpack.Decoder()
        msg_to_send = msg_encoder.encode(self.local_agent_metadata)
        logger.debug(f"Start to listen to address: {url}")
        logger.debug(
            f"The local agent metadata have {len(msg_to_send)} bytes here")
        logger.info(
            f"LLMDataDistCMgrConnectorWorker: Cluster {self.local_agent_metadata.cluster_id} start to listen request from peers"
        )
        with zmq_ctx(zmq.ROUTER, url) as sock:  # type: ignore[attr-defined]
            event.set()
            while True:
                identity, _, msg = sock.recv_multipart()
                event_msg, decode_msg = msg_decoder.decode(msg)
                event_msg = LLMDataDistCMgrEvent(event_msg)
                if event_msg == LLMDataDistCMgrEvent.ReqForMetadata:
                    if "cluster_id" in decode_msg:
                        decode_msg = LLMDataDistCMgrAgentMetadata(**decode_msg)
                        logger.info(
                            f"LLMDataDistCMgrConnectorWorker: Receive message from cluster {decode_msg.cluster_id}"
                        )
                        sock.send_multipart((identity, b"", msg_to_send))
                        self.add_remote_agent(decode_msg)
                    else:
                        logger.warning(
                            f"LLMDataDistCMgrConnectorWorker: receiving unrecognized data {decode_msg}"
                        )
                elif event_msg == LLMDataDistCMgrEvent.ReqForFinished:
                    finished_req_id = decode_msg[0]
                    with self.thread_lock:
                        logger.debug(
                            f"LLMDataDistCMgrConnectorWorker: Receiving request {finished_req_id} finished"
                        )
                        self.finished_reqs.add(finished_req_id)
                    sock.send_multipart(
                        (identity, b"", b"receiving decode finished"))
                else:
                    raise RuntimeError(
                        f"LLMDataDistCMgrConnectorWorker: Receiving unexpected request event {event_msg} from remote !"
                    )

    def init_llm_datadist(self):
        assert self.local_agent_metadata is not None
        llm_config = LLMConfig()
        llm_config.device_id = self.local_rank
        llm_config.sync_kv_timeout = 20000
        llm_config.enable_switch_role = True
        llm_config.enable_cache_manager = True
        llm_config.enable_remote_cache_accessible = True
        llm_config_options = llm_config.generate_options()
        self.llm_datadist.init(llm_config_options)
        self.cache_manager = self.llm_datadist.cache_manager
        logger.info(
            f"Done initialize llm_datadist in rank {self.rank}, local rank {self.local_rank}, cluster id {self.local_agent_metadata.cluster_id}"
        )

    def read_offline_rank_table(self):
        assert (
            envs_ascend.DISAGGREGATED_PREFILL_RANK_TABLE_PATH
        ), "Please set path of rank_table to env variable DISAGGREGATED_PREFILL_RANK_TABLE_PATH"
        rank_table_path = envs_ascend.DISAGGREGATED_PREFILL_RANK_TABLE_PATH
        with open(rank_table_path, "r", encoding="utf-8") as f:
            global_rank_table = json.load(f)
        decode_device_list = global_rank_table["decode_device_list"]
        for decode_device in decode_device_list:
            server_id = decode_device["server_id"]
            device_id = decode_device["device_id"]
            self.decode_device_list.append((server_id, device_id))
        prefill_device_list = global_rank_table["prefill_device_list"]
        for prefill_device in prefill_device_list:
            server_id = prefill_device["server_id"]
            device_id = prefill_device["device_id"]
            self.prefill_device_list.append((server_id, device_id))

        # global_rank_table = json.dumps(global_rank_table)
        return global_rank_table

    @staticmethod
    def _get_visible_devices() -> Callable[[str], bool]:
        """
        Return a test function that check if the given device ID is visible.
        i.e. ASCEND_RT_VISIBLE_DEVICES is not set or contains the device_id.
        """
        visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
        if not visible_devices:
            return lambda device_id: True
        visible_device_list = visible_devices.split(",")
        return lambda device_id: device_id in visible_device_list

    def read_agent_metadata(self, global_rank_table):
        device_filter = LLMDataDistCMgrConnectorWorker._get_visible_devices()
        devices_type_list = []
        agent_metadata = None
        if self.llm_datadist_role == LLMRole.PROMPT:
            devices_type_list.append("prefill_device_list")
        elif self.llm_datadist_role == LLMRole.DECODER:
            devices_type_list.append("decode_device_list")
        else:
            devices_type_list.append("prefill_device_list")
            devices_type_list.append("decode_device_list")
        for device_type in devices_type_list:
            device_list = global_rank_table[device_type]
            device_list = [
                d for d in device_list if d.get("server_id") == self.local_ip
                and device_filter(d.get("device_id", ""))
            ]
            if len(device_list) <= self.cp_rank * self.tp_size + self.tp_rank:
                continue
            device_info = device_list[self.cp_rank * self.tp_size + self.tp_rank]
            super_pod_id_ = device_info.get("super_pod_id", None)
            server_id_ = device_info["server_id"]
            device_id_ = device_info["device_id"]
            device_ip_ = device_info["device_ip"]
            super_device_id_ = device_info.get("super_device_id", None)
            cluster_id_ = int(device_info["cluster_id"])
            agent_metadata = LLMDataDistCMgrAgentMetadata(
                super_pod_id=super_pod_id_,
                server_id=server_id_,
                device_id=device_id_,
                device_ip=device_ip_,
                super_device_id=super_device_id_,
                cluster_id=cluster_id_,
            )
        assert agent_metadata is not None, f"Can't read the target server_id {self.local_ip} and device_rank {self.rank} from rank table"
        return agent_metadata

    def register_kv_caches(self, kv_caches: dict[str, Tuple[torch.Tensor]]):
        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache = first_kv_cache_tuple[0]
        assert len(first_kv_cache_tuple) > 1
        assert self.local_agent_metadata is not None
        kv_cache_dtype = first_kv_cache.dtype
        self.use_mla: bool = first_kv_cache_tuple[0].size(
            -1) != first_kv_cache_tuple[1].size(-1)
        # MLA case. [2 (k_normed, k_pe), num_blocks, ...]
        # MHA case. [2 (k and v), num_blocks, ...]
        self.num_blocks = first_kv_cache.shape[0]
        block_rank = 3  # [block_size, latent_dim]
        block_shape = first_kv_cache.shape[-block_rank:]

        self.block_len = math.prod(block_shape)
        self.cache_addr: list[int] = []
        alignment = 2 * 1024 * 1024
        if self.use_mla:
            cache_k_normed_addr_list = []
            cache_k_pe_addr_list = []
            k_normed = None
            k_pe = None
            for cache_or_caches in kv_caches.values():
                assert len(cache_or_caches) > 1
                k_normed, k_pe = cache_or_caches[0], cache_or_caches[1]
                cache_k_normed_addr_list.append(k_normed.data_ptr())
                cache_k_pe_addr_list.append(k_pe.data_ptr())
            self.cache_addr = (cache_k_normed_addr_list, cache_k_pe_addr_list)

            cache_desc_k_normed = CacheDesc(
                len(self.cache_addr[0]), [*k_normed.shape],
                TORCH_DTYPE_TO_NPU_DTYPE[kv_cache_dtype])
            cache_desc_k_pe = CacheDesc(
                len(self.cache_addr[1]), [*k_pe.shape],
                TORCH_DTYPE_TO_NPU_DTYPE[kv_cache_dtype])
            cache_key_k_normed = BlocksCacheKey(cluster_id=int(
                self.local_agent_metadata.cluster_id),
                                                model_id=0)
            cache_key_k_pe = BlocksCacheKey(cluster_id=int(
                self.local_agent_metadata.cluster_id),
                                            model_id=1)
            self.cache_desc = (cache_desc_k_normed, cache_desc_k_pe)
            self.cache_key = (cache_key_k_normed, cache_key_k_pe)
            try:
                cache_k_normed = self.cache_manager.register_blocks_cache(
                    self.cache_desc[0], self.cache_addr[0], self.cache_key[0])
                cache_k_pe = self.cache_manager.register_blocks_cache(
                    self.cache_desc[1], self.cache_addr[1], self.cache_key[1])
                self.cache = (cache_k_normed, cache_k_pe)
                logger.info("LLMDataDistWorker: End of register Paged Cache.")
            except (TypeError, ValueError):
                raise RuntimeError(
                    f"LLMDataDistCMgrConnectorWorker: Passing unexpected parameter to register_block_cache, receiving [cache_desc: {self.cache_desc}, cache_addr: {self.cache_addr}, cache_key: {self.cache_key}]"
                )
        else:
            for cache_or_caches in kv_caches.values():
                for cache in cache_or_caches:
                    base_addr = cache.data_ptr()
                    assert base_addr % alignment == 0, "The address of the registered kv cache should be aligned to 2M"
                    self.cache_addr.append(base_addr)
            # register paged kv cache into the llm_cache manager
            self.cache_desc = CacheDesc(
                len(self.cache_addr), [*cache.shape],
                TORCH_DTYPE_TO_NPU_DTYPE[kv_cache_dtype])
            self.cache_key = BlocksCacheKey(
                cluster_id=int(self.local_agent_metadata.cluster_id))
            logger.info(
                f"num of cache: {len(self.cache_addr)}, size of cache: {[*cache.shape]}, real size of cache: {first_kv_cache.shape}"
            )
            try:
                self.cache = self.cache_manager.register_blocks_cache(
                    self.cache_desc, self.cache_addr, self.cache_key)
                logger.info(
                    "LLMDataDistCMgrConnectorWorker: End of register Paged Cache."
                )
            except (TypeError, ValueError):
                raise RuntimeError(
                    f"LLMDataDistCMgrConnectorWorker: Passing unexpected parameter to register_block_cache, receiving [cache_desc: {self.cache_desc}, cache_addr: {self.cache_addr}, cache_key: {self.cache_key}]"
                )
        self.ready_event = threading.Event()
        self.metadata_agent_listener_t = threading.Thread(
            target=self.listen_for_agent_metadata_req,
            args=(self.ready_event, ),
            daemon=True,
            name="metadata_agent_listener")
        self.metadata_agent_listener_t.start()
        self.ready_event.wait()

    def start_load_kv(self, metadata: LLMDataDistCMgrConnectorMetadata):
        futures = []
        for req_id, meta in metadata.requests.items():
            logger.debug(f"Start to transmit {req_id}")
            future = self.executor.submit(
                self._read_blocks,
                local_block_ids=meta.local_block_ids,
                remote_block_ids=meta.remote_block_ids,
                remote_ip=meta.remote_host,
                remote_port=int(meta.remote_port),
                remote_engine_id=meta.engine_id,
                request_id=req_id,
                remote_tp_size=meta.remote_tp_size,
                remote_cp_size=meta.remote_cp_size,
                remote_sp_size=meta.remote_sp_size,
            )
            futures.append(future)

        def handle_exception(future):
            if future.exception():
                logger.error(f"KV transfer task failed: {future.exception()}")

        for future in futures:
            future.add_done_callback(handle_exception)

    def add_remote_agent(self, metadata: LLMDataDistCMgrAgentMetadata) -> int:
        assert self.local_agent_metadata is not None
        remote_cluster_id = metadata.cluster_id
        if remote_cluster_id in self.linked_cluster:
            logger.debug(
                f"LLMDataDistCMgrConnectorWorker: remote cluster_id: {metadata.cluster_id} already linked with this server, skip the connection"
            )
            return remote_cluster_id
        remote_super_pod_id = metadata.super_pod_id
        remote_server_id = metadata.server_id
        is_same_server = remote_server_id == self.local_agent_metadata.server_id
        is_same_pod = remote_super_pod_id == self.local_agent_metadata.super_pod_id
        if self.llm_datadist_role == LLMRole.PROMPT:
            prefill_metadata = self.local_agent_metadata
            decode_metadata = metadata
        else:
            prefill_metadata = metadata
            decode_metadata = self.local_agent_metadata
        comm_name = f"pd_comm_{prefill_metadata.device_ip}_{decode_metadata.device_ip}"
        cluster_rank_info = {
            prefill_metadata.cluster_id: 0,
            decode_metadata.cluster_id: 1
        }
        rank_table = {}
        rank_table["version"] = "1.2"
        rank_table["server_count"] = "1" if is_same_server else "2"
        rank_table["status"] = "completed"

        # generate server_list for rank table
        rank_table["server_list"] = []  # type: ignore[assignment]
        decode_server_device_info = None
        prefill_server_device_info = {
            "device": [{
                k: v
                for k, v in [(
                    "device_id", prefill_metadata.device_id
                ), ("device_ip", prefill_metadata.device_ip
                    ), ("super_device_id",
                        prefill_metadata.super_device_id), ("rank_id", "0")]
                if v is not None
            }],
            "server_id":
            prefill_metadata.server_id
        }
        if is_same_server:
            prefill_server_device_info["device"].append(      # type: ignore[attr-defined]
                {
                    k: v
                    for k, v in [(
                        "device_id", decode_metadata.device_id
                    ), ("device_ip", decode_metadata.device_ip
                        ), ("super_device_id",
                            decode_metadata.super_device_id), ("rank_id", "1")]
                    if v is not None
                })
        else:
            decode_server_device_info = {
                "device": [{
                    k: v
                    for k, v in [(
                        "device_id", decode_metadata.device_id
                    ), ("device_ip", decode_metadata.device_ip
                        ), ("super_device_id",
                            decode_metadata.super_device_id), ("rank_id", "1")]
                    if v is not None
                }],
                "server_id":
                decode_metadata.server_id
            }
        rank_table["server_list"].append(  # type: ignore[attr-defined]
            prefill_server_device_info)
        if decode_server_device_info is not None:
            rank_table["server_list"].append(  # type: ignore[attr-defined]
                decode_server_device_info)

        if self.soc_info == AscendSocVersion.A3:
            # generate super_pod_list for rank table
            super_pod_list = []
            prefill_super_pod_info = {
                "super_pod_id": prefill_metadata.super_pod_id,
                "server_list": [{
                    "server_id": prefill_metadata.server_id
                }],
            }
            if is_same_pod and not is_same_server:
                prefill_super_pod_info[
                    "server_list"].append(  # type: ignore[attr-defined]
                        {"server_id": decode_metadata.server_id})
            super_pod_list.append(prefill_super_pod_info)
            if not is_same_pod:
                decode_super_pod_id = {
                    "super_pod_id": decode_metadata.super_pod_id,
                    "server_list": [{
                        "server_id": decode_metadata.server_id
                    }],
                }
                super_pod_list.append(decode_super_pod_id)
            rank_table[
                "super_pod_list"] = super_pod_list  # type: ignore[assignment]
        logger.info(
            f"LLMDataDistCMgrConnectorWorker: try link with remote, comm id: {comm_name}"
        )
        logger.info(f"rank table \n{rank_table}")
        logger.info(f"comm name: {comm_name}")
        logger.info(f"cluster rank info: {cluster_rank_info}")
        comm_id = self.llm_datadist.link(comm_name, cluster_rank_info,
                                         json.dumps(rank_table))
        while True:
            ret = self.llm_datadist.query_register_mem_status(comm_id=comm_id)
            if ret == llm_datadist.RegisterMemStatus.OK:
                logger.info(
                    f"LLMDataDistCMgrConnectorWorker: Linking success, comm id: {comm_id}"
                )
                break
            elif ret == llm_datadist.RegisterMemStatus.FAILED:
                raise RuntimeError(
                    f"LLMDataDistCMgrConnectorWorker: Linking failed, comm id: {comm_id}"
                )
            time.sleep(1)
            logger.info("Checking query_register_mem_status again")
        self.linked_cluster.update({remote_cluster_id: comm_id})
        logger.info(f"cached linked cluster: {self.linked_cluster}")
        logger.info(
            f"Successfully build link with cluster id {remote_cluster_id} with cluster name {comm_name} !"
        )
        return remote_cluster_id

    def remove_remote_agent(self, cluster_id: int):
        if cluster_id not in self.linked_cluster:
            logger.warning(
                f"LLMDataDistCMgrConnectorWorker: Warning! Can't remove remote client with cluster id {cluster_id} for its not exist in linked_cluster list"
            )
        comm_id = self.linked_cluster[cluster_id]
        try:
            self.llm_datadist.unlink(comm_id)
            self.linked_cluster.pop(cluster_id)
        except LLMException:
            logger.error(
                f"Try to remove remote client with cluster id {cluster_id} failed!, program won't terminate, but please carefully check your environment"
            )
        logger.info(
            f"Successfully remove remote client with cluster id {cluster_id} !"
        )

    def connect_to_remote_agent(self, host: str, port: int) -> int:
        url = f"tcp://{host}:{port}"
        logger.debug(f"Querying metadata from url: {url}")
        msg_encoder = msgspec.msgpack.Encoder()
        msg_send = msg_encoder.encode(
            [LLMDataDistCMgrEvent.ReqForMetadata, self.local_agent_metadata])
        with zmq_ctx(zmq.REQ, url) as sock:  # type: ignore[attr-defined]
            logger.info("Try request remote metadata from socket......")
            sock.send(msg_send)
            metadata_bytes = sock.recv()
            decoder = msgspec.msgpack.Decoder()
            metadata = decoder.decode(metadata_bytes)
            metadata = LLMDataDistCMgrAgentMetadata(**metadata)
            logger.info(f"recving metadata: {metadata}")
            cluster_id = self.add_remote_agent(metadata)
        return cluster_id

    def send_finish_to_remote(self, host: str, ports: list[int], request_id):
        for port in ports:
            url = f"tcp://{host}:{port}"
            logger.debug(f"Sending finished to remote: {url}")
            msg_encoder = msgspec.msgpack.Encoder()
            msg_send = msg_encoder.encode(
                [LLMDataDistCMgrEvent.ReqForFinished, [request_id]])
            with zmq_ctx(zmq.REQ, url) as sock:  # type: ignore[attr-defined]
                try:
                    sock.send(msg_send)
                    logger.debug(
                        f"Request id {request_id} finished message send to remote {url}"
                    )
                    _ = sock.recv()
                except Exception as e:
                    logger.error(
                        f"Failed to send reqest_id {request_id} to prefill: {e}"
                    )

    def _read_blocks(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_ip: str,
        remote_port: int,
        remote_engine_id: str,
        request_id: str,
        remote_tp_size: str,
        remote_cp_size: str,
        remote_sp_size: str,
    ):
        remote_cp_size = int(remote_cp_size)
        remote_sp_size = int(remote_sp_size)
        remote_tp_size = int(remote_tp_size)
        if self.cp_size == remote_cp_size and self.sp_size == remote_sp_size:
            # same as original, P cpi_spj -> D cpi_spj
            remote_kv_num = 1
            # remote_ports = [remote_port + self.tp_rank % int(remote_tp_size)]
            remote_ports = list(
                range(remote_port + self.tp_rank,
                    remote_port + int(remote_tp_size), self.tp_size))
            num_remote_blocks = [len(remote_block_ids)]
        elif self.cp_size == 1 and self.sp_size == 1:
            # only cp/sp in P, each D needs to pull from cp*sp P (to all-gather kv_cache)
            remote_kv_num = remote_cp_size * remote_sp_size
            # remote_ports = [remote_port + offset for offset in range(remote_cp_size * remote_sp_size)]
            remote_ports = [remote_port + self.tp_rank + offset for offset in
                            range(0, remote_cp_size * remote_tp_size, remote_tp_size // remote_sp_size)]
            # recompute cp/sp block assign here, maybe we can also pass it from P node meta
            num_local_blocks = len(local_block_ids)
            num_remote_blocks = [num_local_blocks // (remote_cp_size * remote_sp_size)] * remote_cp_size * remote_sp_size
            num_remain_blocks = num_local_blocks % (remote_cp_size * remote_sp_size)
            for i in range(num_remain_blocks):
                num_remote_blocks[i] += 1
        else:
            raise NotImplementedError('cp/sp resharding not supported now, only support D cp/sp == 1 or D cp/sp == P cp/sp')

        local_block_offset = 0
        remote_block_ids_full = remote_block_ids
        local_block_ids_full = local_block_ids
        for remote_kv_id in range(remote_kv_num):
            remote_port = remote_ports[remote_kv_id]
            num_blocks_to_pull = num_remote_blocks[remote_kv_id]
            if num_blocks_to_pull == 0:
                continue
            remote_block_ids = remote_block_ids_full[:num_blocks_to_pull]
            local_block_ids = local_block_ids_full[local_block_offset:local_block_offset+num_blocks_to_pull]
            local_block_offset += num_blocks_to_pull
            # if remote_ip not in self.linked_cluster:
            # tp_offset = self.tp_rank % int(remote_tp_size)
            # remote_cluster_id = self.connect_to_remote_agent(
            #     remote_ip, remote_port + tp_offset)
            remote_cluster_id = self.connect_to_remote_agent(remote_ip, remote_port)
            # TODO maybe this part is for prefix cache, not considered now, need to check
            assert len(local_block_ids) > 0
            assert len(local_block_ids) == len(remote_block_ids)
            """
            num_local_blocks = len(local_block_ids)
            if num_local_blocks == 0:
                return
            num_remote_blocks = len(remote_block_ids)
            assert num_local_blocks <= num_remote_blocks
            if num_local_blocks < num_remote_blocks:
                remote_block_ids = remote_block_ids[-num_local_blocks:]
            """

            logger.info(f"remote cluster id is: {remote_cluster_id}")
            if self.use_mla:
                remote_cache_key_k_normed = BlocksCacheKey(
                    cluster_id=remote_cluster_id, model_id=0)
                remote_cache_key_k_pe = BlocksCacheKey(
                    cluster_id=remote_cluster_id, model_id=1)
                logger.info("Try pull blocks from remote server")
                try:
                    self.cache_manager.pull_blocks(
                        remote_cache_key_k_normed,
                        self.cache[0],  # type: ignore[has-type]
                        remote_block_ids,
                        local_block_ids)
                    self.cache_manager.pull_blocks(
                        remote_cache_key_k_pe,
                        self.cache[1],  # type: ignore[has-type]    
                        remote_block_ids,
                        local_block_ids)
                except (TypeError, ValueError):
                    raise RuntimeError(
                        f"LLMDataDistCMgrConnectorWorker: Passing unexpected parameter to pull_blocks remote_cache_key: {remote_cache_key_k_normed} {remote_cache_key_k_pe}, cache: {self.cache}, local_block_ids: {local_block_ids}, remote_block_ids: {remote_block_ids}"  # type: ignore[has-type]
                    )
                except LLMException:
                    raise RuntimeError(
                        "LLMDataDistCMgrConnectorWorker: Timeout during pull_blocks, you can try to increase the sync_kv_timeout config or checking your connect status"
                    )
            else:
                remote_cache_key = BlocksCacheKey(cluster_id=remote_cluster_id)
                logger.info("Try pull blocks from remote server")
                try:
                    self.cache_manager.pull_blocks(
                        remote_cache_key,
                        self.cache,  # type: ignore[has-type]
                        remote_block_ids,
                        local_block_ids)
                except (TypeError, ValueError):
                    raise RuntimeError(
                        f"LLMDataDistCMgrConnectorWorker: Passing unexpected parameter to pull_blocks remote_cache_key: {remote_cache_key}, cache: {self.cache}, local_block_ids: {local_block_ids}, remote_block_ids: {remote_block_ids}"  # type: ignore[has-type]
                    )
                except LLMException:
                    raise RuntimeError(
                        "LLMDataDistCMgrConnectorWorker: Timeout during pull_blocks, you can try to increase the sync_kv_timeout config or checking your connect status"
                    )
        self.send_finish_to_remote(remote_ip, remote_ports, request_id)
        with self.thread_lock:
            self.finished_reqs.add(request_id)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """Get the finished recving and sending requuests."""
        import copy
        with self.thread_lock:
            req_ids_to_ret = copy.deepcopy(self.finished_reqs)
            self.finished_reqs.clear()
        if self.llm_datadist_role == LLMRole.PROMPT:
            return req_ids_to_ret, None
        else:
            return None, req_ids_to_ret


# adopt this from  https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py
@contextlib.contextmanager
def zmq_ctx(socket_type: Any,
            addr: str) -> Iterator[zmq.Socket]:  # type: ignore[name-defined]
    """Context manager for a ZMQ socket"""

    ctx: Optional[zmq.Context] = None  # type: ignore[name-defined]
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]

        if socket_type == zmq.ROUTER:  # type: ignore[attr-defined]
            socket = ctx.socket(zmq.ROUTER)  # type: ignore[attr-defined]
            socket.bind(addr)
        elif socket_type == zmq.REQ:  # type: ignore[attr-defined]
            socket = ctx.socket(zmq.REQ)  # type: ignore[attr-defined]
            socket.connect(addr)
        else:
            raise ValueError(f"Unexpected socket type: {socket_type}")

        yield socket
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)
