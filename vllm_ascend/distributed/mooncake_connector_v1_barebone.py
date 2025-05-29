from dataclasses import dataclass
import msgspec
import math
from typing import Optional, Any, Tuple, Union
import torch
from vllm_ascend import envs
from vllm.config import VllmConfig
from collections.abc import Iterator
import json
import zmq
import threading
import contextlib
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from enum import Enum
from vllm.forward_context import ForwardContext
from vllm.distributed.parallel_state import (
    get_world_group,
)
from vllm.config import KVTransferConfig
from vllm.utils import round_down, get_ip
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import RequestStatus
from vllm.v1.request import Request
from vllm.utils import logger
from typing import TYPE_CHECKING
import llm_datadist
from vllm.distributed.kv_transfer.kv_pipe.mooncake_pipe import MooncakeTransferEngine
from mooncake.engine import TransferEngine

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request

TORCH_DTYPE_TO_NPU_DTYPE = {
    torch.half: llm_datadist.DataType.DT_FLOAT16,
    torch.float16: llm_datadist.DataType.DT_FLOAT16,
    torch.bfloat16: llm_datadist.DataType.DT_BF16,
    torch.float: llm_datadist.DataType.DT_FLOAT,
    torch.float32: llm_datadist.DataType.DT_FLOAT,
    torch.int8: llm_datadist.DataType.DT_INT8,
    torch.int64: llm_datadist.DataType.DT_INT64,
    torch.int32: llm_datadist.DataType.DT_INT32,
}

GET_META_MSG = b"get_meta_msg"


class MooncakeAgentMetadata(msgspec.Struct):
    # Currently using metadata from Mooncake Metadataserver
    pass


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: str
    engine_id: str


class MooncakeRole(Enum):
    PROMPT = 1
    DECODER = 2
    MIX = 3


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}

    def add_new_req(
            self,
            request_id: str,
            local_block_ids: list[int],
            kv_transfer_params: dict[str, Any],
    ):
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,  # Not use
            remote_block_ids=kv_transfer_params["remote_block_ids"],  # Not use
            engine_id=kv_transfer_params["remote_engine_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
        )


class EnhanceMooncakeTransferEngine(MooncakeTransferEngine):
    """
    Enhance Mooncake Transfer Engine
    """

    pass


class MooncakeConnectorV1_barebone(KVConnectorBase_V1):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: Optional[MooncakeConnectorScheduler] = (
                MooncakeConnectorScheduler(vllm_config, self.engine_id)
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(vllm_config)

        ############################################################
        # Scheduler Side Methods
        ############################################################

    def get_num_new_matched_tokens(
            self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
            self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

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

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving, the load is in blocking manager."""
        pass

    def save_kv_layer(
            self,
            layer_name: str,
            kv_layer: torch.Tensor,
            attn_metadata: "AttentionMetadata",
            **kwargs,
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        """MooncakeConnector does not save explicitly."""
        pass


class MooncakeConnectorScheduler:
    def __init__(self, vllm_config: VllmConfig, engine_id: int):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id = engine_id
        self.local_ip = get_ip()

        self._reqs_need_recv: dict[str, tuple[Request, list[int]]] = {}

    # def get_num_new_matched_tokens(
    #     self, request: "Request", num_computed_tokens: int
    # ) -> tuple[int, bool]:
    #     """
    #     For remote prefill, pull all prompt blocks from remote
    #     asynchronously relative to engine execution.

    #     Args:
    #         request (Request): the request object.
    #         num_computed_tokens (int): the number of locally
    #             computed tokens for this request
    #     Returns:
    #         * the number of tokens that can be loaded from the
    #           external KV cache beyond what is already computed.
    #         * true if the external KV cache tokens will be loaded
    #           asynchronously (between scheduler steps).
    #     """

    #     params = request.kv_transfer_params
    #     logger.debug(
    #         f"MooncakeConnector get_num_new_matched_tokens: num_computed_tokens={num_computed_tokens}, kv_transfer_params={params}"
    #     )
    #     print("params:", params)
    #     if params is not None and params.get("do_remote_prefill"):
    #         # Remote prefill: get all prompt blocks from remote.
    #         assert num_computed_tokens % self.block_size == 0
    #         rounded_num_prompt_tokens = round_down(
    #             len(request.prompt_token_ids), self.block_size
    #         )
    #         count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
    #         return count, count > 0

    #     # No remote prefill for this request.
    #     return 0, False

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
            f"MooncakeConnector get_num_new_matched_tokens: num_computed_tokens={num_computed_tokens}, kv_transfer_params={params}")

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert num_computed_tokens % self.block_size == 0
            # Note: We use the full token count as transmit data here.
            count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
            return count, count > 0

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
            self, request: Request, blocks: KVCacheBlocks, num_externel_tokens: int
    ):
        params = request.kv_transfer_params
        logger.debug(
            f"MooncakeConnector update states num_externel_tokens: {num_externel_tokens} kv_transfer_params: {params}"
        )
        if params is not None and params.get("do_remote_prefill"):
            if all(
                    p in params for p in ("remote_engine_id", "remote_host", "remote_port")
            ):
                self._reqs_need_recv[request.request_id] = (
                    request,
                    blocks.get_unhashed_block_ids(),
                )
            else:
                logger.warning(
                    ""
                    f"Invalid KVTransferParams {params}, This request will be discard"
                )
        else:
            assert num_externel_tokens == 0
        params["do_remote_prefill"] = False

    def build_connector_meta(
            self,
            scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )
        self._reqs_need_recv.clear()

        return meta

    def request_finished(
            self,
            request: "Request",
            block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished, request_status=%s, "
            "kv_transfer_params=%s",
            request.status,
            params,
        )

        if (
                params is None
                or not params.get("do_remote_decode")
                or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            return False, None

        # NIXL transfer the full block only, but I don't see any reason to do that, so here
        # we just transfer any data that computed from prefill node
        # note: there might be some issue on this, check it if there is any unexpected result

        all_full = request.num_computed_tokens % self.block_size == 0
        computed_block_ids = block_ids if all_full else block_ids[:-1]
        # computed_block_ids = block_ids
        # If prompt < block_size, no xfer so free blocks immediately.
        delay_free_blocks = len(computed_block_ids) > 0

        # TODO：VLLM_LLMDD_CHANNEL_PORT -》 VLLM_MOONCAKE_CHANNEL_PORT
        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            remote_host=self.local_ip,
            remote_port=envs.VLLM_LLMDD_CHANNEL_PORT,
        )


class MooncakeConnectorWorker:
    """
    Implementation of Worker side methods
    """

    def __init__(self, vllm_config: VllmConfig):
        logger.info("Initialize the MooncakeConnectorWorker")
        self.local_rank = get_world_group().local_rank
        self.rank = get_world_group().rank
        self.local_ip = get_ip()
        self.kv_transfer_config: Optional[KVTransferConfig] = (
            vllm_config.kv_transfer_config
        )
        self.local_agent_metadata: Optional[MooncakeAgentMetadata] = None
        self.mooncake_role = None
        self.mooncake_remote_role = None
        # self.linked_cluster = {}
        # self.prefill_device_list = []
        # self.decode_device_list = []
        # get role
        if self.kv_transfer_config.kv_role == "kv_producer":
            self.mooncake_role = MooncakeRole.PROMPT
            self.mooncake_remote_role = MooncakeRole.DECODER
        elif self.kv_transfer_config.kv_role == "kv_consumer":
            self.mooncake_role = MooncakeRole.DECODER
            self.mooncake_remote_role = MooncakeRole.PROMPT
        else:
            raise RuntimeError(
                f"MooncakeWorker: Receive unexpected kv role in MooncakeWorker, this worker now only suppoert kv_producer and kv_consumer, but receiving {vllm_config.kv_transfer_config.kv_role}"
            )
        # read rank table
        # global_rank_table = self.read_offline_rank_table()

        #
        current_device_id = envs.ASCEND_RT_VISIBLE_DEVICES

        ## get remote ip and rank
        # self.remote_ip, self.remote_rank = self.get_remote_ip_and_rank()

        ## get agent_metadata
        # self.local_agent_metadata = self.read_agent_metadata()

        # init engine
        self.kv_rank = self.kv_transfer_config.kv_rank

        # TODO:currently there are some hard-coding in Transfer Engine for PD ranks
        self.engine = MooncakeTransferEngine(self.kv_rank, self.local_rank)

        self.finished_reqs = set()
        # global addr parar, formate: list of Tuple(addr, capacity)
        self.kv_cache_tensor_addrs = list()

    def read_offline_rank_table(self):
        assert envs.DISAGGREGATED_RPEFILL_RANK_TABLE_PATH, "Please set path of rank_table to env variable DISAGGREGATED_RPEFILL_RANK_TABLE_PATH"
        rank_table_path = envs.DISAGGREGATED_RPEFILL_RANK_TABLE_PATH
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

    def get_remote_ip_and_rank(self):
        # TODO return results need to check, err now
        current_device_id = envs.ASCEND_RT_VISIBLE_DEVICES
        local_info = (self.local_ip, str(current_device_id))
        remote_device_ids = []
        remote_ranks = []
        if self.mooncake_role == MooncakeRole.PROMPT:
            remote_device_list = self.decode_device_list
            device_list = self.prefill_device_list
        elif self.mooncake_role == MooncakeRole.DECODER:
            remote_device_list = self.prefill_device_list
            device_list = self.decode_device_list
        else:
            raise RuntimeError(f"kv_both role in MooncakeConnectorV1_Barebone is not supported now")
        remote_list_num = len(remote_device_list)
        list_num = len(device_list)
        local_idx = device_list.index(local_info)
        if remote_list_num >= list_num:
            for idx in range(local_idx, remote_list_num, list_num):
                device_ids, ranks = remote_device_list[idx]
                remote_device_ids.append(device_ids)
                remote_ranks.append(ranks)
        else:
            device_id, rank = remote_device_list[local_idx % remote_list_num]
            remote_device_ids.append(device_id)
            remote_ranks.append(rank)
        return remote_device_ids, remote_ranks

    def read_agent_metadata(self):
        pass

    def init_cluster_info(self, global_rank_table):
        pass

    def register_kv_caches(self, kv_caches: dict[str, Tuple[torch.Tensor]]):
        """
        Note: The kv cache exist on NPU Device.
        """
        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache = first_kv_cache_tuple[0]
        kv_elem_size = first_kv_cache.element_size()
        self.use_mla = first_kv_cache_tuple[0].size(-1) != first_kv_cache_tuple[1].size(-1)
        if self.use_mla:
            # MLA case.
            self.num_blocks = first_kv_cache.shape[0]
            block_rank = 2  # [block_size, latent_dim]
            block_shape = first_kv_cache.shape[-block_rank:]
        else:
            # [2 (k and v), num_blocks, ...]
            self.num_blocks = first_kv_cache.shape[1]
            block_rank = 3  # [block_size, kv_heads, head_dim]
            block_shape = first_kv_cache.shape[-block_rank:]

        self.block_len = kv_elem_size * math.prod(block_shape)

        # TODO: Currently 1P1D,for xPyD case, need to use engine_ids when constructing kv_cache_tensor_addrs
        # i.e. kv_cache_tensor_addrs[self.engine_ids].append()

        for cache_or_caches in kv_caches.values():
            for cache in cache_or_caches:
                base_addr = cache.data_ptr()
                region_len = self.num_blocks * self.block_len
                self.engine.register_memory(
                    base_addr, region_len
                )
                self.kv_cache_tensor_addrs.append((base_addr, region_len))

    # TODO: Currently loading all kv caches in kv_cache_tensor_addrs, need to load based on req_id and its
    # corresponding block ids. We also need to define a method for aquiring mem_addrs based on block ids given.
    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        logger.info("start to load kv")
        logger.info(f"load kv metadata is: {metadata}")

        for req_id, meta in metadata.requests.items():
            logger.info(f"Start to transmit {req_id}")
            # TODO:peer_buffer_address now set 0, offset calculated in Transfer Engine(see transfer_engine_py.cpp)
            peer_buffer_address = 0

            for base_addr, capacity in self.kv_cache_tensor_addrs:
                # Note: v1 only need to read
                self.engine.transfer_sync(
                    base_addr, peer_buffer_address, capacity
                )
            self.finished_reqs.add(req_id)

    def get_finished(
            self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """Get the finished recving and sending requuests."""
        import copy

        req_ids_to_ret = copy.deepcopy(self.finished_reqs)
        self.finished_reqs.clear()
        if self.mooncake_role == MooncakeRole.PROMPT:
            return req_ids_to_ret, None
        else:
            return None, req_ids_to_ret


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    ctx: Optional[zmq.Context] = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]

        if socket_type == zmq.ROUTER:
            socket = ctx.socket(zmq.ROUTER)
            socket.bind(addr)
        elif socket_type == zmq.REQ:
            socket = ctx.socket(zmq.REQ)
            socket.connect(addr)
        else:
            raise ValueError(f"Unexpected socket type: {socket_type}")

        yield socket
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)
