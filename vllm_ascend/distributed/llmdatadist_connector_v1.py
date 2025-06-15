import enum
import hashlib
import json
import random
import struct
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import requests
import torch
import torch_npu
import torchair  # type: ignore
from vllm.distributed import get_tensor_model_parallel_rank, get_world_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig, KVTransferConfig
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request

import llm_datadist  # type: ignore
from llm_datadist import LLMConfig, LLMException, LLMStatusCode

import vllm_ascend.envs as envs_ascend
from vllm_ascend.attention.mla_v1 import AscendMLAMetadata

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

GLOBAL_RANKTABLE = envs_ascend.LLMDATADIST_GLOBAL_RANKTABLE


class ServerRole(enum.Enum):
    Router = "router"
    Prefill = "prefill"
    Decode = "decode"


@dataclass
class DeviceInfo:
    device_id: int
    device_ip: str
    dp_rank: int
    tp_rank: int
    cluster_id: int


@dataclass
class ServerInfo:
    server_id: str
    server_ip: str
    role: ServerRole
    devices: List[DeviceInfo]

    def get_device(self, tp_rank: int,
                   dp_rank: int) -> Union[DeviceInfo, None]:
        for device in self.devices:
            if device.tp_rank == tp_rank and device.dp_rank == dp_rank:
                return device
        return None


def get_servers_from_ranktable(ranktable_path: str, prefill_tp: int,
                               decode_tp: int) -> List[ServerInfo]:
    cluster_index = 0

    def parse_server_group(group, role: ServerRole,
                           tp_size: int) -> List[ServerInfo]:
        nonlocal cluster_index

        server_infos: List[ServerInfo] = []
        for server in group.get("server_list", []):
            server_ip = server.get("server_ip")
            server_id = server.get("server_id")

            device_infos: List[DeviceInfo] = []
            for device in server.get("device", []):
                device_id = int(device.get("device_id").strip())
                device_ip = device.get("device_ip")
                device_infos.append(
                    DeviceInfo(
                        device_id=int(device_id),
                        device_ip=device_ip,
                        dp_rank=-1,
                        tp_rank=-1,
                        cluster_id=-1,
                    ))

            # Assign dp, tp rank and unique cluster_id to all devices in this
            # server
            device_infos = sorted(device_infos, key=lambda x: x.device_id)
            for i, device_info in enumerate(device_infos):
                device_info.dp_rank = i // tp_size
                device_info.tp_rank = i % tp_size
                device_info.cluster_id = cluster_index
                cluster_index += 1

            server_infos.append(
                ServerInfo(server_id=server_id,
                           server_ip=server_ip,
                           role=role,
                           devices=device_infos))
        return server_infos

    with open(ranktable_path, "r") as file:
        rank_table = json.load(file)

    for group in rank_table.get("server_group_list", []):
        group_id = group.get("group_id", None)
        if group_id == "0":  # router
            router_servers = parse_server_group(group,
                                                ServerRole.Router,
                                                tp_size=-1)
            assert len(
                router_servers
            ) == 1, f"Must have only one server in group 0, but got {len(router_servers)}"
            router = router_servers[0]
        elif group_id == "1":  # prefill
            prefill_servers = parse_server_group(group, ServerRole.Prefill,
                                                 prefill_tp)
        elif group_id == "2":  # decode
            decode_servers = parse_server_group(group, ServerRole.Decode,
                                                decode_tp)
        else:
            raise ValueError(
                f"Unknown group_id {group_id} in server_group_list")
    return [router] + prefill_servers + decode_servers


class ClusterInfo:

    def __init__(self, vllm_config: "VllmConfig") -> None:
        # If tensor parallel (tp) and data parallel (dp) sizes are not found in
        # the extra config, use the parallel configuration of the current
        # instance as default. This is useful when the prefill and decode nodes
        # share the same parallel configuration.
        self._tp_size = vllm_config.parallel_config.tensor_parallel_size
        self._dp_size = vllm_config.parallel_config.data_parallel_size

        kv_transfer_config: "KVTransferConfig" = vllm_config.kv_transfer_config
        self._prefill_parallel_config: dict[
            str,
            Any] = kv_transfer_config.get_from_extra_config("prefill", {})
        self._decode_parallel_config: dict[
            str, Any] = kv_transfer_config.get_from_extra_config("decode", {})

        self._servers: List[ServerInfo] = get_servers_from_ranktable(
            GLOBAL_RANKTABLE, self.prefill_tp_size, self.decode_tp_size)

        self._num_prefill_instances = len(
            self.get_servers_by_role(ServerRole.Prefill))
        self._num_decode_instances = len(
            self.get_servers_by_role(ServerRole.Decode))

    def get_device(self, server_id: str, dp_rank: int,
                   tp_rank: int) -> Union[DeviceInfo, None]:
        for server in self._servers:
            if server.server_id != server_id:
                continue
            return server.get_device(tp_rank, dp_rank)
        return None

    def get_cluster_id(self, server_id: str, dp_rank: int,
                       tp_rank: int) -> int:
        device_info = self.get_device(server_id, dp_rank, tp_rank)
        if device_info is None:
            raise ValueError(
                f"Could not find device({server_id},{dp_rank},{tp_rank}) in cluster info."
            )
        return device_info.cluster_id

    def get_servers_by_role(self, role: ServerRole) -> List[ServerInfo]:
        return [server for server in self._servers if server.role == role]

    def is_1p1d(self) -> bool:
        return (self._num_prefill_instances == 1
                and self._num_decode_instances == 1)

    @property
    def router_endpoint(self):
        for server in self._servers:
            if server.role == ServerRole.Router:
                return f"http://{server.server_ip}:9000"
        raise ValueError("Router endpoint not found")

    @property
    def prefill_dp_size(self):
        candidate_keys = ["data_parallel_size", "dp_size", "dp"]
        return int(
            self._get_first_matching_value(self._prefill_parallel_config,
                                           candidate_keys, self._dp_size))

    @property
    def prefill_tp_size(self):
        candidate_keys = ["tensor_parallel_size", "tp_size", "tp"]
        return int(
            self._get_first_matching_value(self._prefill_parallel_config,
                                           candidate_keys, self._tp_size))

    @property
    def decode_dp_size(self):
        candidate_keys = ["data_parallel_size", "dp_size", "dp"]
        return int(
            self._get_first_matching_value(self._decode_parallel_config,
                                           candidate_keys, self._dp_size))

    @property
    def decode_tp_size(self):
        candidate_keys = ["tensor_parallel_size", "tp_size", "tp"]
        return int(
            self._get_first_matching_value(self._decode_parallel_config,
                                           candidate_keys, self._tp_size))

    def _get_first_matching_value(self, config_dict: dict,
                                  candidate_keys: List[str],
                                  default: Any) -> Any:
        for key in candidate_keys:
            if key in config_dict:
                return config_dict[key]
        return default


_CLUSTER_INFO: Optional["ClusterInfo"] = None


def init_cluster_info(vllm_config: "VllmConfig") -> None:
    global _CLUSTER_INFO
    if _CLUSTER_INFO is not None:
        raise ValueError("ClusterInfo is already initialized.")
    _CLUSTER_INFO = ClusterInfo(vllm_config)


def get_cluster_info() -> "ClusterInfo":
    global _CLUSTER_INFO
    if _CLUSTER_INFO is None:
        raise ValueError("ClusterInfo is not initialized.")
    return _CLUSTER_INFO


def report_prefill_info(meta_server_url, prefill_info):
    response = requests.post(f"{meta_server_url}/put", json=prefill_info)
    if response.status_code != 200:
        logger.error(
            f"put_prefill_info failed status_code: {response.status_code}, response: {response.text}"
        )


def fetch_prefill_info(meta_server_url, request_ids):
    response = requests.get(f"{meta_server_url}/get", json=request_ids)
    if response.status_code != 200:
        logger.error(
            f"get_prefill_info failed status_code: {response.status_code}, response: {response.text}"
        )
        return None
    return response.json()


class KVTransferEngine:

    def __init__(self, role: llm_datadist.LLMRole, local_rank: int,
                 dp_rank: int, tp_rank: int, local_server_id: str) -> None:
        self.role = role
        self.local_rank = local_rank
        self.tp_rank = tp_rank
        self.cluster_info = get_cluster_info()

        local_device_info = self.cluster_info.get_device(
            local_server_id, dp_rank, tp_rank)
        assert local_device_info is not None, \
            "Could not find local device from cluster info."

        self.cluster_id = local_device_info.cluster_id
        self.local_device_ip = local_device_info.device_ip
        self.datadist_engine = llm_datadist.LLMDataDist(
            self.role, self.cluster_id)

    def prepare_data_dist(self):
        buff_size = envs_ascend.LLMDATADIST_BUFFSIZE_MB * 1024 * 1024
        llm_config = LLMConfig()
        llm_config.ge_options = {
            "llm.SyncKvCacheWaitTime":
            envs_ascend.LLMDATADIST_SYNC_CACHE_WAIT_TIME,
            "ge.flowGraphMemMaxSize": f"{buff_size:d}",
            "ge.exec.deviceId": str(self.local_rank),
        }
        llm_config.buf_pool_cfg = '{"buf_cfg": [{"total_size":2097152,"blk_size":256,"max_buf_size":256}]}'
        if self.role == llm_datadist.LLMRole.PROMPT:
            llm_config.listen_ip_info = f"{self.local_device_ip}:26000"
        engine_options = llm_config.generate_options()
        self.datadist_engine.init(engine_options)
        logger.info("llm_datadist init done")
        self.kv_transfer = self.datadist_engine.kv_cache_manager

    def make_cluster(self, prefill_ip, cluster_id=-1):
        cluster = llm_datadist.LLMClusterInfo()
        cluster.remote_cluster_id = cluster_id
        cluster.append_local_ip_info(self.local_device_ip, 0)
        cluster.append_remote_ip_info(prefill_ip, 26000)
        logger.info(f"link decode ip {self.local_device_ip} -> {prefill_ip}")
        return cluster

    def make_clusters(self):
        clusters = []
        # Find all devices from prefill servers this rank need to connect
        for server in self.cluster_info.get_servers_by_role(
                ServerRole.Prefill):
            for device in server.devices:
                cluster = self.make_cluster(device.device_ip,
                                            device.cluster_id)
                clusters.append(cluster)
        return clusters


@dataclass
class ReqMeta:
    # Request ID, unique for each request
    request_id: str
    # Request tokens
    token_ids: torch.Tensor
    # Slot mappings, should have the same length as token_ids
    slot_mapping: torch.Tensor
    # Is store or load
    is_store: bool

    @staticmethod
    def make_meta(request_id: str, token_ids: list[int], block_ids: list[int],
                  block_size: int, is_store: bool) -> "ReqMeta":
        token_ids_tensor = torch.tensor(token_ids)
        valid_num_tokens = len(token_ids)
        block_ids_tensor = torch.tensor(block_ids, dtype=torch.int32)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size, dtype=torch.int32)
        slot_mapping = block_offsets.reshape(
            (1, block_size)) + block_ids_tensor.reshape(
                (num_blocks, 1)) * block_size
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return ReqMeta(
            request_id=request_id,
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
        )


@dataclass
class LLMDataDistConnectorV1Metadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(self, request_id: str, token_ids: list[int],
                    block_ids: list[int], block_size: int,
                    is_store: bool) -> None:
        self.requests.append(
            ReqMeta.make_meta(request_id, token_ids, block_ids, block_size,
                              is_store))


class LLMDataDistConnectorV1(KVConnectorBase_V1):

    def __init__(self, vllm_config: "VllmConfig",
                 role: KVConnectorRole) -> None:
        super().__init__(vllm_config=vllm_config, role=role)

        # Used by both scheduler and worker process
        kv_transfer_config: "KVTransferConfig" = self._vllm_config.kv_transfer_config
        self._block_size = vllm_config.cache_config.block_size
        if kv_transfer_config.is_kv_producer:
            self.kv_role = llm_datadist.LLMRole.PROMPT
        elif kv_transfer_config.is_kv_consumer:
            self.kv_role = llm_datadist.LLMRole.DECODER
        else:
            raise ValueError(
                "The value of kv_role must be either `kv_producer` or"
                f" `kv_consumer`, but received {kv_transfer_config.kv_role}.")

        # Used by scheduler process
        self._requests_need_load: dict[str, Request] = {}

        if role == KVConnectorRole.SCHEDULER:
            # In the scheduler process, the distributed environment is not
            # initialized. As a result, functions like `get_world_group` cannot
            # be used. Additionally, the scheduler does not require initializing
            # the KVTransferEngine. Therefore, simply return.
            return

        # Used by worker process
        init_cluster_info(self._vllm_config)
        self.cluster_info = get_cluster_info()

        if self.cluster_info.prefill_tp_size < self.cluster_info.decode_tp_size:
            raise ValueError(
                "The prefill tensor parallel size must be greater than or "
                f"equal to the decode tensor parallel size, but got "
                f"{self.cluster_info.prefill_tp_size} < "
                f"{self.cluster_info.decode_tp_size}.")

        self.local_server_id = kv_transfer_config.get_from_extra_config(
            "local_server_id", None)
        if self.local_server_id is None:
            if not self.cluster_info.is_1p1d(
            ) or self.cluster_info.prefill_dp_size != 1:
                raise ValueError(
                    "Cannot find `local_server_id` from"
                    " `kv_transfer_config.kv_connector_extra_config`.")
            # In a 1p1d configuration (1 prefill node and 1 decode node), the
            # server ID can be directly determined from the rank table based on
            # the KV role.
            servers = self.cluster_info.get_servers_by_role(
                ServerRole.Prefill if self.kv_role ==
                llm_datadist.LLMRole.PROMPT else ServerRole.Decode)
            assert len(servers) == 1, \
                f"Expected only one server for {self.kv_role}, but got {len(servers)}"
            self.local_server_id = servers[0].server_id

        self.dp_rank = self._vllm_config.parallel_config.data_parallel_rank_local
        self.tp_size = self._vllm_config.parallel_config.tensor_parallel_size
        self.tp_rank = get_tensor_model_parallel_rank()
        self.num_layers = self._vllm_config.model_config.get_num_layers(
            self._vllm_config.parallel_config)
        if self.tp_size == 1:
            local_rank = self.dp_rank
        local_rank = get_world_group().local_rank
        self.llm_datadist_engine = KVTransferEngine(self.kv_role, local_rank,
                                                    self.dp_rank, self.tp_rank,
                                                    self.local_server_id)
        self.llm_datadist_engine.prepare_data_dist()
        if self.kv_role == llm_datadist.LLMRole.DECODER:
            # Each decoding rank should correspond to each prefilling rank.
            clusters = self.llm_datadist_engine.make_clusters()
            random.shuffle(clusters)
            for cluster in clusters:
                while True:
                    link_ret, link_rets = \
                        self.llm_datadist_engine.datadist_engine.link_clusters(
                        [cluster], timeout=30_000)

                    if link_ret == LLMStatusCode.LLM_SUCCESS \
                        and link_rets[0] == LLMStatusCode.LLM_SUCCESS:
                        break

                    sleep_time = random.uniform(5, 17)
                    logger.warning(
                        f"Failed to link cluster({cluster.remote_cluster_id}), "
                        f"retrying in {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
            logger.info("Successfully linked clusters")

        # LLMDataDist will deallocate the cache buffer either when the cache
        # buffer's Python object goes out of scope or when deallocate_cache() is
        # explicitly called. This can lead to accuracy issues if the cache
        # buffer is deallocated while still being used in the NPU stream. To
        # prevent this, we maintain a reference to the cache buffer until the
        # next round, ensuring it is not prematurely deallocated.
        self.kv_buffers: List = []

        # In graph mode (migrated from v0), the layer KV cache format differs
        # from the v1 format. As a result, the KV transfer process requires
        # specific handling to accommodate this difference.
        additional_config = self._vllm_config.additional_config
        self.enable_graph_mode = additional_config and additional_config.get(
            "enable_graph_mode", False)

        if self.enable_graph_mode and \
            self.kv_role == llm_datadist.LLMRole.PROMPT:
            raise NotImplementedError(
                "The graph mode is not supported for prefill node now.")

    def _detach_kv_buffers(self):
        for kv_buffer in self.kv_buffers:
            self.llm_datadist_engine.kv_transfer.deallocate_cache(kv_buffer)
        self.kv_buffers.clear()

    def _attach_kv_buffer(self, kv_buffer: torch.Tensor):
        self.kv_buffers.append(kv_buffer)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        # Note: we recv kv cache by request now, so we do not need by
        # layer operations, all recv is done in start_load_kv

        if self.kv_role == llm_datadist.LLMRole.PROMPT:
            # In the prefilling node, do not need to load KV cache.
            return

        # Release the KV cache buffer from the previous round
        self._detach_kv_buffers()

        # Get the metadata
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LLMDataDistConnectorV1Metadata)
        assert metadata is not None, "The connector metadata should not be None."
        if len(metadata.requests) == 0:
            # No requests to load
            return

        attn_metadata = forward_context.attn_metadata
        assert attn_metadata is not None, "The attn_metadata should not be None."

        request_metadatas = {}
        for request in metadata.requests:
            if request.is_store:
                continue
            datadist_request_id = request_id_hex_to_number(request.request_id)
            target_tp_rank = self._get_target_tp_rank(datadist_request_id)
            unique_req_id = self._get_unique_req_id(request.request_id,
                                                    target_tp_rank)
            request_metadatas[request.request_id] = {
                "datadist_request_id": datadist_request_id,
                "target_tp_rank": target_tp_rank,
                "unique_req_id": unique_req_id,
            }

        unique_req_ids = [
            meta["unique_req_id"] for meta in request_metadatas.values()
        ]
        if self.cluster_info.is_1p1d(
        ) and self.cluster_info.prefill_dp_size == 1:
            # In a 1p1d configuration (1 prefill node and 1 decode node), the
            # server ID can be directly determined from the rank table based on
            # the KV role.
            servers = self.cluster_info.get_servers_by_role(ServerRole.Prefill)
            assert len(servers) == 1, \
                f"Expected only one server for {self.kv_role}, but got {len(servers)}"
            prefill_infos: Dict[str, Any] = {
                unique_req_id: {
                    "dp_rank": 0,
                    "server_id": servers[0].server_id,
                }
                for unique_req_id in unique_req_ids
            }
        else:
            prefill_infos = fetch_prefill_info(
                self.cluster_info.router_endpoint, unique_req_ids)

        # If prefill_infos is None, it indicates that get_prefill_info failed.
        # Therefore, we need to recalculate the kv cache during the decoding
        # phase. If there is a performance issue, we should consider whether
        # this is the cause.
        if prefill_infos is None:
            logger.error(
                "[rank%d][D]: Failed to get prefill info, redo model forwarding.",
                torch.distributed.get_rank())
            return None

        kv_cache_layers = []
        for _, attn_layer in forward_context.no_compile_layers.items():
            kv_cache_layer = attn_layer.kv_cache[
                forward_context.virtual_engine]
            kv_cache_layers.append(kv_cache_layer)

        if self.enable_graph_mode:
            # Currently, the graph mode is migrated from the v0, and the kv
            # cache layer is a tuple. The first element is
            # 'layer_kv_cache_nope', and the second element is
            # 'layer_kv_cache_pe'.
            assert isinstance(kv_cache_layers[0], tuple) and \
                len(kv_cache_layers[0]) == 2, (
                    "The kv_cache_layer should be a tuple of two tensors for "
                    "current graph mode.")
            layer_kv_cache_nope_shape = kv_cache_layers[0][0].shape
            layer_kv_cache_pe_shape = kv_cache_layers[0][1].shape
            kv_lora_rank = layer_kv_cache_nope_shape[-1]
            qk_rope_head_dim = layer_kv_cache_pe_shape[-1]
            kv_cache_layer_shape = list(layer_kv_cache_nope_shape[:-1]) + \
                [kv_lora_rank + qk_rope_head_dim]
            kv_hidden_dtype = kv_cache_layers[0][0].dtype
        else:
            kv_cache_layer_shape = list(kv_cache_layers[0].shape)
            kv_hidden_dtype = kv_cache_layers[0].dtype

        is_mla = isinstance(attn_metadata, AscendMLAMetadata)
        num_heads = int(kv_cache_layer_shape[-2])
        head_dim = int(kv_cache_layer_shape[-1])
        # Load the KV for each request each layer
        for request in metadata.requests:
            if request.is_store:
                continue
            # NOTE: slen is the len of kv cache need to load for this request
            # in decode, request_len =  prefill_prompt_len + 1
            slen = request.token_ids.shape[0] - 1
            req_slot_mapping = request.slot_mapping[:slen].to(device="npu")

            # For the datadist tensor, the first dimension is 1, the reason can
            # be found in wait_for_save function
            if is_mla:
                # [1, slen, num_heads, head_dim]
                kv_cache_shape: Tuple[int,
                                      ...] = (1, slen, num_heads, head_dim)
            else:
                # [1, 2, slen, num_heads, head_dim]
                kv_cache_shape = (1, 2, slen, num_heads, head_dim)

            # Each request uses the same llm_datadist request_id, which needs to
            # be converted into an integer value.
            request_metadata = request_metadatas[request.request_id]
            datadist_request_id = request_metadata["datadist_request_id"]
            target_tp_rank = request_metadata["target_tp_rank"]
            unique_req_id = request_metadata["unique_req_id"]

            dp_rank: int = prefill_infos[unique_req_id]["dp_rank"]
            server_id: str = prefill_infos[unique_req_id]["server_id"]
            remote_cluster_id = self.cluster_info.get_cluster_id(
                server_id, dp_rank, target_tp_rank)
            kv_cache_key = llm_datadist.CacheKey(remote_cluster_id,
                                                 datadist_request_id, 1)

            # pull kv cache from prefill node by request
            kv_buffer, pulled_kv_caches = self._create_cache_tensors(
                self.num_layers, kv_cache_shape, kv_hidden_dtype)
            self._attach_kv_buffer(kv_buffer)
            self.llm_datadist_engine.kv_transfer.pull_cache(
                kv_cache_key, kv_buffer, 0)

            # Check for any transmission failures; we need to redo the
            # forwarding to compute the missing states.
            if pulled_kv_caches is None:
                logger.error(
                    "[rank%d][D]: Failed to receive all KVs and hidden "
                    "states, redo model forwarding.",
                    torch.distributed.get_rank(),
                )
                # TODO: break or continue?
                break

            for layer_id, kv_cache_layer in enumerate(kv_cache_layers):
                pulled_kv_cache = pulled_kv_caches[layer_id]

                if self.enable_graph_mode:
                    pulled_kv_cache = torch.split(pulled_kv_cache,
                                                  dim=-1,
                                                  split_size_or_sections=[
                                                      kv_lora_rank,
                                                      qk_rope_head_dim
                                                  ])

                self._inject_kv_into_layer(
                    kv_cache_layer, pulled_kv_cache, req_slot_mapping, is_mla
                    and not self.enable_graph_mode)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        pass

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        # Note: we send kv cache by request now, so we do not need by
        # layer operations, all send is done in wait_for_save

        if self.kv_role == llm_datadist.LLMRole.DECODER:
            # In the decoder role, we do not need to save KV cache.
            return

        forward_context = get_forward_context()
        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, LLMDataDistConnectorV1Metadata), \
            ("metadata should be LLMDataDistConnectorV1Metadata, but got "
            f"{type(metadata)}.")

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the attn_metadata is None")
            return

        is_mla = isinstance(attn_metadata, AscendMLAMetadata)
        indices = torch.tensor([0], dtype=torch.int64, device="npu")

        prefill_info_input = {}
        # kv cache should be transferred by request
        for _, request in enumerate(metadata.requests):
            if not request.is_store:
                continue

            slen = request.token_ids.shape[0]
            req_slot_mapping = request.slot_mapping[:slen]

            uniq_req_id = self._get_unique_req_id(request.request_id,
                                                  self.tp_rank)
            prefill_info_input[uniq_req_id] = {
                "dp_rank": self.dp_rank,
                "server_id": self.local_server_id,
            }

            # Initialize LLMDatadist data structure. Each request uses the same
            # llm_datadist request_id, which needs to be converted to an integer
            # value.
            datadist_request_id = request_id_hex_to_number(request.request_id)
            kv_cache_keys = [
                llm_datadist.CacheKey(self.llm_datadist_engine.cluster_id,
                                      datadist_request_id, 1)
            ]

            if not self._need_save_kv(datadist_request_id):
                # We choose some ranks to save kv cache randomly, if the rank is
                # not selected, we do not need to save kv cache.
                continue

            kv_caches: List[torch.Tensor] = []
            for _, attn_layer in forward_context.no_compile_layers.items():
                kv_cache_layer = attn_layer.kv_cache[
                    forward_context.virtual_engine]
                kv_cache = self._extract_kv_from_layer(kv_cache_layer,
                                                       req_slot_mapping,
                                                       is_mla)
                kv_caches.append(kv_cache.detach())

            # If MLA is used, the kv_cache_shape should be (1, slen, num_heads,
            # head_dim). Otherwise, it should be (1, 2, slen, num_heads,
            # head_dim). The first dimension must be 1, because the following
            # `scatter_update_` operation will fail otherwise. The exact reason
            # for this limitation is currently unknown.
            kv_cache_shape = (1, ) + tuple(kv_caches[0].shape)
            kv_hidden_dtype = kv_caches[0].dtype
            kv_buffer, pushed_kv_caches = self._create_cache_tensors(
                self.num_layers, kv_cache_shape, kv_hidden_dtype,
                kv_cache_keys)
            for layer_idx, kv_cache in enumerate(kv_caches):
                datadist_kv_cache = pushed_kv_caches[layer_idx]
                kv_cache = kv_cache.unsqueeze(0)
                torch_npu.scatter_update_(datadist_kv_cache,
                                          indices,
                                          kv_cache,
                                          axis=-2)

            # Release reference count
            self.llm_datadist_engine.kv_transfer.deallocate_cache(kv_buffer)

        # If the cluster is configured as 1p1d (1 prefill node and 1 decode
        # node), and the data parallel size on the prefill node is 1, we don't
        # need to report the prefill information to the router. This is because
        # there is only one candidate server for the decode node to request the
        # KV cache from.
        if len(prefill_info_input) > 0 and (
                not self.cluster_info.is_1p1d()
                or self.cluster_info.prefill_dp_size != 1):
            report_prefill_info(self.cluster_info.router_endpoint,
                                prefill_info_input)
        logger.info("[rank%d][P]: KV send DONE.", torch.distributed.get_rank())

    def _inject_kv_into_layer(
        self,
        dst_kv_cache_layer: Union[torch.Tensor, tuple[torch.Tensor,
                                                      torch.Tensor]],
        pulled_kv_cache: Union[torch.Tensor, tuple[torch.Tensor,
                                                   torch.Tensor]],
        slot_mapping: torch.Tensor,
        use_siso: bool,
    ) -> None:
        """Inject the KV cache into the layer.

        Args:
            dst_kv_cache_layer (torch.Tensor): the destination KV cache
                layer. In shape [2, num_blocks, block_size, num_heads, head_dim]
                if not using MLA, [num_blocks, block_size, num_heads, head_dim]
                otherwise.
            src_kv_cache (torch.Tensor): the source KV cache. In shape
                [1, 2, num_tokens, num_heads, head_dim] if not using MLA, [1,
                num_tokens, num_heads, head_dim] otherwise.
            slot_mapping (torch.Tensor): the slot mapping. In shape
                [num_tokens].
        """
        # The `wait_for_save` function explains why the first dimension is
        # necessary.
        if isinstance(pulled_kv_cache, tuple):
            kv_cache = [cache.squeeze(0) for cache in pulled_kv_cache]
        else:
            kv_cache = pulled_kv_cache.squeeze(0)

        if use_siso:
            torch_npu._npu_reshape_and_cache_siso(
                key=kv_cache,
                key_cache=dst_kv_cache_layer,
                slot_indices=slot_mapping,
            )

        else:
            kv_cache[0] = kv_cache[0].contiguous()
            kv_cache[1] = kv_cache[1].contiguous()
            torch_npu._npu_reshape_and_cache(
                key=kv_cache[0],
                value=kv_cache[1],
                key_cache=dst_kv_cache_layer[0],
                value_cache=dst_kv_cache_layer[1],
                slot_indices=slot_mapping,
            )

    def _extract_kv_from_layer(
        self,
        kv_cache_layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        is_mla: bool,
    ) -> torch.Tensor:
        """Extract the KV cache from the layer.

        Assume the shape of the layer is [2, num_blocks, block_size, num_heads,
        head_dim] if MLA is not used, and [num_blocks, block_size, num_heads,
        head_dim] otherwise.
        """
        if is_mla:
            num_heads, head_dim = kv_cache_layer.shape[
                -2], kv_cache_layer.shape[-1]
            return kv_cache_layer.view(-1, num_heads, head_dim)[slot_mapping,
                                                                ...]

        num_heads, head_dim = kv_cache_layer.shape[-2], kv_cache_layer.shape[
            -1]
        return kv_cache_layer.view(2, -1, num_heads, head_dim)[:, slot_mapping,
                                                               ...]

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the external KV cache
        beyond the num_computed_tokens.

        Args:
            request (Request): the request object. num_computed_tokens (int):
            the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the external KV cache
            beyond what is already computed.
        """
        # NOTE: in current v1 scheduler, the num_computed_tokens is aligned with
        # the block granularity. And it expects the returned blocks and
        # num_computed_tokens to also be aligned with the block granularity.

        # NOTE: only requests in waiting queue will come here. we use datadist
        # pull cache to do transfer, so we don't align to block_size in prefill,
        # we won't have extra new matched tokens; in decode, new request kv
        # cache will be transferred from prefill, so num_computed_tokens = 0,
        # and extra new matched tokens should be len(request.prompt_token_ids) -
        # 1
        if self.kv_role == llm_datadist.LLMRole.PROMPT:
            return 0, False
        return len(request.prompt_token_ids) - 1, False

    def update_state_after_alloc(self, request: "Request",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.
        """
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output. Also,
        calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = LLMDataDistConnectorV1Metadata()

        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self._requests_need_load:
                meta.add_request(
                    request_id=new_req.req_id,
                    token_ids=new_req.prompt_token_ids,
                    block_ids=new_req.block_ids,
                    block_size=self._block_size,
                    is_store=False,
                )
                total_need_load += 1
            else:
                # NOTE: here, we set the store and load being exclusive,
                # but a single request can have both store and load.
                # NOTE(rob): for this debug implementation, we only cache
                # the original prompt tokens.
                meta.add_request(
                    request_id=new_req.req_id,
                    token_ids=new_req.prompt_token_ids,
                    block_ids=new_req.block_ids,
                    block_size=self._block_size,
                    is_store=True,
                )

        for cached_req in scheduler_output.scheduled_cached_reqs:
            # NOTE(rob): here we rely on the resumed requests being
            # the first N requests in the list scheduled_cache_reqs.
            if not cached_req.resumed_from_preemption:
                break
            raise NotImplementedError(
                "Resumed requests are not supported in this version of the "
                "connector.")

        assert total_need_load == len(self._requests_need_load)
        self._requests_need_load.clear()
        return meta

    def _create_cache_tensors(self,
                              num_layer: int,
                              shape: Tuple[int, ...],
                              dtype: torch.dtype,
                              cache_keys=[]):
        seq_len_dim_index = -2 + len(shape)
        cache_desc = llm_datadist.CacheDesc(
            num_layer,
            shape,
            TORCH_DTYPE_TO_NPU_DTYPE[dtype],
            seq_len_dim_index=seq_len_dim_index)
        # TODO(jianzs): At present, there is no method to determine the
        # available space in the mbuf memory. Therefore, we can only attempt to
        # handle allocation failures; if the failure is due to insufficient
        # space, we pause briefly before retrying until the allocation succeeds.
        while True:
            try:
                cache_buf = self.llm_datadist_engine.kv_transfer.allocate_cache(
                    cache_desc, cache_keys)
                break
            except LLMException as e:
                if e.status_code == LLMStatusCode.LLM_DEVICE_OUT_OF_MEMORY:
                    logger.warning(
                        "allocate_cache failed due to insufficient space in the"
                        " mbuf memory.")
                    time.sleep(0.03)  # wait for cache buf to be ready
                else:
                    raise e
        cache_buf_addrs = cache_buf.per_device_tensor_addrs[0]
        cache_tensors = torchair.llm_datadist.create_npu_tensors(
            cache_desc.shape, dtype, cache_buf_addrs)
        return cache_buf, cache_tensors

    def _get_unique_req_id(self, request_id: str, tp_rank: int) -> str:
        return f"{request_id}-{tp_rank}"

    def _get_prefill_tp_ranks_for_req(self, datadist_req_id: int) -> list[int]:
        """Based on the LLMDataDist request id, select a subset of tensor
        parallel ranks. Specifically, choose `decode_tp_size` ranks randomly
        from all available prefill TP ranks. These selected ranks are
        responsible for saving the KV cache for the current request."""
        if self.cluster_info.prefill_tp_size == self.cluster_info.decode_tp_size:
            return list(range(self.cluster_info.prefill_tp_size))

        rand = random.Random(datadist_req_id)
        sampled_nums = rand.sample(range(self.cluster_info.prefill_tp_size),
                                   self.cluster_info.decode_tp_size)
        return sampled_nums

    def _need_save_kv(self, datadist_req_id: int) -> bool:
        """Determines whether the current rank needs to save the KV cache for a
        given LLMDataDist request ID."""
        return self.tp_rank in self._get_prefill_tp_ranks_for_req(
            datadist_req_id)

    def _get_target_tp_rank(self, datadist_req_id: int) -> int:
        """Determines the target tensor parallel (TP) rank for a given TP rank
        and LLMDataDist request ID."""
        return self._get_prefill_tp_ranks_for_req(datadist_req_id)[
            self.tp_rank]


# ==============================
# Helper functions
# ==============================


def parse_config_string(config_string: str) -> dict:
    config_dict = {}
    parts = config_string.split(";")

    for part in parts:
        if ":" in part:
            key, values = part.split(":")
            value_parts = values.split(",")
            for value_part in value_parts:
                if "=" in value_part:
                    sub_key, sub_value = value_part.split("=")
                    config_dict[f"{key}_{sub_key}"] = int(sub_value)
                else:
                    sub_key, sub_value = value_part.split("p")
                    config_dict[f"{key}_{sub_key}p"] = int(sub_value)

    return config_dict


def string_to_int64_hash(input_str):
    """
    Hash the string using SHA-256 and convert it into an int64 integer.
    """
    hashed_bytes = hashlib.sha256(input_str.encode("utf-8")).digest()
    trunked_bytes = hashed_bytes[:8]
    uint64_value = struct.unpack("<Q", trunked_bytes)[0]
    return uint64_value


def request_id_hex_to_number(request_id: str):
    """
    Convert the hex part of a request ID string to an int64 number. For example,
    for the string "cmpl-6e1a2f3d-0", this extracts "6e1a2f3d" and converts it
    to its integer value (e.g., 1847209789).
    """
    try:
        hex_str = request_id.split("-")[1]
        if len(hex_str) != 8:
            raise ValueError(
                "The length of the hex string in request_id should be 8, but "
                f"got {len(hex_str)}.")
        return int(hex_str, 16)
    except ValueError:
        logger.warning(
            f"Invalid hex string in request_id: {request_id}. Using hash value "
            "instead.")
        return string_to_int64_hash(request_id)
