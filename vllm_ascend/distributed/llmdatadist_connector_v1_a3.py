import msgspec
from dataclasses import dataclass

from typing import Optional, Any
import torch
import math
from vllm import envs
from vllm.config import VllmConfig
from collections.abc import Iterator
import json
import zmq
import threading
import contextlib
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
  KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole
)
from vllm.forward_context import ForwardContext
from vllm.distributed.parallel_state import (
  get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
  get_tp_group, get_world_group
)
from vllm.config import KVTransferConfig
from vllm.utils import round_down, get_ip
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import RequestStatus
from .llmdatadist_connector_v1 import TORCH_DTYPE_TO_NPU_DTYPE
from vllm.v1.request import Request
from vllm.utils import logger

from llm_datadist import LLMDataDist, LLMRole, CacheDesc, BlocksCacheKey, LLMConfig, LLMException

GET_META_MSG = b"get_meta_msg"

class LLMDataDistAgentMetadata(msgspec.Struct):
  super_pod_id: str
  server_id: str
  device_id: str
  device_ip: str
  super_device_id: str
  cluster_id: str

@dataclass
class ReqMeta:
  local_block_ids: list[int]
  remote_block_ids: list[int]
  remote_host: str
  remote_port: str
  remote_cluster_id: str

class LLMDataDistConnectorMetadata(KVConnectorMetadata):

  def __init__(self):
     self.requests: dict[str, ReqMeta] = {}

  def add_new_req(
        self,
        request_id: str,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any]
  ):
    self.requests[request_id] = ReqMeta(
        local_block_ids=local_block_ids,
        remote_block_ids=kv_transfer_params["remote_block_ids"],
        remote_cluster_id=kv_transfer_params["remote_cluster_id"],
        remote_host=kv_transfer_params["remote_host"],
        remote_port=kv_transfer_params["remote_port"],
    )

class LLMDataDistConnectorA3(KVConnectorBase_V1):
  def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
    if role == KVConnectorRole.SCHEDULER:
      self.connector_scheduler: Optional[LLMDataDistConnectorScheduler] = LLMDataDistConnectorScheduler(vllm_config)
    elif role == KVConnectorRole.WORKER:
      self.connector_scheduler = None
      self.connector_worker = LLMDataDistConnectorWorker(vllm_config)
  
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

  def get_finished(self,
                    finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
      """Get the finished recving and sending requests."""
      assert self.connector_worker is not None
      return self.connector_worker.get_finished()

  def start_load_kv(self, forward_context: "ForwardContext",
                    **kwargs) -> None:
      assert self.connector_worker is not None
      assert isinstance(self._connector_metadata, LLMDataDistConnectorMetadata)
      self.connector_worker.start_load_kv(self._connector_metadata)

  def wait_for_layer_load(self, layer_name: str) -> None:
      """LLMDataDistConnector does not do layerwise saving, the load is in blocking manager."""
      pass

  def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                    attn_metadata: "AttentionMetadata", **kwargs) -> None:
      """LLMDataDistConnector does not save explicitly."""
      pass

  def wait_for_save(self):
      """LLMDataDistConnector does not save explicitly."""
      pass


class LLMDataDistConnectorScheduler():

  def __init__(self, vllm_config: VllmConfig, cluster_id: int):
     self.vllm_config = vllm_config
     self.block_size = vllm_config.cache_config.block_size
     self.cluster_id = cluster_id
     self.local_ip = get_ip()
     self.local_rank = get_world_group().local_rank

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
        f"LLMDataDistConnector get_num_new_matched_tokens: num_computed_tokens={num_computed_tokens}, kv_transfer_params={params}")

    if params is not None and params.get("do_remote_prefill"):
        # Remote prefill: get all prompt blocks from remote.
        assert num_computed_tokens % self.block_size == 0
        rounded_num_prompt_tokens = round_down(
            len(request.prompt_token_ids), self.block_size)
        count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
        return count, count > 0

    # No remote prefill for this request.
    return 0, False

  def update_state_after_alloc(
    self,
    request: Request,
    blocks: KVCacheBlocks,
    num_externel_tokens: int):
    params = request.kv_transfer_params
    logger.debug(
      f"LLMDataDistConnector update states num_externel_tokens: {num_externel_tokens} kv_transfer_params: {params}"
     )
    if params is not None and params.get("do_remote_prefill"):
      if all(p in params for p in ("remote_engine_id", "remote_host", "remote_port")):
        self._reqs_need_recv[request.request_id] = (request, blocks.get_unhashed_block_ids())
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
    meta = LLMDataDistConnectorMetadata()

    for req_id, (req, block_ids) in self._reqs_need_recv.items():
      assert req.kv_transfer_params is not None
      meta.add_new_req(
        request_id=req_id,
        local_block_ids=block_ids,
        kv_transfer_params=req.kv_transfer_params
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
        "LLMDataDistConnector request_finished, request_status=%s, "
        "kv_transfer_params=%s", request.status, params)

    if (params is None or not params.get("do_remote_decode")
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED):
        return False, None

    # NIXL transfer the full block only, but I don't see any reason to do that, so here
    # we just transfer any data that computed from prefill node
    # note: there might be some issue on this, check it if there is any unexpected result

    # all_full = request.num_computed_tokens % self.block_size == 0
    # computed_block_ids = block_ids if all_full else block_ids[:-1]
    computed_block_ids = block_ids
    # If prompt < block_size, no xfer so free blocks immediately.
    delay_free_blocks = len(computed_block_ids) > 0

    return delay_free_blocks, dict(
        do_remote_prefill=True,
        do_remote_decode=False,
        remote_block_ids=computed_block_ids,
        remote_engine_id=self.cluster_id,
        remote_host=self.local_ip,
        remote_port=envs.VLLM_LLMDD_CHANNEL_PORT + self.local_rank,
    )

class LLMDataDistConnectorWorker():
  """
  Implementation of Worker side methods
  """
  def __init__(
        self,
        vllm_config: VllmConfig):
      logger.info("Initialize the LLMDataDistConnectorWorker")
      self.local_rank = get_world_group().local_rank
      self.rank = get_world_group().rank
      self.local_ip = get_ip()
      self.kv_transfer_config: Optional[KVTransferConfig] = vllm_config.kv_transfer_config
      self.local_agent_metadata: Optional[LLMDataDistAgentMetadata] = None

      self.llm_datadist_role = None
      if self.kv_transfer_config.kv_role is "kv_producer":
        self.llm_datadist_role = LLMRole.PROMPT
      elif self.kv_transfer_config.kv_role is "kv_consumer":
        self.llm_datadist_role = LLMRole.DECODER
      else:
        raise RuntimeError(f"LLMDataDistWorker: Receive unexpected kv role in LLMDataDistWorker, this worker now only suppoert kv_producer and kv_consumer, but receiving {vllm_config.kv_transfer_config.kv_role}")

      # linked_cluster record the cluster that already build the connection its format should be {"cluster_id": "comm_name"}
      self.linked_cluster = {}
      self.prefill_device_list = []
      self.decode_device_list = []
      global_rank_table = self.read_offline_rank_table()
      self.local_agent_metadata = self.read_agent_metadata(global_rank_table, self.local_ip, self.local_rank)
      self.llm_datadist = LLMDataDist(self.llm_datadist_role, self.local_agent_metadata.cluster_id)
      self.init_llm_datadist()
      remote_ip, remote_rank = self.get_remote_ip_and_rank()
      for idx in range(len(remote_ip)):
        remote_agent_meta = self.read_agent_metadata(global_rank_table, remote_ip[idx], remote_rank[idx])
        self.add_remote_agent(remote_agent_meta)


  def listen_for_agent_metadat_req(self, event: threading.Event):
    port = envs.VLLM_LLMDD_CHANNEL_PORT + self.local_rank
    url = f"tcp://{self.local_ip}:{port}"
    msg_encoder = msgspec.msgpack.Encoder()
    msg_decoder = msgspec.msgpack.Decoder()
    msg_to_send = msg_encoder.encode(self.local_agent_metadata)
    logger.debug(f"The local agent metadata have {len(msg)} bytes here")
    with zmq_ctx(zmq.ROUTER, url) as sock:
      event.set()
      while True:
        identity, _, msg = sock.recv_multipart()
        decode_msg = msg_decoder.decode(msg)
        if isinstance(decode_msg, LLMDataDistAgentMetadata):
          self.add_remote_agent(decode_msg)
        else:
          logger.warning(f"LLMDataDistConnectorWorker: receiving unrecognized data {decode_msg}")
        sock.send_multipart((identity, b"", msg_to_send))


  def init_llm_datadist(self):
    llm_config = LLMConfig()
    llm_config.device_id = self.local_rank
    llm_config.sync_kv_timeout = 1000
    llm_config.enable_switch_role = True
    llm_config.enable_cache_manager = True
    llm_config_options = llm_config.generate_options()
    self.llm_datadist.init(llm_config_options)
    self.cache_manager = self.llm_datadist.cache_manager
    logger.info(f"Done initialize llm_datadist in rank {self.rank}, local rank {self.local_rank}, cluster id {self.cluster_id}")


  def read_offline_rank_table(self):
    assert (
        envs.DISAGGREGATED_RPEFILL_RANK_TABLE_PATH
    ), "Please set path of rank_table to env variable DISAGGREGATED_RPEFILL_RANK_TABLE_PATH"
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


  def read_agent_metadata(self, global_rank_table, server_id, device_id):
    devices_type_list = []
    agent_metadata = None
    if self.llm_datadist_role == LLMRole.PROMPT:
      devices_type_list.append("prefill_device_list")
    elif self.llm_datadist_role == LLMRole.Decoder:
      devices_type_list.append("decode_device_list")
    else:
      devices_type_list.append("prefill_device_list")
      devices_type_list.append("decode_device_list")
    for device_type in devices_type_list:
      if self.local_agent_metadata is not None:
        break
      device_list = global_rank_table[device_type]
      for device_info in device_list:
        if device_info["server_id"] != server_id:
          continue
        if device_info["device_id"] != device_id:
          continue
        super_pod_id_ = device_info["super_pod_id"]
        server_id_ = device_info["server_id"]
        device_id_ = device_info["device_id"]
        device_ip_ = device_info["device_ip"]
        super_device_id_ = device_info["super_device_id"]
        cluster_id_ = device_info["cluster_id"]
        agent_metadata = LLMDataDistAgentMetadata(
          super_pod_id=super_pod_id_,
          server_id=server_id_,
          device_id=device_id_,
          device_ip=device_ip_,
          super_device_id=super_device_id_,
          cluster_id=cluster_id_,
        )
        break
    assert agent_metadata is not None, f"Can't read the target server_id {server_id} and device_id {device_id} from rank table"
    return agent_metadata

  def get_remote_ip_and_rank(self):
    local_info = (self.local_ip, self.local_rank)
    remote_device_ids = []
    remote_ranks = []
    if self.llm_datadist_role == LLMRole.PROMPT:
      remote_device_list = self.decode_device_list
      device_list = self.prefill_device_list
    elif self.llm_datadist_role == LLMRole.DECODER:
      remote_device_list = self.prefill_device_list
      device_list = self.decode_device_list
    else:
      raise RuntimeError(f"kv_both role in LLMDataDist is not supported now")
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


  def init_cluster_info(self, global_rank_table):
    if self.kv_transfer_config.kv_role is "kv_producer":
      self.cluster_id = global_rank_table["prefill_device_list"][self.rank]["cluster_id"]
    else:
      self.cluster_id = global_rank_table["decode_device_list"][self.rank]["cluster_id"]


  def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    _, first_kv_cache = next(iter(kv_caches.items()))
    kv_cache_dtype = first_kv_cache.dtype()
    use_mla = len(first_kv_cache) == 3
    if use_mla:
        # MLA case.
        self.num_blocks = first_kv_cache.shape[0]
        block_rank = 2  # [block_size, latent_dim]
        block_shape = first_kv_cache.shape[-block_rank:]
    else:
        # [2 (k and v), num_blocks, ...]
        self.num_blocks = first_kv_cache.shape[1]
        block_rank = 3  # [block_size, kv_heads, head_dim]
        block_shape = first_kv_cache.shape[-block_rank:]

    self.block_len = math.prod(block_shape)
    self.cache_addr = []
    for cache_or_caches in kv_caches.values():
      cache_list = [cache_or_caches] if use_mla else cache_or_caches
      for cache in cache_list:
        base_addr = cache.data_ptr()
        self.cache_addr.append(base_addr)
    # register paged kv cache into the llm_cache manager
    self.cache_desc = CacheDesc(len(self.cache_addr), [self.num_blocks, self.block_len], TORCH_DTYPE_TO_NPU_DTYPE[kv_cache_dtype])
    self.cache_key = BlocksCacheKey(cluster_id=self.cluster_id)
    try:
      self.cache = self.cache_manager.register_blocks_cache(self.cache_desc, self.cache_addr, self.cache_key)
      logger.info("LLMDataDistWorker: End of register Paged Cache.")
    except (TypeError, ValueError) as e:
      raise RuntimeError(f"LLMDataDistConnectorWorker: Passing unexpected parameter to register_block_cache, receiving [cache_desc: {self.cache_desc}, cache_addr: {self.cache_addr}, cache_key: {self.cache_key}]")
    self.ready_event = threading.Event()
    self.metadata_agent_listener_t = threading.Thread(
      target=self.listen_for_agent_metadat_req,
      args=(self.ready_event),
      daemon=True,
      name="metadata_agent_listener")
    self.metadata_agent_listener_t.start()
    self.ready_event.wait()

  def start_load_kv(self, metadata: LLMDataDistConnectorMetadata):
    for req_id, meta in metadata.requests.items():
      logger.debug(f"Start to transmit {req_id}")
      self._read_blocks(
        meta.local_block_ids,
        meta.remote_block_ids,
        meta.remote_host,
        meta.remote_port,
        meta.remote_cluster_id,
        req_id
      )

  def add_remote_agent(self, metadata: LLMDataDistAgentMetadata) -> bool:
    remote_cluster_id = metadata.cluster_id
    if remote_cluster_id in self.linked_cluster:
      logger.debug(f"LLMDataDistConnectorWorker: remote cluster_id: {metadata.cluster_id} already linked with this server, skip the connection")
      return False
    remote_super_pod_id = metadata.super_pod_id
    remote_device_id = metadata.device_id
    remote_device_ip = metadata.device_ip
    remote_super_device_id = metadata.super_device_id
    remote_server_id = metadata.server_id
    is_same_server = remote_server_id == self.local_agent_metadata.server_id
    is_same_pod = remote_super_pod_id == self.local_agent_metadata.super_pod_id
    # remote_rank_id = remote_cluster_id
    comm_name = f"pd_comm_{remote_device_ip}_{self.local_agent_metadata.device_ip}"
    local_rank_in_table = 0 if remote_server_id > self.local_agent_metadata.server_id else 1
    remote_rank_in_table = 1 if remote_server_id > self.local_agent_metadata.server_id else 0
    cluster_rank_info = {
       self.local_agent_metadata.cluster_id: local_rank_in_table,
       remote_cluster_id: remote_rank_in_table,
    }
    rank_table = {}
    rank_table["version"] = "1.2"
    rank_table["server_count"] = "1" if remote_server_id == self.local_agent_metadata.server_id else "2"
    rank_table["status"] = "completed"
  
    # generate server_list for rank table
    rank_table["server_list"] = []
    local_server_device_info = {
      "device": [
        {
          "device_id": self.local_agent_metadata.device_id,
          "device_ip": self.local_agent_metadata.device_ip,
          "super_device_id": self.local_agent_metadata.super_device_id,
          "rank_id": local_rank_in_table
        }
      ],
      "server_id": self.local_agent_metadata.server_id
    }
    if is_same_server:
      local_server_device_info["device"].append(
        {
          "device_id": remote_device_id,
          "device_ip": remote_device_ip,
          "super_device_id": remote_super_device_id,
          "rank_id": remote_rank_in_table
        }
      )
    else:
      remote_server_device_info = {
        "device": [
          {
            "device_id": remote_device_id,
            "device_ip": remote_device_ip,
            "super_device_id": remote_super_device_id,
            "rank_id": remote_rank_in_table
          }
        ],
        "server_id": remote_server_id
      }
      rank_table["server_list"].append(remote_server_device_info)
    rank_table["server_list"].append(local_server_device_info)

    # generate super_pod_list for rank table
    super_pod_list = []
    remote_super_pod_info = {
       "super_pod_id": remote_super_pod_id,
       "server_list": [
          {"server_id": remote_server_id}
        ],
    }
    if is_same_pod and not is_same_server:
      remote_super_pod_info["server_list"].append(
        {"server_id": self.local_agent_metadata.server_id}
      )
    super_pod_list.append(remote_super_pod_info)
    if not is_same_pod:
      local_super_pod_info = {
        "super_pod_id": self.local_agent_metadata.super_pod_id,
        "server_list": [
          {"server_id": self.local_agent_metadata.server_id}
        ],
      }
      super_pod_list.append(local_super_pod_info)
    rank_table["super_pod_list"] = super_pod_list
    comm_id = self.llm_datadist.link(comm_name, cluster_rank_info, rank_table)
    self.linked_cluster.update({remote_cluster_id: comm_id})
    logger.info(f"Sucessfully build link with cluster id {remote_cluster_id} with cluster name {comm_name} !")
    return True


  def remove_remote_agent(self, cluster_id: int):
    if cluster_id not in self.linked_cluster:
      logger.warning(f"LLMDataDistConnectorWorker: Warning! Can't remove remote client with cluster id {cluster_id} for its not exist in linked_cluster list")
    comm_id = self.linked_cluster[cluster_id]
    try:
      self.llm_datadist.unlink(comm_id)
      self.linked_cluster.pop(cluster_id)
    except LLMException:
      logger.error(f"Try to remove remote client with cluster id {cluster_id} failed!, program won't terminate, but please carefully check your environment")
    logger.info(f"Successfully remove remote client with cluster id {cluster_id} !")


  def connect_to_remote_agent(
    self,
    host: str,
    port: int
  ):
    url = f"tcp://{host}:{port + self.local_rank}"
    logger.debug(f"Querying metadata from url: {url}")
    msg_encoder = msgspec.msgpack.Encoder()
    msg_send = msg_encoder.encode(self.local_agent_metadata)
    with zmq_ctx(zmq.REQ, url) as sock:
      sock.send(msg_send)
      metadata_bytes = sock.recv()
      decoder = msgspec.msgpack.Decoder(LLMDataDistAgentMetadata)
      metadata = decoder.decode(metadata_bytes)
      sucess = self.add_remote_agent(metadata)

  def _read_blocks(
    self,
    local_block_ids:  list[int],
    remote_block_ids: list[int],
    remote_ip: str,
    remote_port: int,
    remote_cluster_id: str,
    request_id: str,
  ):
    if remote_cluster_id not in self.linked_cluster:
      self.connect_to_remote_agent(remote_ip, remote_port)
    num_local_blocks = len(local_block_ids)
    if num_local_blocks == 0:
       return 
    num_remote_blocks = len(remote_block_ids)
    assert num_local_blocks <= num_remote_blocks
    if num_local_blocks < num_remote_blocks:
      remote_block_ids = remote_block_ids[-num_local_blocks:]

    remote_cache_key = BlocksCacheKey(cluster_id=remote_cluster_id)
    try:
      self.cache_manager.pull_blocks(remote_cache_key, self.cache, local_block_ids, remote_block_ids)
    except (TypeError, ValueError) as e:
      raise RuntimeError(f"LLMDataDistConnectorWorker: Passing unexpected parameter to pull_blocks remote_cache_key: {remote_cache_key}, cache: {self.cache}, local_block_ids: {local_block_ids}, remote_block_ids: {remote_block_ids}")
    except LLMException:
      raise RuntimeError(f"LLMDataDistConnectorWorker: Timeout during pull_blocks, you can try to increase the sync_kv_timeout config or checking your connect status")




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