import math
import os
from typing import TYPE_CHECKING, Dict, Optional

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import get_tp_group, get_world_group
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_ascend.attention.mla_v1 import AscendMLAMetadata

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

from chariot_client import ChariotClient  # type: ignore

logger = init_logger(f"vllm.{__name__}")

CHARIOT_CONNECTION_TIMEOUT = 6000  # ms, 6 seconds as default
CHARIOT_KVCACHE_TIMEOUT = 60000  # ms, 60 seconds as default
ASYNC_SAVE_LOAD = True


class RequestMetadata:

    def __init__(self, request_id: str, token_ids: list[int],
                 block_ids: list[int], block_size: int):
        slot_mapping = torch.arange(0, block_size).unsqueeze(
            0) + torch.tensor(block_ids).unsqueeze(1) * block_size
        self.slot_mapping = slot_mapping.flatten()[:len(token_ids)]
        self.request_id = request_id
        self.is_mla = None

    @staticmethod
    def _extract_layer_index(layer_name: str) -> Optional[int]:
        """
        Extract the layer index from the layer name.
        """
        for chunk in layer_name.split("."):
            if chunk.isdigit():
                return int(chunk)
        return None

    def set_mla(self, is_mla):
        if self.is_mla:
            logger.debug(f"request metadata reset MLA enabled to {is_mla}")
        self.is_mla = is_mla

    def generate_kvcache_id(self, layer_name: str, tp_rank: int) -> str:
        """
        Construct the object ID for the KV cache layer.
        """
        layer_idx = RequestMetadata._extract_layer_index(layer_name)
        return f"{self.request_id}-{layer_idx}" if self.is_mla else f"{self.request_id}-{layer_idx}-{tp_rank}"


class ChariotConnectorMetadata(KVConnectorMetadata):

    def __init__(self):
        self.saving_requests = []
        self.loading_requests = []

    def add_load_request(self, request_id: str, token_ids: list[int],
                         block_ids: list[int], block_size: int):
        self.loading_requests.append(
            RequestMetadata(request_id, token_ids, block_ids, block_size))

    def add_save_request(self, request_id: str, token_ids: list[int],
                         block_ids: list[int], block_size: int):
        self.saving_requests.append(
            RequestMetadata(request_id, token_ids, block_ids, block_size))


class ChariotFutureWrapper:

    def __init__(self,
                 future,
                 kvcache_id: str,
                 dst_kv_layer: torch.Tensor,
                 src_kv_layer: torch.Tensor,
                 slot_mapping: torch.Tensor,
                 is_mla: bool,
                 timeout: int = CHARIOT_KVCACHE_TIMEOUT):
        self.kvcache_id = kvcache_id
        self.dst_kv_layer = dst_kv_layer
        self.src_kv_layer = src_kv_layer
        self.slot_mapping = slot_mapping
        self.is_mla = is_mla
        self.future = future
        self.timeout = timeout

    def wait_for_load(self, timeout: Optional[int] = None):
        if not self.future:  # synchronous load
            ChariotConnector.inject_kv_into_layer(self.dst_kv_layer,
                                                  self.src_kv_layer,
                                                  self.slot_mapping,
                                                  self.is_mla)
            return
        if timeout is None:
            timeout = self.timeout
        try:
            failed_ids = self.future.get(timeout)
        except TimeoutError:
            logger.error(
                f"kvcache with id {self.kvcache_id} wait for load timed out after {timeout} seconds."
            )
            return
        if failed_ids:
            logger.error(
                f"kvcache with id {self.kvcache_id} wait for load error")
        else:
            ChariotConnector.inject_kv_into_layer(self.dst_kv_layer,
                                                  self.src_kv_layer,
                                                  self.slot_mapping,
                                                  self.is_mla)

    def wait_for_save(self, timeout: Optional[int] = None):
        if not self.future:  # synchronous save
            return
        if timeout is None:
            timeout = self.timeout
        try:
            failed_ids = self.future.get(timeout)
        except TimeoutError:
            logger.error(
                f"kvcache with id {self.kvcache_id} wait for save timed out after {timeout} seconds."
            )
            return
        if failed_ids:
            logger.error(
                f"kvcache with id {self.kvcache_id} wait for save error")


class ChariotKvcacheStore:

    def __init__(self,
                 device_id: int,
                 connection_timeout: int = CHARIOT_CONNECTION_TIMEOUT):
        self.chariot_ds_worker_address = os.getenv("DS_WORKER_ADDR", None)
        assert self.chariot_ds_worker_address is not None, "Chariot-DS worker address got None, make sure environment $DS_WORKER_ADDR is set"
        logger.info(
            f"Chariot-DS worker address: {self.chariot_ds_worker_address}")
        self.ip, self.port = self.chariot_ds_worker_address.split(":")
        self.device_id = device_id
        self.connection_timeout = connection_timeout
        self.client = ChariotClient(host=self.ip,
                                    port=int(self.port),
                                    device_id=self.device_id)
        self.client.init()

    def load(self, kvcache_id: str, kv_tensor: torch.Tensor):
        return self.client.mget_h2d([kvcache_id], [kv_tensor],
                                    CHARIOT_KVCACHE_TIMEOUT)

    def save(self, kvcache_id: str, kv_tensor: torch.Tensor):
        return self.client.mset_d2h([kvcache_id], [kv_tensor])

    def async_load(self, kvcache_id: str, kv_tensor: torch.Tensor):
        return self.client.async_mget_h2d([kvcache_id], [kv_tensor],
                                          CHARIOT_KVCACHE_TIMEOUT)

    def async_save(self, kvcache_id: str, kv_tensor: torch.Tensor):
        return self.client.async_mset_d2h([kvcache_id], [kv_tensor])


class ChariotConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        super().__init__(vllm_config, role)
        self.block_size = vllm_config.cache_config.block_size
        self.device = self.tp_rank = None
        self.kv_store: Optional[ChariotKvcacheStore] = None
        self.is_producer = vllm_config.kv_transfer_config.is_kv_producer  # type: ignore

        if role == KVConnectorRole.WORKER:
            self.device = get_world_group().local_rank
            self.tp_rank = get_tp_group().rank
            logger.info(
                f"ChariotConnector initialized with device={self.device} and is producer={self.is_producer}"
            )
            self.kv_store = ChariotKvcacheStore(device_id=self.device)

        self.saving_futures: Dict[str, Dict[str, ChariotFutureWrapper]] = {}
        self.loading_futures: Dict[str, Dict[str, ChariotFutureWrapper]] = {}
        self.requests_waiting_for_load: Dict[str, Request] = {}

    @staticmethod
    def inject_kv_into_layer(dst_kv_layer: torch.Tensor,
                             src_kv_layer: torch.Tensor,
                             slot_mapping: torch.Tensor, is_mla: bool):
        kv_layer_shape = dst_kv_layer.shape
        # TODO: incurs implicit copy, zero-copy optimization later
        if is_mla:
            num_blocks, block_size = kv_layer_shape[0], kv_layer_shape[1]
            dst_kv_layer = dst_kv_layer.reshape(num_blocks * block_size, -1)
            dst_kv_layer[slot_mapping, ...] = src_kv_layer
            dst_kv_layer = dst_kv_layer.reshape(kv_layer_shape)
        else:
            num_blocks, block_size = kv_layer_shape[1], kv_layer_shape[2]
            dst_kv_layer = dst_kv_layer.reshape(2, num_blocks * block_size, -1)
            dst_kv_layer[:, slot_mapping, ...] = src_kv_layer
            dst_kv_layer = dst_kv_layer.reshape(kv_layer_shape)

    @staticmethod
    def extract_kv_from_layer(kv_layer: torch.Tensor,
                              slot_mapping: torch.Tensor,
                              is_mla: bool) -> torch.Tensor:
        if is_mla:
            num_blocks, block_size = kv_layer.shape[0], kv_layer.shape[1]
            return kv_layer.reshape(num_blocks * block_size, -1)[slot_mapping,
                                                                 ...]
        else:
            num_blocks, block_size = kv_layer.shape[1], kv_layer.shape[2]
            return kv_layer.reshape(2, num_blocks * block_size,
                                    -1)[:, slot_mapping, ...]

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        if self.is_producer:
            logger.debug("kv producer calls start_load_kv, skip")
            return
        kvconnector_metadata: ChariotConnectorMetadata = self._get_connector_metadata(
        )  # type: ignore
        if kvconnector_metadata is None:
            logger.warning(
                "kvconnector calls start_load_kv, but the kvconnector metadata is None, skip"
            )
            return
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning(
                "kvconnector calls start_load_kv, but the attention metadata is None, skip"
            )
            return

        is_mla_attn = isinstance(attn_metadata, AscendMLAMetadata)

        for layer_name in forward_context.no_compile_layers:
            for req_meta in kvconnector_metadata.loading_requests:
                seq_len = len(req_meta.slot_mapping)
                if seq_len == 0:
                    return
                kv_layer: torch.Tensor = forward_context.no_compile_layers[
                    layer_name].kv_cache[forward_context.virtual_engine]
                kv_dtype = next(iter(kv_layer.values())).dtype if isinstance(
                    kv_layer, dict) else kv_layer[0].dtype

                req_meta.set_mla(is_mla_attn)
                kvcache_id = req_meta.generate_kvcache_id(
                    layer_name, self.tp_rank)  # type: ignore
                if is_mla_attn:
                    kv_dim = math.prod(kv_layer.shape[2:])
                    kv_holder = torch.zeros((seq_len, kv_dim),
                                            dtype=kv_dtype,
                                            device=self.device)
                else:
                    kv_dim = math.prod(kv_layer.shape[3:])
                    kv_holder = torch.zeros((2, seq_len, kv_dim),
                                            dtype=kv_dtype,
                                            device=self.device)

                if ASYNC_SAVE_LOAD:
                    future = self.kv_store.async_load(  # type: ignore
                        kvcache_id, kv_holder)
                else:
                    future = self.kv_store.load(  # type: ignore
                        kvcache_id, kv_holder)
                chariot_future_wrapper = ChariotFutureWrapper(
                    future, kvcache_id, kv_layer, kv_holder,
                    req_meta.slot_mapping, is_mla_attn)
                if req_meta.request_id not in self.loading_futures:
                    self.loading_futures[req_meta.request_id] = {}
                self.loading_futures[
                    req_meta.request_id][kvcache_id] = chariot_future_wrapper

    def wait_for_layer_load(self, layer_name: str) -> None:
        kvconnector_metadata: ChariotConnectorMetadata = self._get_connector_metadata(
        )  # type: ignore
        if kvconnector_metadata is None:
            logger.debug(
                "kvconnector calls wait_for_layer_load, but the kvconnector metadata is None, skip"
            )
            return
        for req_meta in kvconnector_metadata.loading_requests:
            if req_meta.request_id not in self.loading_futures:
                logger.warning(
                    f"request with id {req_meta.request_id} wait_for_layer_load failed to find future, skip"
                )
                continue
            kvcache_id = req_meta.generate_kvcache_id(
                layer_name, self.tp_rank)  # type: ignore
            chariot_future_wrapper = self.loading_futures[
                req_meta.request_id].pop(kvcache_id, None)
            if chariot_future_wrapper:
                chariot_future_wrapper.wait_for_load()
            else:
                logger.warning(
                    f"kvcache with id {kvcache_id} wait_for_layer_load failed to find future, skip"
                )
            if not self.loading_futures[req_meta.request_id]:
                self.loading_futures.pop(req_meta.request_id, None)
                logger.debug(
                    f"request with id {req_meta.request_id} finished all layer loading"
                )

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        if not self.is_producer:
            logger.debug("kv consumer calls save_kv_layer, skip")
            return

        if attn_metadata is None:
            logger.warning(
                "kvconnector calls save_kv_layer, but the attention metadata is None, skip"
            )
            return

        is_mla_attn = isinstance(attn_metadata, AscendMLAMetadata)
        if is_mla_attn and self.tp_rank > 0:  # type: ignore
            logger.debug(
                f"TP rank = {self.tp_rank} > 0 skips kv save for MLA models.")
            return

        kvconnector_metadata: ChariotConnectorMetadata = self._get_connector_metadata(
        )  # type: ignore
        if kvconnector_metadata is None:
            logger.warning(
                "kvconnector calls save_kv_layer, but the kvconnector metadata is None, skip"
            )
            return

        for req_meta in kvconnector_metadata.saving_requests:
            if len(req_meta.slot_mapping) == 0:
                return
            kv_cache = ChariotConnector.extract_kv_from_layer(
                kv_layer, req_meta.slot_mapping, is_mla_attn)

            req_meta.set_mla(is_mla_attn)
            kvcache_id = req_meta.generate_kvcache_id(
                layer_name, self.tp_rank)  # type: ignore

            if ASYNC_SAVE_LOAD:
                future = self.kv_store.async_save(  # type: ignore
                    kvcache_id, kv_cache)
            else:
                future = self.kv_store.save(  # type: ignore
                    kvcache_id, kv_cache)
            chariot_future_wrapper = ChariotFutureWrapper(
                future, kvcache_id, kv_layer, kv_cache, req_meta.slot_mapping,
                is_mla_attn)
            if req_meta.request_id not in self.saving_futures:
                self.saving_futures[req_meta.request_id] = {}
            self.saving_futures[
                req_meta.request_id][kvcache_id] = chariot_future_wrapper

    def wait_for_save(self) -> None:
        kvconnector_metadata: ChariotConnectorMetadata = self._get_connector_metadata(
        )  # type: ignore
        if kvconnector_metadata is None:
            logger.warning(
                "kvconnector calls wait_for_layer_load, but the kvconnector metadata is None, skip"
            )
            return
        for req_meta in kvconnector_metadata.saving_requests:
            if req_meta.request_id not in self.saving_futures:
                logger.warning(
                    f"request with id {req_meta.request_id} wait_for_save failed to find future, skip"
                )
                continue
            has_futures = False
            for kvcache_id, chariot_future_wrapper in self.saving_futures[
                    req_meta.request_id].items():
                chariot_future_wrapper.wait_for_save()
                has_futures = True
            if not has_futures:
                logger.warning(
                    f"kvcache with id {kvcache_id} wait_for_save failed to find future, skip"
                )
            self.saving_futures.pop(req_meta.request_id, None)
        for request_id in self.saving_futures.keys():
            logger.warning(
                f"request with id {request_id} has a saving future left, but not in kvconnector metadata, ignore"
            )
        self.saving_futures.clear()

    def get_num_new_matched_tokens(
            self, request, num_computed_tokens: int) -> tuple[int, bool]:
        if self.is_producer:
            return 0, False
        num_total_tokens = len(request.prompt_token_ids) - 1
        return num_total_tokens - num_computed_tokens, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if not self.is_producer:
            logger.debug(
                f"request with id {request.request_id} updated state to waiting for kvcache load"
            )
            self.requests_waiting_for_load[request.request_id] = request

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        kvconnector_metadata = ChariotConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self.requests_waiting_for_load:
                kvconnector_metadata.add_load_request(
                    request_id=new_req.req_id,
                    token_ids=new_req.prompt_token_ids,
                    block_ids=new_req.
                    block_ids[0],  # only one kvcache group now
                    block_size=self.block_size)
            elif self.is_producer:
                kvconnector_metadata.add_save_request(
                    request_id=new_req.req_id,
                    token_ids=new_req.prompt_token_ids,
                    block_ids=new_req.
                    block_ids[0],  # only one kvcache group now
                    block_size=self.block_size)
        for cached_req in scheduler_output.scheduled_cached_reqs:
            if not cached_req.resumed_from_preemption:
                break
            if cached_req.req_id in self.requests_waiting_for_load:
                total_tokens = len(
                    cached_req.new_token_ids) + cached_req.num_computed_tokens
                if total_tokens > self.requests_waiting_for_load[
                        cached_req.req_id].num_prompt_tokens:
                    logger.debug(
                        f"just resumed from preempted request {cached_req.req_id} has already loaded kvcache, skip"
                    )
                    continue
                token_ids = self.requests_waiting_for_load[
                    cached_req.req_id].all_token_ids[:total_tokens]
                block_ids = cached_req.new_block_ids
                kvconnector_metadata.add_load_request(
                    request_id=cached_req.req_id,
                    token_ids=token_ids,
                    block_ids=block_ids[0],  # only one kvcache group now
                    block_size=self.block_size)
        self.requests_waiting_for_load.clear()
        return kvconnector_metadata
