# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch
from vllm.attention import AttentionType
from vllm.attention.layer import Attention
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.utils import logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_ascend.distributed.cpu_offload_manager.metadata import (
    MetadataServer, MetadataServerProc, MLAConfig)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request


@dataclass
class ReqMeta:
    gpu_block_ids: list[int]
    cpu_block_ids: list[int]
    num_scheduled_tokens: int
    num_computed_tokens: int
    num_gpu_computed_tokens: int
    num_cpu_computed_tokens: int

    def update(self, other: "ReqMeta"):
        self.gpu_block_ids.extend(other.gpu_block_ids)
        self.cpu_block_ids.extend(other.cpu_block_ids)
        self.num_scheduled_tokens = other.num_scheduled_tokens
        self.num_computed_tokens = other.num_computed_tokens
        self.num_gpu_computed_tokens = other.num_gpu_computed_tokens
        self.num_cpu_computed_tokens = other.num_cpu_computed_tokens


@dataclass
class CPUOffloadingConnectorMetadata(KVConnectorMetadata):
    requests: dict[str, ReqMeta]
    finished_req_ids: set[str]


class CPUOffloadingConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        if not vllm_config.cache_config.enable_prefix_caching:
            self.connector_scheduler: Optional[
                CPUOffloadingConnectorScheduler] = None
            self.connector_worker: Optional[
                CPUOffloadingConnectorWorker] = None
        elif role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = CPUOffloadingConnectorScheduler(
                vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = CPUOffloadingConnectorWorker(vllm_config)

    # ==============================
    # Worker-side methods
    # ==============================

    def bind_connector_metadata(
            self, connector_metadata: KVConnectorMetadata) -> None:
        if self.connector_worker is not None:
            assert isinstance(connector_metadata,
                              CPUOffloadingConnectorMetadata)
            self.connector_worker.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        assert self.connector_worker is not None
        self.connector_worker.clear_connector_metadata()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if self.connector_worker is not None:
            self.connector_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        if self.connector_worker is not None:
            self.connector_worker.start_load_kv()

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self.connector_worker is not None:
            self.connector_worker.wait_for_layer_load()

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        pass

    def wait_for_save(self):
        pass

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(), None

    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        if self.connector_scheduler is not None:
            return self.connector_scheduler.get_num_new_matched_tokens(
                request, num_computed_tokens)
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if self.connector_scheduler is not None:
            return self.connector_scheduler.update_state_after_alloc(request)

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        if self.connector_scheduler is not None:
            return self.connector_scheduler.build_connector_meta(
                scheduler_output)
        return KVConnectorMetadata()

    def request_finished(
            self, request: "Request",
            block_ids: list[int]) -> tuple[bool, Optional[dict[str, Any]]]:
        if self.connector_scheduler is not None:
            self.connector_scheduler.request_finished(request)
        return True, None


class CPUOffloadingConnectorScheduler:

    def __init__(self, vllm_config: VllmConfig):
        logger.info("init CPUOffloadingConnectorScheduler")
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.use_mla = vllm_config.model_config.use_mla
        self.num_gpu_computed_tokens: dict[str, int] = {}
        self.num_cpu_computed_tokens: dict[str, int] = {}
        self.allocated_req_ids: set[str] = set()
        self.finished_req_ids: list[str] = []
        self.zmq_rpc_client = MetadataServer.ZMQRPCClient()
        self.zmq_rpc_client.call("post_init")
        if vllm_config.kv_transfer_config is not None:
            self.swap_in_threshold = vllm_config.kv_transfer_config.get_from_extra_config(
                "swap_in_threshold", 0)
        else:
            self.swap_in_threshold = 0
        logger.info(f"swap_in_threshold: {self.swap_in_threshold}")

    def get_num_new_matched_tokens(
            self, ori_request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        request = copy.deepcopy(ori_request)
        request.get_hash_new_full_blocks = None
        num_cpu_computed_tokens, load_async = self.zmq_rpc_client.call(
            "get_matched_num_and_touch", request)
        self.num_gpu_computed_tokens[request.request_id] = num_computed_tokens
        self.num_cpu_computed_tokens[
            request.request_id] = num_cpu_computed_tokens
        if num_cpu_computed_tokens - num_computed_tokens >= self.swap_in_threshold:
            return num_cpu_computed_tokens - num_computed_tokens, load_async
        else:
            return 0, load_async

    def update_state_after_alloc(self, request: "Request"):
        self.allocated_req_ids.add(request.request_id)

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        num_tokens = {}
        # process scheduled_new_reqs
        for req in scheduler_output.scheduled_new_reqs:
            req_id = req.req_id
            num_tokens[req_id] = (
                req.num_computed_tokens +
                scheduler_output.num_scheduled_tokens[req_id])

        # process scheduled_cached_reqs
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(cached_reqs.req_ids):
            num_tokens[req_id] = (
                cached_reqs.num_computed_tokens[idx] +
                scheduler_output.num_scheduled_tokens[req_id])

        unallocated_req_ids = set(self.num_gpu_computed_tokens.keys() -
                                  self.allocated_req_ids -
                                  scheduler_output.num_scheduled_tokens.keys())
        new_cpu_block_ids = self.zmq_rpc_client.call("allocate_slots",
                                                     num_tokens,
                                                     unallocated_req_ids)
        metadata = CPUOffloadingConnectorMetadata(
            requests={},
            finished_req_ids=set(self.finished_req_ids),
        )
        for req in scheduler_output.scheduled_new_reqs:
            req_id = req.req_id
            gpu_block_ids = req.block_ids[0]
            metadata.requests[req_id] = ReqMeta(
                gpu_block_ids=[] if gpu_block_ids is None else gpu_block_ids,
                cpu_block_ids=new_cpu_block_ids.get(req_id, []),
                num_scheduled_tokens=scheduler_output.
                num_scheduled_tokens[req_id],
                num_computed_tokens=req.num_computed_tokens,
                num_gpu_computed_tokens=self.num_gpu_computed_tokens[req_id],
                num_cpu_computed_tokens=self.num_cpu_computed_tokens[req_id])

        for idx, req_id in enumerate(cached_reqs.req_ids):
            gpu_block_ids = cached_reqs.new_block_ids[idx]
            metadata.requests[req_id] = ReqMeta(
                gpu_block_ids=[] if gpu_block_ids is None else gpu_block_ids,
                cpu_block_ids=new_cpu_block_ids.get(req_id, []),
                num_scheduled_tokens=scheduler_output.
                num_scheduled_tokens[req_id],
                num_computed_tokens=cached_reqs.num_computed_tokens[idx],
                num_gpu_computed_tokens=cached_reqs.num_computed_tokens[idx],
                num_cpu_computed_tokens=cached_reqs.num_computed_tokens[idx])
        self.num_gpu_computed_tokens.clear()
        self.num_cpu_computed_tokens.clear()
        self.allocated_req_ids.clear()
        self.finished_req_ids.clear()
        return metadata

    def request_finished(self, ori_request: "Request"):
        request = copy.deepcopy(ori_request)
        request.get_hash_new_full_blocks = None
        self.finished_req_ids.append(request.request_id)
        # inform metadata server to record request, and free it after finish sending
        self.zmq_rpc_client.call("record_request_cache_and_free_slots",
                                 request)


class CPUOffloadingConnectorWorker:

    def __init__(self, vllm_config: VllmConfig):
        logger.info("init CPUOffloadingConnectorWorker")
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.pp_rank = get_pp_group().rank_in_group
        self.tp_group = get_tp_group()
        self.tp_rank = self.tp_group.rank_in_group
        self.tp_world_size = self.tp_group.world_size
        self.use_mla = vllm_config.model_config.use_mla

        self.requests: dict[str, ReqMeta] = {}
        self.load_stream = torch.npu.Stream()
        self.save_stream = torch.npu.Stream()
        self.zmq_rpc_client = MetadataServer.ZMQRPCClient()
        self.load_block_mapping: list[tuple[int, int]] = []
        self.save_input_queue: queue.Queue[tuple[str, ReqMeta]] = queue.Queue()
        self.save_output_queue: queue.Queue[str] = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_listener)
        self.save_thread.start()
        self.done_sending_count: defaultdict[str, int] = defaultdict(int)

        # start metadata server to init cpu_kv_cache_manager and handle rpc requests
        # all dp shared the same metadata server, only start the process on data_rank 0
        if vllm_config.parallel_config.data_parallel_rank == 0 and self.tp_rank == 0 and self.pp_rank == 0:
            config = VllmConfig()
            config.cache_config = vllm_config.cache_config
            config.parallel_config = vllm_config.parallel_config
            config.kv_transfer_config = vllm_config.kv_transfer_config
            self.init_metadata_server(config)
        self._wait_for_metadata_process_start()

    def init_metadata_server(self, vllm_config: VllmConfig):
        self.metadata_thread = threading.Thread(
            target=MetadataServerProc.run_metadata_server,
            args=(vllm_config, ),
        )
        self.metadata_thread.daemon = True
        self.metadata_thread.start()

    def _wait_for_metadata_process_start(self):
        # TODO: wait for metadata server to start, add a rpc to check if ready
        while True:
            try:
                if self.zmq_rpc_client.call("ready"):
                    break
            except Exception as e:
                logger.info(f"wait for metadata server to start, error: {e}")
                time.sleep(1)

    def bind_connector_metadata(
            self, connector_metadata: CPUOffloadingConnectorMetadata) -> None:
        for req_id, req in connector_metadata.requests.items():
            if req_id in self.requests:
                self.requests[req_id].update(req)
                req = self.requests[req_id]
            else:
                self.requests[req_id] = req
            for i in range(req.num_gpu_computed_tokens // self.block_size,
                           req.num_computed_tokens // self.block_size):
                self.load_block_mapping.append(
                    (req.cpu_block_ids[i], req.gpu_block_ids[i]))
        for req_id in connector_metadata.finished_req_ids:
            if req_id in self.requests:
                self.save_input_queue.put((req_id, self.requests[req_id]))

    def clear_connector_metadata(self) -> None:
        self.load_block_mapping.clear()

    def register_kv_caches(self, kv_caches: dict[str, Sequence[torch.Tensor]]):
        self.gpu_kv_caches = kv_caches
        model_config = self.vllm_config.model_config
        mla_config: Optional[MLAConfig] = None
        if model_config.use_mla:
            mla_config = MLAConfig(
                model_config.hf_text_config.kv_lora_rank,
                model_config.hf_text_config.qk_rope_head_dim)
        self.cpu_kv_caches = list(
            self.zmq_rpc_client.call(
                "init_cpu_kv_caches",
                self.pp_rank,
                self.tp_rank,
                get_kv_cache_spec(self.vllm_config),
                mla_config,
            ).values())

    def start_load_kv(self) -> None:
        self.current_layer = 0
        self.gpu_kv_caches_load_iter = iter(self.gpu_kv_caches.values())
        self.load_kv_layer(0)

    def wait_for_layer_load(self) -> None:
        # TODO: Replace with `torch.npu.current_stream().wait_stream(self.load_stream)` after fixing the bug.
        self.load_stream.synchronize()
        self.current_layer += 1
        self.load_kv_layer(self.current_layer)

    def load_kv_layer(self, layer: int):
        if layer == len(self.gpu_kv_caches):
            return
        gpu_kv_caches = next(self.gpu_kv_caches_load_iter)
        cpu_kv_caches = self.cpu_kv_caches[layer]
        with torch.npu.stream(self.load_stream):
            for cpu_block_id, gpu_block_id in self.load_block_mapping:
                for gpu_layer_part, cpu_layer_part in zip(
                        gpu_kv_caches, cpu_kv_caches):
                    gpu_layer_part[gpu_block_id].copy_(
                        cpu_layer_part[cpu_block_id], non_blocking=True)

    def get_finished(self) -> set[str]:
        done_sending: set[str] = set()
        while True:
            try:
                id = self.save_output_queue.get_nowait()
            except queue.Empty:
                break
            done_sending.add(id)
        for id in done_sending:
            del self.requests[id]
        if self.tp_world_size == 1:
            return done_sending
        if self.tp_rank == 0:
            for req_id in done_sending:
                self.done_sending_count[req_id] += 1
            other_ranks_finished_ids: list[str] = []
            for i in range(1, self.tp_world_size):
                other_ranks_finished_ids.extend(
                    self.tp_group.recv_object(src=i))
            for req_id in other_ranks_finished_ids:
                self.done_sending_count[req_id] += 1
            all_done_sending: set[str] = set()
            for req_id in list(self.done_sending_count.keys()):
                if self.done_sending_count[req_id] == self.tp_world_size:
                    del self.done_sending_count[req_id]
                    all_done_sending.add(req_id)
            # release cpu_kv_cache after request sending finished
            # to avoid rpc blocking, use thread to call rpc asynchronously
            sending_finished_thread = threading.Thread(
                target=self._sending_finished, args=(all_done_sending, ))
            sending_finished_thread.daemon = True
            sending_finished_thread.start()

            return all_done_sending
        else:
            self.tp_group.send_object(done_sending, dst=0)
            return done_sending

    def _sending_finished(self, all_done_sending):
        for req_id in all_done_sending:
            logger.debug(f"call cache_and_free_slots for req_id: {req_id}")
            self.zmq_rpc_client.call("cache_and_free_slots", req_id)

    def _save_listener(self):
        save_block_mapping = []
        while True:
            req_id, req = self.save_input_queue.get()
            for i in range(
                    req.num_cpu_computed_tokens // self.block_size,
                    min((req.num_computed_tokens + req.num_scheduled_tokens) //
                        self.block_size, len(req.cpu_block_ids))):
                save_block_mapping.append(
                    (req.gpu_block_ids[i], req.cpu_block_ids[i]))
            with torch.npu.stream(self.save_stream):
                # MLA: kv_layer is tuple[tensor, tensor] means (rope, nope).
                # non-MLA: kv_layer is list[tensor], typically means [k, v].
                if self.use_mla:
                    start, step = self.tp_rank, self.tp_world_size
                else:
                    start, step = 0, 1
                for i in range(start, len(save_block_mapping), step):
                    gpu_block_id, cpu_block_id = save_block_mapping[i]
                    for cpu_kv_caches, gpu_kv_caches in zip(
                            self.cpu_kv_caches, self.gpu_kv_caches.values()):
                        for cpu_layer_part, gpu_layer_part in zip(
                                cpu_kv_caches, gpu_kv_caches):
                            cpu_layer_part[cpu_block_id].copy_(
                                gpu_layer_part[gpu_block_id],
                                non_blocking=True)
            self.save_stream.synchronize()
            self.save_output_queue.put(req_id)
            save_block_mapping.clear()


# Copied from vllm_ascend/worker/model_runner_v1.py.
def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    forward_ctx = vllm_config.compilation_config.static_forward_context
    block_size = vllm_config.cache_config.block_size
    use_mla = vllm_config.model_config.use_mla
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    for layer_name, attn_module in forward_ctx.items():
        if isinstance(attn_module, FusedMoE):
            continue
        assert isinstance(attn_module, Attention)
        if attn_module.attn_type == AttentionType.DECODER:
            kv_cache_spec[layer_name] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=attn_module.num_kv_heads,
                head_size=attn_module.head_size,
                dtype=attn_module.dtype,
                use_mla=use_mla)
        elif attn_module.attn_type in (AttentionType.ENCODER,
                                       AttentionType.ENCODER_ONLY):
            continue
        elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
            raise NotImplementedError
        else:
            raise ValueError(
                f"Unknown attention type: {attn_module.attn_type}")
    return kv_cache_spec
