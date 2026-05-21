import importlib
import math
import threading

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.cpu_binding import (
    bind_thread_to_cpus,
    get_kv_transfer_thread_cpus,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    backend_map,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerBlockRange,
    LayerLoadTask,
    LayerTransferTask,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
    _circular_shift,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_config import (
    get_layerwise_config,
)
from vllm_ascend.memcache_comm_fence import (
    get_attention_compute_start_gate,
    reset_attention_compute_start_gate,
)


class KVPoolWorker:
    # The main class for the cache engine.

    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwise: bool,
    ):
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.local_rank = envs.LOCAL_RANK
        self.use_mla = False
        if hasattr(model_config, "use_mla") and isinstance(model_config.use_mla, bool) and model_config.use_mla:
            self.use_mla = True
        self.use_sparse = hasattr(model_config.hf_text_config, "index_topk")
        self.use_layerwise = use_layerwise
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = (parallel_config.rank // self.tp_size) % self.pp_size

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0

        self.kv_role = vllm_config.kv_transfer_config.kv_role
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.load_async = extra_config.get("load_async", False)
        self.consumer_is_to_put = extra_config.get(
            "consumer_is_to_put", False
        )
        self.backend = extra_config.get("backend", "mooncake")
        self.h2d_stagger_us = int(extra_config.get("h2d_stagger_us", 0))
        self.h2d_stagger_group_size = int(
            extra_config.get("h2d_stagger_group_size", 0))
        self.h2d_stagger_dynamic_addrs_per_us = int(
            extra_config.get("h2d_stagger_dynamic_addrs_per_us", 0))
        self.h2d_stagger_max_us = int(
            extra_config.get("h2d_stagger_max_us", 0))
        self.layerwise_max_transfer_blocks = int(
            extra_config.get("layerwise_max_transfer_blocks", 0))
        self.layerwise_max_transfer_bytes = int(
            extra_config.get("layerwise_max_transfer_bytes", 0))
        self.block_size = vllm_config.cache_config.block_size

        if self.pcp_size > 1:
            self.block_size *= self.pcp_size
        if self.dcp_size > 1:
            self.block_size *= self.dcp_size
        self.current_layer = 0
        self.num_layers = model_config.get_num_layers(parallel_config)

        if self.use_mla:
            self.num_kv_head = 1
        else:
            self.num_kv_head = model_config.get_total_num_kv_heads()

        if self.num_kv_head < self.tp_size:
            self.put_step = self.tp_size // self.num_kv_head
            self.head_or_tp_rank = self.tp_rank // self.put_step
        else:
            self.head_or_tp_rank = self.tp_rank
            self.put_step = 1
        self.my_key_index = (self.pcp_rank * self.dcp_size * (self.tp_size // self.put_step) +
                             self.dcp_rank * (self.tp_size // self.put_step) +
                             self.head_or_tp_rank)
        self.num_ranks_per_layer = self.pcp_size * self.dcp_size * (self.tp_size // self.put_step)

        self.metadata = KeyMetadata(
            model_config.model.rstrip("/").split("/")[-1],
            self.head_or_tp_rank,
            self.pcp_rank,
            self.dcp_rank,
            self.pp_rank,
        )

        partitions = None
        if self.kv_role == "kv_consumer" and self.consumer_is_to_put:
            num_hidden_layers = model_config.hf_text_config.num_hidden_layers
            partition_list_str = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
                "prefill_pp_layer_partition", None
            )
            prefill_pp_size = int(vllm_config.kv_transfer_config.kv_connector_extra_config.get("prefill_pp_size", 1))

            if partition_list_str is not None:
                try:
                    partitions = [int(layer) for layer in partition_list_str.split(",")]
                except ValueError as err:
                    raise ValueError("Invalid partition string: {}".format(partition_list_str)) from err
                if len(partitions) != prefill_pp_size:
                    raise ValueError(f"{len(partitions)=} does not match {prefill_pp_size=}.")
                if sum(partitions) != num_hidden_layers:
                    raise ValueError(f"{sum(partitions)=} does not match {num_hidden_layers=}.")
            else:
                layers_per_partition = num_hidden_layers // prefill_pp_size
                partitions = [layers_per_partition for _ in range(prefill_pp_size)]

                if remaining_layers := num_hidden_layers % prefill_pp_size:
                    for i in range(2, remaining_layers + 2):
                        partitions[-i] += 1

        self.token_database = ChunkedTokenDatabase(self.metadata, self.block_size, partitions)

        backend = backend_map.get(self.backend.lower())
        assert backend is not None
        backend_path = backend.get("path")
        backend_name = backend.get("name")
        assert backend_path is not None and backend_name is not None
        backend_module = importlib.import_module(backend_path)
        real_backend = getattr(backend_module, backend_name)

        self.m_store = real_backend(  # type: ignore[misc]
            parallel_config
        )
        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = False
        if kv_event_config and kv_event_config.enable_kv_cache_events:
            self.enable_kv_events = True

        self.kv_send_thread: KVTransferThread | None = None
        self.kv_recv_thread: KVTransferThread | None = None

        self.layer_load_tasks: list[list[LayerTransferTask]] = [[] for i in range(self.num_layers)]
        self.layer_save_tasks: list[list[LayerTransferTask]] = [[] for i in range(self.num_layers)]
        self.layer_load_finished_events = None
        self.layer_save_finished_events = None
        self.layer_transfer_finished_events = None
        self.kv_transfer_thread_cpus: list[int] = []
        if get_ascend_config().enable_cpu_binding:
            try:
                self.kv_transfer_thread_cpus = get_kv_transfer_thread_cpus(
                    self.local_rank,
                    2,
                )
            except Exception as err:
                logger.warning(
                    "Failed to get KV transfer thread CPUs for rank%d: %s",
                    self.local_rank,
                    err,
                )
        self.next_layer_to_submit = 0
        layerwise_config = get_layerwise_config(self.num_layers)
        self.layerwise_offload = layerwise_config.has_layer_reuse
        self.NUM_PREFETCH_LAYERS = layerwise_config.num_prefetch_layers
        self.independent_layers = layerwise_config.independent_layers
        self.prefetch_layer_map = layerwise_config.prefetch_layer_map
        self.sync_save_events = None

    def _bind_kv_transfer_thread(
        self,
        thread: KVTransferThread | None,
        cpu_index: int,
        name: str,
    ) -> None:
        if thread is None or thread.native_id is None:
            return
        if cpu_index >= len(self.kv_transfer_thread_cpus):
            return
        cpu = self.kv_transfer_thread_cpus[cpu_index]
        try:
            bind_thread_to_cpus(thread.native_id, [cpu])
            logger.info("Bound %s thread %d to CPU%d", name, thread.native_id, cpu)
        except Exception as err:
            logger.warning(
                "Failed to bind %s thread %d to CPU%d: %s",
                name,
                thread.native_id,
                cpu,
                err,
            )

    def rebind_kv_transfer_threads(self) -> None:
        if not self.use_layerwise:
            return
        self._bind_kv_transfer_thread(
            self.kv_send_thread,
            0,
            "layerwise send",
        )
        self._bind_kv_transfer_thread(
            self.kv_recv_thread,
            1,
            "layerwise recv",
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache = first_kv_cache_tuple[0]

        self.num_blocks = first_kv_cache.shape[0]
        logger.info("num_blocks: %s", self.num_blocks)
        block_rank = 3
        self.block_len = []
        for i in range(len(first_kv_cache_tuple)):
            block_shape = first_kv_cache_tuple[i].shape[-block_rank:]
            logger.info("block_shape: %s", block_shape)
            self.block_len.append(first_kv_cache_tuple[i].element_size() * math.prod(block_shape))

        logger.info(
            "Registering KV_Caches. use_mla: %s, use_sparse: %s, shape %s",
            self.use_mla,
            self.use_sparse,
            first_kv_cache.shape,
        )

        self.kv_caches_base_addr = []
        ptrs = []
        lengths = []
        length = len(self.block_len)
        for cache_or_caches in kv_caches.values():
            for i, cache in enumerate(cache_or_caches, 0):
                base_addr = cache.data_ptr()
                if base_addr not in self.kv_caches_base_addr:
                    region_len = self.num_blocks * self.block_len[i % length]
                    ptrs.append(base_addr)
                    lengths.append(region_len)
                self.kv_caches_base_addr.append(base_addr)

        self.m_store.register_buffer(ptrs, lengths)
        self.page_size_bytes = sum(self.block_len)
        self.token_database.set_kv_caches_base_addr(self.kv_caches_base_addr)
        self.token_database.set_block_len(self.block_len)

        if self.use_layerwise:
            self.get_event = threading.Event()
            self.layer_load_finished_events = [threading.Event() for i in range(self.num_layers)]
            self.layer_save_finished_events = [threading.Event() for i in range(self.num_layers)]
            self.sync_save_events = [torch.npu.Event() for i in range(self.num_layers)]
            if self.kv_role in ["kv_producer", "kv_both"]:
                ready_event_sending = threading.Event()
                self.kv_send_thread = KVCacheStoreLayerSendingThread(
                    self.m_store,
                    self.token_database,
                    self.block_size,
                    self.tp_rank,
                    self.tp_size,
                    self.dcp_size,
                    self.put_step,
                    self.my_key_index,
                    self.num_ranks_per_layer,
                    self.page_size_bytes,
                    ready_event_sending,
                    self.num_layers,
                    self.layer_save_finished_events,
                    self.sync_save_events,
                    self.enable_kv_events,
                    self.layer_transfer_finished_events,
                    self.layerwise_max_transfer_blocks,
                    self.layerwise_max_transfer_bytes,
                )
                self.kv_send_thread.start()
                self._bind_kv_transfer_thread(
                    self.kv_send_thread,
                    0,
                    "layerwise send",
                )
            ready_event = threading.Event()
            self.kv_recv_thread = KVCacheStoreLayerRecvingThread(
                self.m_store,
                self.token_database,
                self.block_size,
                self.tp_rank,
                self.tp_size,
                self.dcp_size,
                self.my_key_index,
                self.num_ranks_per_layer,
                self.page_size_bytes,
                ready_event,
                self.get_event,
                self.layer_load_finished_events,
                self.layer_save_finished_events,
                self.num_layers,
                self.h2d_stagger_us,
                self.h2d_stagger_group_size,
                self.h2d_stagger_dynamic_addrs_per_us,
                self.h2d_stagger_max_us,
                self.layerwise_max_transfer_blocks,
                self.layerwise_max_transfer_bytes,
            )
            self.kv_recv_thread.start()
            self._bind_kv_transfer_thread(
                self.kv_recv_thread,
                1,
                "layerwise recv",
            )
            ready_event.wait()
        else:
            if self.kv_role in ["kv_producer", "kv_both"] or self.consumer_is_to_put:
                ready_event_sending = threading.Event()
                self.kv_send_thread = KVCacheStoreSendingThread(
                    self.m_store,
                    self.token_database,
                    self.block_size,
                    self.tp_rank,
                    self.tp_size,
                    self.dcp_size,
                    self.put_step,
                    self.kv_role,
                    ready_event_sending,
                    self.enable_kv_events,
                )
                self.kv_send_thread.start()
            if self.load_async:
                ready_event = threading.Event()
                self.kv_recv_thread = KVCacheStoreRecvingThread(
                    self.m_store,
                    self.token_database,
                    self.block_size,
                    self.tp_rank,
                    self.tp_size,
                    self.dcp_size,
                    ready_event,
                )
                self.kv_recv_thread.start()
                ready_event.wait()

    def start_load_kv(self, metadata: AscendConnectorMetadata):
        self.current_layer = 0
        if self.use_layerwise:
            self.next_layer_to_submit = 0
            reset_attention_compute_start_gate()
        if len(metadata.requests) == 0:
            return
        if self.use_layerwise:
            self.process_layer_data(metadata.requests)
            return

        for request in metadata.requests:
            load_spec = request.load_spec
            if load_spec is None or not load_spec.can_load:  # load =0
                continue
            token_len = request.save_end_token
            if (load_spec.kvpool_cached_tokens % self.block_size != 0) and (
                load_spec.kvpool_cached_tokens == token_len - 1
            ):
                token_len = request.load_spec.kvpool_cached_tokens + 1
            else:
                token_len = request.load_spec.kvpool_cached_tokens
            request.load_spec.token_len = token_len
            if self.load_async:
                self.kv_recv_thread.add_request(  # type: ignore[union-attr]
                    request,
                )
            else:
                addr_list = []
                size_list = []
                key_list = []
                mask_num = request.load_spec.vllm_cached_tokens // self.block_size * self.block_size
                for start, end, key in self.token_database.process_tokens(
                    token_len, request.block_hashes, mask_num
                ):
                    addr, size, _ = self.token_database.prepare_value(start, end, request.block_ids)
                    key_list.append(key.to_string())
                    addr_list.append(addr)
                    size_list.append(size)
                key_list_c = _circular_shift(key_list, self.tp_rank % len(key_list))
                addr_list_c = _circular_shift(addr_list, self.tp_rank % len(addr_list))
                size_list_c = _circular_shift(size_list, self.tp_rank % len(size_list))
                self.m_store.get(key_list_c, addr_list_c, size_list_c)

    def _process_save_for_layer_batch(
        self,
        requests: list[ReqMeta],
        layer_id: int,
    ) -> None:
        request_block_ranges = []
        for request in requests:
            if request.can_save is None or not request.can_save:
                continue
            save_start_block = request.save_start_token // self.block_size
            save_end_block = request.save_end_token // self.block_size
            if save_start_block >= save_end_block and request.partial_block_index is None:
                continue
            partial_block_index = request.partial_block_index
            request_block_ranges.append(
                LayerBlockRange(
                    request=request,
                    start_block=save_start_block,
                    end_block=save_end_block,
                    partial_block_index=partial_block_index,
                )
            )
        if request_block_ranges:
            self.layer_save_tasks[layer_id].append(
                LayerTransferTask(
                    layer_id=layer_id,
                    block_ranges=request_block_ranges,
                )
            )

    def _process_load_for_layer_batch(
        self,
        requests: list[ReqMeta],
        layer_id: int,
    ) -> None:
        request_block_ranges = []
        for request in requests:
            if request.load_spec is None or not request.load_spec.can_load:
                continue
            cached_tokens = request.load_spec.kvpool_cached_tokens
            if self.layerwise_offload and layer_id not in self.independent_layers:
                load_start_block = 0
            else:
                load_start_block = request.load_spec.vllm_cached_tokens // self.block_size
            cached_full_blocks = cached_tokens // self.block_size
            full_blocks = min(cached_full_blocks, len(request.block_hashes))
            load_previous_last_block = (
                self.layerwise_offload
                and request.last_block_gva is not None
                and cached_tokens > 0
                and cached_tokens % self.block_size == 0
                and cached_tokens == request.target_token_len - 1
                and request.load_spec.vllm_cached_tokens == cached_tokens
            )
            if load_previous_last_block:
                full_blocks = max(0, cached_full_blocks - 1)
            needs_last_block_at_boundary = (
                cached_tokens > 0
                and cached_tokens % self.block_size == 0
                and full_blocks < cached_full_blocks
            )
            if request.last_block_gva is not None and (
                cached_tokens % self.block_size != 0
                or needs_last_block_at_boundary
            ):
                partial_block_index = (
                    cached_full_blocks
                    if cached_tokens % self.block_size != 0
                    else cached_full_blocks - 1
                )
            else:
                partial_block_index = None
            if (partial_block_index is not None
                    and partial_block_index < load_start_block):
                partial_block_index = None
            if load_start_block >= full_blocks and partial_block_index is None:
                continue
            request_block_ranges.append(
                LayerBlockRange(
                    request=request,
                    start_block=load_start_block,
                    end_block=full_blocks,
                    partial_block_index=partial_block_index,
                )
            )
        if request_block_ranges:
            self.layer_load_tasks[layer_id].append(
                LayerTransferTask(
                    layer_id=layer_id,
                    block_ranges=request_block_ranges,
                )
            )

    def process_layer_data(self, requests: list[ReqMeta]) -> None:
        if not requests:
            return
        for layer_id in range(self.num_layers):
            self._process_save_for_layer_batch(requests, layer_id)
        for layer_id in range(self.num_layers):
            self._process_load_for_layer_batch(requests, layer_id)

    def _submit_ready_layer_loads(self) -> None:
        assert self.kv_recv_thread is not None

        def submit_layer_load(layer_id: int) -> bool:
            if not self.layer_load_tasks[layer_id]:
                return False
            wait_for_save_layer = self.prefetch_layer_map.get(layer_id)
            attention_start_gate = None
            if layer_id != self.current_layer:
                attention_start_gate = get_attention_compute_start_gate()
            self.kv_recv_thread.add_request(
                LayerLoadTask(
                    wait_for_save_layer=wait_for_save_layer,
                    transfer_tasks=self.layer_load_tasks[layer_id],
                    layer_id=layer_id,
                    attention_start_gate=attention_start_gate,
                )
            )
            return True

        submit_count = self.NUM_PREFETCH_LAYERS if self.current_layer == 0 else 1
        submitted_layers = 0
        while (submitted_layers < submit_count
               and self.next_layer_to_submit < self.num_layers):
            layer_id = self.next_layer_to_submit
            self.next_layer_to_submit += 1
            if submit_layer_load(layer_id):
                submitted_layers += 1

    def wait_for_layer_load(self) -> None:
        reset_attention_compute_start_gate()
        self._submit_ready_layer_loads()
        should_wait = bool(self.layer_load_tasks[self.current_layer])
        if not should_wait:
            self.layer_load_finished_events[self.current_layer].clear()
            return
        is_finish = self.layer_load_finished_events[self.current_layer].wait(timeout=10)
        if not is_finish:
            logger.info("Layerwise %d load wait timed out", self.current_layer)
        logger.debug(f">>>>>>>>>>>>>>>>>>>> clear load layer {self.current_layer}")
        self.layer_load_finished_events[self.current_layer].clear()

    def save_kv_layer(self, connector_metadata: AscendConnectorMetadata) -> None:
        # Wait for KV cache saving to complete on the final layer that requires offloading.
        self.sync_save_events[self.current_layer].record()
        if self.layer_save_tasks[self.current_layer]:
            for block_range in self.layer_save_tasks[self.current_layer][0].block_ranges:
                self.kv_send_thread.add_stored_request(
                    block_range.request.req_id)
            self.kv_send_thread.add_request(self.layer_save_tasks[self.current_layer])
        else:
            self.layer_save_finished_events[self.current_layer].set()
        if self.current_layer == self.num_layers - 1:
            is_finish = self.layer_save_finished_events[self.num_layers - 1].wait(timeout=10)
            if not is_finish:
                logger.info("Layerwise %d save wait timed out", self.current_layer)
            for layer_id in range(self.num_layers):
                if self.layer_save_finished_events[layer_id].is_set():
                    logger.debug(f">>>>>>>>>>>>>>>>>>>> clear save layer {layer_id}")
                    self.layer_save_finished_events[layer_id].clear()

        self.current_layer = self.current_layer + 1

    def wait_for_save(self, connector_metadata: AscendConnectorMetadata):
        current_event = None
        for request in connector_metadata.requests:
            can_save = request.can_save
            if can_save is None or not can_save:
                continue
            current_event = torch.npu.Event()
            current_event.record()
            break

        for request in connector_metadata.requests:
            can_save = request.can_save
            if can_save is None or not can_save:
                continue

            request.current_event = current_event
            self.kv_send_thread.add_stored_request(  # type: ignore[union-attr]
                request.req_id
            )
            self.kv_send_thread.add_request(  # type: ignore[union-attr]
                request,
            )

    def get_finished(self, finished_req_ids: set[str], meta: AscendConnectorMetadata) -> tuple[set[str], set[str]]:
        if self.kv_send_thread is not None:
            for req_id in meta.preempted_req_ids:
                self.kv_send_thread.delete_finished_stored_request(req_id)
            self.kv_send_thread.discard_finished_requests(meta.preempted_req_ids)
            if self.use_layerwise:
                self.kv_send_thread.get_and_clear_finished_requests()
                done_sending = set()
            else:
                stale_finished_req_ids = (
                    finished_req_ids - meta.delayed_free_req_ids)
                self.kv_send_thread.discard_finished_requests(
                    stale_finished_req_ids)
                done_sending = self.kv_send_thread.get_and_clear_finished_requests(
                    meta.delayed_free_req_ids
                )
        else:
            done_sending = set()

        done_recving = set()
        if self.kv_recv_thread is not None:
            self.kv_recv_thread.discard_finished_requests(
                meta.preempted_req_ids)
            if self.load_async:
                done_recving = self.kv_recv_thread.get_and_clear_finished_requests(
                    meta.loading_req_ids)

        logger.debug(
            "Number of completed KV cache send requests: %d, receive requests: %d, tp_rank:%d",
            len(done_sending),
            len(done_recving),
            self.tp_rank,
        )
        return done_sending, done_recving

    def get_kv_events(self) -> list[BlockStored]:
        if self.enable_kv_events and self.kv_send_thread is not None:
            # collect store kv events form sending thread
            events = self.kv_send_thread.get_kv_events()
            return events
        return []
