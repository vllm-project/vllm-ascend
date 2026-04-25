import importlib
import math
import threading

import torch
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
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerMultiBlockReqMeta,
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

backend_map = {
    "mooncake": {
        "name": "MooncakeBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend",
    },
    "memcache": {
        "name": "MemcacheBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend",
    },
    "yuanrong": {
        "name": "YuanrongBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend",
    },
}


class KVPoolWorker:
    # The main class for the cache engine.

    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwize: bool,
    ):
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.dp_rank = parallel_config.data_parallel_rank
        self.use_mla = False
        if hasattr(model_config, "use_mla") and isinstance(model_config.use_mla, bool) and model_config.use_mla:
            self.use_mla = True
        self.use_sparse = hasattr(model_config.hf_text_config, "index_topk")
        self.use_layerwise = use_layerwize
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = (parallel_config.rank // self.tp_size) % self.pp_size

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0

        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get("load_async", False)
        self.consumer_is_to_put = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_put", False
        )
        self.backend = vllm_config.kv_transfer_config.kv_connector_extra_config.get("backend", "mooncake")
        self.original_block_size = vllm_config.cache_config.block_size
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



        self.layer_load_tasks = [[] for i in range(self.num_layers)]
        self.layer_save_tasks = [[] for i in range(self.num_layers)]
        self.layer_load_finished_events = None
        self.layer_save_finished_events = None
        self.layer_transfer_finished_events = None
        # req_id, layer_id, block info
        self._request_addr_tracker: dict[str, dict[int, dict]] = {}

        NUM_SHARED_BUFFERS = 2
        self.NUM_SHARED_BUFFERS = NUM_SHARED_BUFFERS
        INDEPENDENT_LAYER_INDICES = {0, self.num_layers - 1}
        self.independent_layers = list(INDEPENDENT_LAYER_INDICES)

        shared_layer_indices = [i for i in range(self.num_layers)
                                if i not in INDEPENDENT_LAYER_INDICES]
        buffer_owner_indices = shared_layer_indices[:NUM_SHARED_BUFFERS]
        buffer_owner_set = set(buffer_owner_indices)

        self.reuse_mapping = {}
        for i, layer_idx in enumerate(shared_layer_indices):
            if layer_idx not in buffer_owner_set:
                owner_idx = shared_layer_indices[i % NUM_SHARED_BUFFERS]
                self.reuse_mapping[layer_idx] = owner_idx

        self.layer_next_map = {}
        for i in range(len(shared_layer_indices) - NUM_SHARED_BUFFERS):
            self.layer_next_map[shared_layer_indices[i]] = shared_layer_indices[i + NUM_SHARED_BUFFERS]
        self.offload_start_ids = buffer_owner_indices
        self.layers_need_to_save = [i for i in range(self.num_layers) if i not in self.independent_layers]
        self.sync_save_events = None

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

        self.kv_caches = kv_caches
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
                    ready_event_sending,
                    self.num_layers,
                    self.layer_save_finished_events,
                    self.sync_save_events,
                    self.enable_kv_events,
                    self.layer_transfer_finished_events
                )
                self.kv_send_thread.start()
            ready_event = threading.Event()
            self.kv_recv_thread = KVCacheStoreLayerRecvingThread(
                self.m_store,
                self.token_database,
                self.block_size,
                self.tp_rank,
                self.tp_size,
                self.dcp_size,
                ready_event,
                self.get_event,
                self.layer_load_finished_events,
                self.layer_save_finished_events,
                self.num_layers,
            )
            self.kv_recv_thread.start()
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
                    self.m_store, self.token_database, self.block_size, self.tp_rank, self.tp_size, self.dcp_size, ready_event
                )
                self.kv_recv_thread.start()
                ready_event.wait()

    def start_load_kv(self, metadata: AscendConnectorMetadata):
        if len(metadata.requests) == 0:
            return
        self.current_layer = 0
        for request in metadata.requests:
            if self.use_layerwise:
                self.process_layer_data(request)
            else:
                load_spec = request.load_spec
                if load_spec is None or not load_spec.can_load:  # load =0
                    continue
                token_len = request.token_len_chunk
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
        # TODO 这里的请求释放可能有问题
        # logger.info(f">>>>>>>>>>>> metadata.requests {len(metadata.requests)} metadata.unfinished_request_ids {metadata.unfinished_request_ids}")
        # TODO 把这个移到 independent layer attention 计算前
        if len(self.independent_layers)==0 and self.use_layerwise and metadata.unfinished_request_ids:
            for layer_id in self.offload_start_ids:
                layer_load_task = self.layer_load_tasks[layer_id]
                self.kv_recv_thread.add_request((None, layer_load_task, layer_id))

    def _get_or_init_layer_tracker(self, req_id: str, layer_id: int, tracker_key: str | None = None, initial_processed_count: int = 0) -> dict:
        key = tracker_key if tracker_key else req_id
        if key not in self._request_addr_tracker:
            self._request_addr_tracker[key] = {}
        if layer_id not in self._request_addr_tracker[key]:
            self._request_addr_tracker[key][layer_id] = {
                'processed_count': initial_processed_count,
                'chunk_addr_list': [],
                'chunk_size_list': [],
                'chunk_gvas_list': [],
                'last_block_addr': [],
                'last_block_size': [],
                'last_block_gvas': [],
            }
        return self._request_addr_tracker[key][layer_id]

    def _process_chunks_incremental(
        self,
        tracker: dict,
        block_ids: list[int],
        layer_id: int,
        chunk_gvas: list[int],
        num_blocks: int,
    ) -> None:
        processed_count = tracker['processed_count']
        if processed_count >= num_blocks:
            return

        rank_layer_offset = (layer_id * self.num_ranks_per_layer + self.my_key_index) * self.page_size_bytes
        blocks_to_process = num_blocks - processed_count
        for block_idx in range(blocks_to_process):
            start = (processed_count + block_idx) * self.block_size
            end = start + self.block_size
            addr, size = self.token_database.prepare_value_layer(
                start, end, block_ids, layer_id)
            tracker['chunk_addr_list'].extend(addr)
            tracker['chunk_size_list'].extend(size)
            base_gva = chunk_gvas[processed_count + block_idx]
            offset = base_gva + rank_layer_offset
            for s in size:
                tracker['chunk_gvas_list'].append(offset)
                offset += s

        tracker['processed_count'] = num_blocks

    def _process_last_block(
        self,
        tracker: dict,
        block_ids: list[int],
        layer_id: int,
        last_block_gva: int,
        num_blocks: int,
    ) -> None:
        last_block_start = num_blocks * self.block_size
        last_block_end = last_block_start + self.block_size

        addr, size = self.token_database.prepare_value_layer(
            last_block_start, last_block_end, block_ids, layer_id)
        rank_layer_offset = (layer_id * self.num_ranks_per_layer + self.my_key_index) * self.page_size_bytes
        offset = last_block_gva + rank_layer_offset
        last_block_gvas = []
        for s in size:
            last_block_gvas.append(offset)
            offset += s

        tracker['last_block_addr'] = addr
        tracker['last_block_size'] = size
        tracker['last_block_gvas'] = last_block_gvas

    def _build_req_meta(
        self,
        request: ReqMeta,
        layer_id: int,
        tracker: dict,
        is_last_chunk: bool = False,
    ) -> LayerMultiBlockReqMeta:
        final_addr_list = tracker['chunk_addr_list'] + tracker['last_block_addr']
        final_size_list = tracker['chunk_size_list'] + tracker['last_block_size']
        final_gvas_list = tracker['chunk_gvas_list'] + tracker['last_block_gvas']

        req_meta = LayerMultiBlockReqMeta(
            request.req_id,
            request.block_ids, layer_id, is_last_chunk
        )
        req_meta.chunk_gvas = request.chunk_gvas
        req_meta.last_block_gva = request.last_block_gva
        req_meta.addr_list = final_addr_list
        req_meta.size_list = final_size_list
        req_meta.gvas_list = final_gvas_list
        return req_meta

    def _process_save_for_layer(
        self,
        request: ReqMeta,
        layer_id: int,
    ) -> None:
        if request.can_save is None or not request.can_save:
            return

        num_cached_blocks = 0
        if request.load_spec is not None:
            num_cached_blocks = request.load_spec.kvpool_cached_tokens // self.block_size

        num_blocks = request.token_len_chunk // self.block_size
        num_cached_blocks = min(num_cached_blocks, num_blocks)

        tracker = self._get_or_init_layer_tracker(
            request.req_id, layer_id, initial_processed_count=num_cached_blocks)

        has_last_block = request.token_len_chunk % self.block_size != 0
        self._process_chunks_incremental(
            tracker, request.block_ids, layer_id,
            request.chunk_gvas, num_blocks)

        if has_last_block:
            self._process_last_block(
                tracker, request.block_ids, layer_id,
                request.last_block_gva, num_blocks)

        req_meta = self._build_req_meta(
            request, layer_id, tracker, request.is_last_chunk)
        self.layer_save_tasks[layer_id].append(req_meta)

    def _process_load_for_layer(
        self,
        request: ReqMeta,
        layer_id: int,
    ) -> None:
        load_spec = request.load_spec
        if load_spec is None or not load_spec.can_load:
            return

        token_len = load_spec.kvpool_cached_tokens
        num_saved_blocks = token_len // self.block_size
        has_load_last_block = token_len % self.block_size != 0

        load_tracker_key = f"{request.req_id}_load"
        tracker = self._get_or_init_layer_tracker(request.req_id, layer_id, load_tracker_key)

        self._process_chunks_incremental(
            tracker, request.block_ids, layer_id,
            request.chunk_gvas, num_saved_blocks)

        if has_load_last_block:
            self._process_last_block(
                tracker, request.block_ids, layer_id,
                request.last_block_gva, num_saved_blocks)

        req_meta = self._build_req_meta(request, layer_id, tracker)
        self.layer_load_tasks[layer_id].append(req_meta)

    def process_layer_data(self, request: ReqMeta) -> None:
        for layer_id in range(self.num_layers):
            if layer_id in self.independent_layers:
                continue
            self._process_save_for_layer(request, layer_id)
            self._process_load_for_layer(request, layer_id)

    def wait_for_layer_load(self) -> None:
        if self.current_layer in self.independent_layers:
            if self.current_layer == self.independent_layers[0]:
                for layer_id in self.offload_start_ids:
                    logger.debug(f">>>>>>>>>>>>>>>>>>>> load layer {layer_id}")
                    layer_load_task = self.layer_load_tasks[layer_id]
                    self.kv_recv_thread.add_request((None, layer_load_task, layer_id))
            self.layer_load_finished_events[self.current_layer].clear()
            return
        is_finish = self.layer_load_finished_events[self.current_layer].wait(timeout=10)
        if not is_finish:
            logger.info("Layerwise %d load wait timed out", self.current_layer)
        logger.debug(f">>>>>>>>>>>>>>>>>>>> clear load layer {self.current_layer}")
        self.layer_load_finished_events[self.current_layer].clear()

    def save_kv_layer(self, connector_metadata: AscendConnectorMetadata) -> None:
        # Wait for KV cache saving to complete on the final layer that requires offloading.
        if self.current_layer in self.layers_need_to_save:
            self.sync_save_events[self.current_layer].record()
            self.kv_send_thread.add_request(self.layer_save_tasks[self.current_layer])
            # add load task, in both prefill and decode stages
            # 1. wait for save, and clear save event
            # 2. start load, for prefill layer_load_tasks is None, skip load in the recv thread.
            # 3. set layer_load_finished_events (both prefill & decode)
            if self.current_layer in self.layer_next_map:
                next_layer = self.layer_next_map[self.current_layer]
                self.kv_recv_thread.add_request(
                    (self.current_layer, self.layer_load_tasks[next_layer], next_layer))
        if self.current_layer == self.num_layers - 1:
            is_finish = self.layer_save_finished_events[self.layers_need_to_save[-1]].wait(timeout=10)
            if not is_finish:
                logger.info("Layerwise %d save wait timed out", self.current_layer)
            for layer_id in self.layers_need_to_save[-1*self.NUM_SHARED_BUFFERS:]:
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
        # TODO 这里需要确定要按照last chunk 进行判断，prefill完了还有decode，这里返回，decode是否还能正常继续？
        done_sending = (
            self.kv_send_thread.get_and_clear_finished_requests()  # type: ignore[union-attr]
            if self.kv_role in ["kv_producer", "kv_both"] or self.consumer_is_to_put
            else set()
        )

        done_recving = (
            self.kv_recv_thread.get_and_clear_finished_requests()  # type: ignore[union-attr]
            if self.load_async
            else set()
        )

        for req_id in done_sending | done_recving:
            self._cleanup_request_tracker(req_id)

        for req_id in meta.preempted_req_ids:
            self.kv_send_thread.delete_finished_stored_request(  # type: ignore[union-attr]
                req_id
            )

        logger.debug(
            "Number of completed KV cache send requests: %d, receive requests: %d, tp_rank:%d",
            len(done_sending),
            len(done_recving),
            self.tp_rank,
        )
        return done_sending, done_recving

    def _cleanup_request_tracker(self, req_id: str):
        if req_id in self._request_addr_tracker:
            del self._request_addr_tracker[req_id]
        load_tracker_key = f"{req_id}_load"
        if load_tracker_key in self._request_addr_tracker:
            del self._request_addr_tracker[load_tracker_key]



    def lookup(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        use_layerwise: bool,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.
        :param tokens: the input tokens, with shape [seq_len]
        :return: An int indicating how many prefix tokens are cached.
        """
        end = 0
        keys = []
        try:
            starts = []
            for start, end, key in self.token_database.process_tokens(token_len, block_hashes):
                if use_layerwise:
                    keys_multi_layer = key.split_layers(self.num_layers)
                    for item in keys_multi_layer:
                        keys.append(item.to_string())
                else:
                    keys.append(key.to_string())
                starts.append(start)

            res = self.m_store.exists(keys)  # type: ignore[assignment]

            if use_layerwise:
                res = self.check_all_layers_exists(res, self.num_layers)
            for index, value in enumerate(res):  # type: ignore[arg-type]
                if value != 1:
                    return starts[index]
            # all tokens where found, return the maximal end
        except Exception as e:
            logger.error(f"Remote connection failed in contains: {e}")
            return start
        return end

    def lookup_scheduler(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        use_layerwise: bool,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.
        :param tokens: the input tokens, with shape [seq_len]
        :return: An int indicating how many prefix tokens are cached.
        """
        end = 0
        keys = []
        try:
            starts = []
            for start, end, key in self.token_database.process_tokens(token_len, block_hashes):
                if use_layerwise:
                    keys_multi_layer = key.split_layers(self.num_layers)
                    for item in keys_multi_layer:
                        keys.append(item.to_string())
                else:
                    keys.append(key.to_string())
                starts.append(start)

            multi_tp_keys = keys[:]
            for i in range(1, min(self.tp_size, self.num_kv_head)):
                for item in keys:
                    new_str = item.replace(  # type: ignore[attr-defined]
                        "@head_or_tp_rank:0", f"@head_or_tp_rank:{i}", 1
                    )
                    multi_tp_keys.append(new_str)

            pp_base_keys = multi_tp_keys.copy()
            for i in range(1, self.pp_size):
                for item in pp_base_keys:
                    new_str = item.replace(  # type: ignore[attr-defined]
                        "@pp_rank:0", f"@pp_rank:{i}", 1
                    )
                    multi_tp_keys.append(new_str)

            res = self.m_store.exists(multi_tp_keys)  # type: ignore[assignment]
            num_block = len(keys)
            if use_layerwise:
                res = self.check_all_layers_exists(res, self.num_layers)
                num_block = len(keys) // self.num_layers
            multi_tp_values = [
                res[i * num_block : (i + 1) * num_block]  # type: ignore[index]
                for i in range(min(self.tp_size, self.num_kv_head) * self.pp_size)
            ]
            index = self.find_min_first_non_one_index(multi_tp_values)
            if index != -1:
                return starts[index]
        # all tokens where found, return the maximal end
        except Exception as e:
            logger.error(f"Remote connection failed in contains: {e}")
            return start
        return end

    def check_all_layers_exists(self, res: list[int], num_layers: int) -> list[int]:
        total_chunks = len(res) // num_layers
        result = []

        for chunk_idx in range(total_chunks):
            start = chunk_idx * num_layers
            end = start + num_layers
            chunk = res[start:end]
            result.append(1 if all(x == 1 for x in chunk) else 0)

        return result

    def find_min_first_non_one_index(self, arr):
        try:
            return min(idx for row in arr for idx, val in enumerate(row) if val != 1)
        except ValueError:
            return -1

    def get_kv_events(self) -> list[BlockStored]:
        if self.enable_kv_events and self.kv_send_thread is not None:
            # collect store kv events form sending thread
            events = self.kv_send_thread.get_kv_events()
            return events
        return []
