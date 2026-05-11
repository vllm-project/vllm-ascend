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

        self.layer_load_tasks: list[list[LayerTransferTask]] = [[] for i in range(self.num_layers)]
        self.layer_save_tasks: list[list[LayerTransferTask]] = [[] for i in range(self.num_layers)]
        self.layer_load_finished_events = None
        self.layer_save_finished_events = None
        self.layer_transfer_finished_events = None
        self.submitted_layer_loads: set[int] = set()
        self.finished_layer_loads: set[int] = set()
        self.next_layer_to_submit = 0
        layerwise_config = get_layerwise_config(self.num_layers)
        self.layerwise_offload = layerwise_config.has_layer_reuse
        self.NUM_SHARED_BUFFERS = layerwise_config.num_shared_buffers
        self.independent_layers = layerwise_config.independent_layers
        self.layers_need_to_save = layerwise_config.save_layers
        self.layers_need_to_load = layerwise_config.load_layers
        self.prefetch_layer_map = layerwise_config.prefetch_layer_map
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
                    self.my_key_index,
                    self.num_ranks_per_layer,
                    self.page_size_bytes,
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
        if len(metadata.requests) == 0:
            return
        self.current_layer = 0
        if self.use_layerwise:
            self.submitted_layer_loads.clear()
            self.finished_layer_loads.clear()
            self.next_layer_to_submit = 0
            self.process_layer_data(metadata.requests)
            return

        for request in metadata.requests:
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
            save_end_block = request.token_len_chunk // self.block_size
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
            full_blocks = cached_tokens // self.block_size
            partial_block_index = full_blocks if cached_tokens % self.block_size != 0 else None
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
        for layer_id in self.layers_need_to_save:
            self._process_save_for_layer_batch(requests, layer_id)
        for layer_id in self.layers_need_to_load:
            self._process_load_for_layer_batch(requests, layer_id)

    def _submit_ready_layer_loads(self) -> None:
        assert self.kv_recv_thread is not None
        if not self.layers_need_to_load:
            return

        self._submit_layer_load_window()

    def _submit_layer_load_window(self) -> None:
        pending_loads = len(self.submitted_layer_loads -
                            self.finished_layer_loads)
        while (pending_loads < self.NUM_SHARED_BUFFERS
               and self.next_layer_to_submit < self.num_layers):
            layer_id = self.next_layer_to_submit
            self.next_layer_to_submit += 1
            if not self.layer_load_tasks[layer_id]:
                continue
            wait_for_save_layer = self.prefetch_layer_map.get(layer_id)
            self._submit_layer_load(layer_id, wait_for_save_layer)
            pending_loads += 1

    def _submit_layer_load(self, layer_id: int,
                           wait_for_save_layer: int | None) -> None:
        if layer_id in self.submitted_layer_loads:
            return
        if not self.layer_load_tasks[layer_id]:
            return
        self.submitted_layer_loads.add(layer_id)
        self.kv_recv_thread.add_request(
            LayerLoadTask(
                wait_for_save_layer=wait_for_save_layer,
                transfer_tasks=self.layer_load_tasks[layer_id],
                layer_id=layer_id,
            )
        )

    def wait_for_layer_load(self) -> None:
        self._submit_ready_layer_loads()
        should_wait = (
            self.current_layer in self.submitted_layer_loads
            and self.current_layer not in self.finished_layer_loads
        )
        if not should_wait:
            self.layer_load_finished_events[self.current_layer].clear()
            return
        is_finish = self.layer_load_finished_events[self.current_layer].wait(timeout=10)
        if not is_finish:
            logger.info("Layerwise %d load wait timed out", self.current_layer)
        logger.debug(f">>>>>>>>>>>>>>>>>>>> clear load layer {self.current_layer}")
        self.layer_load_finished_events[self.current_layer].clear()
        if is_finish:
            self.finished_layer_loads.add(self.current_layer)
        self._submit_layer_load_window()

    def save_kv_layer(self, connector_metadata: AscendConnectorMetadata) -> None:
        # Wait for KV cache saving to complete on the final layer that requires offloading.
        if self.current_layer in self.layers_need_to_save:
            self.sync_save_events[self.current_layer].record()
            self.kv_send_thread.add_request(self.layer_save_tasks[self.current_layer])
        if self.layers_need_to_save and self.current_layer == self.num_layers - 1:
            is_finish = self.layer_save_finished_events[self.layers_need_to_save[-1]].wait(timeout=10)
            if not is_finish:
                logger.info("Layerwise %d save wait timed out", self.current_layer)
            for layer_id in self.layers_need_to_save:
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
            # The layerwise sender can finish saving the prompt KV before the
            # request finishes decoding. Only clear and report requests after
            # the scheduler has marked them finished.
            done_sending = self.kv_send_thread.get_and_clear_finished_requests(
                finished_req_ids
            )
        else:
            done_sending = set()

        done_recving = (
            self.kv_recv_thread.get_and_clear_finished_requests()  # type: ignore[union-attr]
            if self.load_async and self.kv_recv_thread is not None
            else set()
        )

        logger.debug(
            "Number of completed KV cache send requests: %d, receive requests: %d, tp_rank:%d",
            len(done_sending),
            len(done_recving),
            self.tp_rank,
        )
        return done_sending, done_recving

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
