import importlib
import math
import threading
from collections.abc import Generator

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

        # Peer TP size:
        # - On prefill (kv_producer / kv_both): peer is decode -> use decode_tp_size
        # - On decode (kv_consumer with consumer_is_to_put): peer is prefill -> use prefill_tp_size
        extra_cfg = vllm_config.kv_transfer_config.kv_connector_extra_config
        if self.kv_role == "kv_consumer":
            self.peer_tp_size = int(extra_cfg.get("prefill_tp_size", self.tp_size))
        else:
            self.peer_tp_size = int(extra_cfg.get("decode_tp_size", self.tp_size))
        # Keep decode_tp_size for backward-compat in lookup_scheduler key expansion
        self.decode_tp_size = int(extra_cfg.get("decode_tp_size", self.tp_size))
        # TP mismatch: local TP differs from peer TP.
        # Only relevant for non-MLA models where each TP rank handles different KV heads,
        # and num_kv_head must be >= max(local_tp, peer_tp) so that heads divide evenly.
        self.effective_tp_size = max(self.tp_size, self.peer_tp_size)
        self.tp_mismatch = (
            self.peer_tp_size != self.tp_size
            and not self.use_mla
            and self.num_kv_head >= self.effective_tp_size
            and self.num_kv_head % self.effective_tp_size == 0
        )
        if self.tp_mismatch:
            self.local_heads_per_rank = self.num_kv_head // self.tp_size
            self.effective_heads_per_rank = self.num_kv_head // self.effective_tp_size
            self.num_sub_keys = self.local_heads_per_rank // self.effective_heads_per_rank
            logger.info(
                "TP mismatch detected: local_tp=%d, peer_tp=%d, effective_tp=%d, "
                "local_heads_per_rank=%d, effective_heads_per_rank=%d, num_sub_keys=%d",
                self.tp_size,
                self.peer_tp_size,
                self.effective_tp_size,
                self.local_heads_per_rank,
                self.effective_heads_per_rank,
                self.num_sub_keys,
            )
        else:
            self.local_heads_per_rank = (
                self.num_kv_head // self.tp_size if self.tp_size <= self.num_kv_head else 1
            )
            self.effective_heads_per_rank = self.local_heads_per_rank
            self.num_sub_keys = 1

        if self.num_kv_head < self.tp_size:
            self.put_step = self.tp_size // self.num_kv_head
            self.head_or_tp_rank = self.tp_rank // self.put_step
        else:
            self.head_or_tp_rank = self.tp_rank
            self.put_step = 1

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

        self.finished_store_req: set[str] = set()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache = first_kv_cache_tuple[0]

        self.num_blocks = first_kv_cache.shape[0]
        logger.info("num_blocks: %s", self.num_blocks)
        block_rank = 3
        self.block_len = []
        if self.use_mla or self.use_sparse:
            for i in range(len(first_kv_cache_tuple)):
                block_shape = first_kv_cache_tuple[i].shape[-block_rank:]
                logger.info("block_shape: %s", block_shape)
                self.block_len.append(first_kv_cache[i].element_size() * math.prod(block_shape))
        else:
            # [num_block, block_size, num_head, hidden_dim]
            block_shape = first_kv_cache.shape[-block_rank:]
            logger.info("block_shape: %s", block_shape)
            self.block_len = [first_kv_cache.element_size() * math.prod(block_shape)]

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
            # Normalize to always be a list of caches
            for i, cache in enumerate(cache_or_caches, 0):
                base_addr = cache.data_ptr()
                region_len = self.num_blocks * self.block_len[i % length]
                self.kv_caches_base_addr.append(base_addr)
                ptrs.append(base_addr)
                lengths.append(region_len)

        self.m_store.register_buffer(ptrs, lengths)
        self.token_database.set_kv_caches_base_addr(self.kv_caches_base_addr)
        self.token_database.set_block_len(self.block_len)

        if self.tp_mismatch:
            first_cache = next(iter(kv_caches.values()))[0]
            self.elem_size = first_cache.element_size()
            self.head_dim = first_cache.shape[-1]
            # Per-token bytes within one block for one layer-entry (K or V):
            # block_len[0] = block_size * num_kv_head_per_local_rank * head_dim * elem_size
            self.per_token_bytes = self.block_len[0] // self.block_size
            # Bytes occupied by one sub-key's heads within one token
            self.sub_size_bytes = self.effective_heads_per_rank * self.head_dim * self.elem_size
            logger.info(
                "TP mismatch strided I/O: per_token_bytes=%d, sub_size_bytes=%d",
                self.per_token_bytes,
                self.sub_size_bytes,
            )

        if self.use_layerwise:
            self.get_event = threading.Event()
            if self.kv_role in ["kv_producer", "kv_both"]:
                ready_event_sending = threading.Event()
                self.kv_send_thread = KVCacheStoreLayerSendingThread(
                    self.m_store,
                    self.token_database,
                    self.block_size,
                    self.tp_rank,
                    self.dcp_size,
                    self.put_step,
                    ready_event_sending,
                    self.num_layers,
                    self.enable_kv_events,
                )
                self.kv_send_thread.start()
            ready_event = threading.Event()
            self.kv_recv_thread = KVCacheStoreLayerRecvingThread(
                self.m_store,
                self.token_database,
                self.block_size,
                self.tp_rank,
                self.dcp_size,
                ready_event,
                self.get_event,
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
                    self.dcp_size,
                    self.put_step,
                    self.kv_role,
                    ready_event_sending,
                    self.enable_kv_events,
                    worker=self if self.tp_mismatch else None,
                )
                self.kv_send_thread.start()
            if self.load_async:
                ready_event = threading.Event()
                self.kv_recv_thread = KVCacheStoreRecvingThread(
                    self.m_store,
                    self.token_database,
                    self.block_size,
                    self.tp_rank,
                    self.dcp_size,
                    ready_event,
                    worker=self if self.tp_mismatch else None,
                )
                self.kv_recv_thread.start()
                ready_event.wait()

    def start_load_kv(self, metadata: AscendConnectorMetadata):
        self.current_layer = 0
        self.layerwise_retrievers = []
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
            if self.use_layerwise:
                layerwise_retriever = self.retrieve_layer(request)
                next(layerwise_retriever)  # first layer load
                self.layerwise_retrievers.append(layerwise_retriever)
            else:
                if self.load_async:
                    self.kv_recv_thread.add_request(  # type: ignore[union-attr]
                        request,
                    )
                else:
                    mask_num = request.load_spec.vllm_cached_tokens // self.block_size * self.block_size
                    if self.tp_mismatch:
                        self._load_kv_tp_mismatch(
                            request.block_hashes, request.block_ids, token_len, mask_num
                        )
                    else:
                        addr_list = []
                        size_list = []
                        key_list = []
                        for start, end, key in self.token_database.process_tokens(
                            token_len, request.block_hashes, mask_num
                        ):
                            addr, size, _ = self.token_database.prepare_value(start, end, request.block_ids)
                            key_list.append(key.to_string())
                            addr_list.append(addr)
                            size_list.append(size)
                        key_list_c = key_list[self.tp_rank % len(key_list) :] + key_list[: self.tp_rank % len(key_list)]
                        addr_list_c = (
                            addr_list[self.tp_rank % len(addr_list) :] + addr_list[: self.tp_rank % len(addr_list)]
                        )
                        size_list_c = (
                            size_list[self.tp_rank % len(size_list) :] + size_list[: self.tp_rank % len(size_list)]
                        )
                        self.m_store.get(key_list_c, addr_list_c, size_list_c)

    def wait_for_layer_load(self) -> None:
        for layerwise_retriever in self.layerwise_retrievers:
            ret_token_mask = next(layerwise_retriever)
            if self.current_layer == self.num_layers - 1:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.debug(f"Retrieved {num_retrieved_tokens} tokens")

    def save_kv_layer(self, connector_metadata: AscendConnectorMetadata) -> None:
        if self.current_layer == 0:
            self.layerwise_storers = []
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

                layerwise_storer = self.store_layer(request, current_event)
                self.layerwise_storers.append(layerwise_storer)
        for layerwise_storer in self.layerwise_storers:
            try:
                next(layerwise_storer)
            except Exception:
                raise
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

    def retrieve_layer(
        self,
        request: ReqMeta,
    ) -> Generator[torch.Tensor | None, None, None]:
        """
        Retrieve the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the KV transfer which
            will be passed into the npu_transfer.

        return: A generator that yields Optional[torch.Tensor]. The tensor will
            be the boolean mask indicating which tokens are retrieved and will
            only be returned in the last iteration.
        """
        token_len = request.token_len_chunk
        mask_num = (
            request.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // self.block_size
            * self.block_size
        )
        num_required_tokens = token_len - mask_num

        ret_mask = torch.zeros(token_len, dtype=torch.bool, device="cpu")

        starts = []
        ends = []
        keys = []
        first_flag = True
        for start, end, key in self.token_database.process_tokens(token_len, request.block_hashes, mask_num):
            keys_multi_layer = key.split_layers(self.num_layers)
            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
            ret_mask[start:end] = True

        if keys:
            # Transpose the keys into layer major format
            keys = [list(row) for row in zip(*keys)]  # [num_layer,block_num]
            for layer_id, keys_multi_chunk in enumerate(keys):
                if not first_flag:
                    is_finish = self.get_event.wait(timeout=3)  # try---cache
                    if not is_finish:
                        logger.info("Layerwise get failed")
                self.get_event.clear()
                req_meta = LayerMultiBlockReqMeta(
                    request.req_id, keys_multi_chunk, starts, ends, request.block_ids, layer_id
                )
                self.kv_recv_thread.add_request(  # type: ignore[union-attr, call-arg]
                    req_meta
                )  # type: ignore[union-attr, call-arg, arg-type]
                first_flag = False
                yield None
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield None

        retrieved_tokens = torch.sum(ret_mask)
        logger.debug(f"Retrieved {retrieved_tokens} out of {num_required_tokens} out of total {token_len} tokens")

        yield ret_mask

    def store_layer(
        self,
        request: ReqMeta,
        current_event: torch.npu.Event | None,
    ) -> Generator[None, None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields None. In the first iteration, the
            generator allocates the memory objects for all layers and moves
            the KV cache of the first layer from GPU to CPU. In the next
            iterations, it moves the KV cache of layer i from GPU to the memory
            objects (on CPU) and puts the memory objects of layer i-1 to the
            storage backends. In the last iteration, it puts the memory objects
            of the last layer to the storage backends.
        """
        starts = []
        ends = []
        keys = []
        for start, end, key in self.token_database.process_tokens(request.token_len_chunk, request.block_hashes):
            keys_multi_layer = key.split_layers(self.num_layers)
            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)  # [block_num,layer_num]

        if keys:
            keys = [list(row) for row in zip(*keys)]  # [layer_num,block_num]
            for layer_id, keys_multi_chunk in enumerate(keys):
                req_meta = LayerMultiBlockReqMeta(
                    request.req_id,
                    keys_multi_chunk,
                    starts,
                    ends,
                    request.block_ids,
                    layer_id,
                    request.is_last_chunk,
                    current_event,
                )
                self.kv_send_thread.add_request(  # type: ignore[union-attr, call-arg]
                    req_meta
                )  # type: ignore[union-attr, call-arg, arg-type]
                yield
        else:
            for layer_id in range(self.num_layers):
                yield

    def _make_sub_key_str(self, base_key, effective_rank: int) -> str:
        """Override the head_or_tp_rank field of base_key.to_string() with effective_rank.

        Under TP mismatch, both sides agree to address the pool at granularity
        ``effective_tp_size = max(local_tp, peer_tp)``. The key field
        ``@head_or_tp_rank:`` therefore carries the effective rank rather than
        the local TP rank.
        """
        key_str = base_key.to_string()
        return key_str.replace(
            f"@head_or_tp_rank:{self.metadata.head_or_tp_rank}",
            f"@head_or_tp_rank:{effective_rank}",
            1,
        )

    def _build_strided_addrs(
        self, block_id: int, token_count: int, sub_idx: int
    ) -> tuple[list[int], list[int]]:
        """Build per-token strided (addr, size) pairs into local KV cache memory
        for one sub-key at one block.

        For each layer-entry (K and V of every layer) and each token within the
        block, emit one (addr, size) pair pointing to the slice of heads that
        belongs to ``sub_idx`` of this rank's local cache. The KV cache layout
        is ``[num_block, block_size, num_kv_head_per_local_rank, head_dim]``,
        so the heads of consecutive tokens are interleaved with token positions
        and a sub-slice of heads requires one transfer per token.
        """
        head_offset_bytes = sub_idx * self.sub_size_bytes
        addrs: list[int] = []
        sizes: list[int] = []
        for base_addr in self.kv_caches_base_addr:
            block_base = base_addr + block_id * self.block_len[0]
            for t in range(token_count):
                addrs.append(block_base + t * self.per_token_bytes + head_offset_bytes)
                sizes.append(self.sub_size_bytes)
        return addrs, sizes

    def _build_tp_mismatch_keys_and_addrs(
        self,
        block_hashes: list,
        block_ids: list[int],
        token_len: int,
        mask_num: int = 0,
    ) -> tuple[list[str], list[list[int]], list[list[int]]]:
        """Walk all chunks × sub-keys and build (keys, addrs, sizes) suitable
        for backend.put / backend.get.

        Each emitted key represents one (chunk, sub_idx) pair. Its addrs/sizes
        cover all layer-entries × all tokens in the chunk, addressed at the
        head-slice owned by sub_idx within this rank's local cache.
        """
        all_keys: list[str] = []
        all_addrs: list[list[int]] = []
        all_sizes: list[list[int]] = []
        for start, end, base_key in self.token_database.process_tokens(
            token_len, block_hashes, mask_num
        ):
            block_id = block_ids[start // self.block_size]
            token_count = end - start
            for sub_idx in range(self.num_sub_keys):
                effective_rank = self.tp_rank * self.num_sub_keys + sub_idx
                addrs, sizes = self._build_strided_addrs(block_id, token_count, sub_idx)
                all_keys.append(self._make_sub_key_str(base_key, effective_rank))
                all_addrs.append(addrs)
                all_sizes.append(sizes)
        return all_keys, all_addrs, all_sizes

    def _load_kv_tp_mismatch(
        self,
        block_hashes: list,
        block_ids: list[int],
        token_len: int,
        mask_num: int,
    ) -> None:
        """Load KV cache with TP mismatch by issuing strided gets.

        Each prefill rank generates ``num_sub_keys`` keys per chunk and tells
        the backend to write the result directly into the per-token strided
        positions of its own KV cache. No temporary buffer is required.
        """
        keys, addrs, sizes = self._build_tp_mismatch_keys_and_addrs(
            block_hashes, block_ids, token_len, mask_num
        )
        if not keys:
            return
        # Optional rotation for load-balancing across mooncake masters,
        # mirroring the non-mismatch path in KVCacheStoreRecvingThread.
        offset = self.tp_rank % len(keys)
        keys_c = keys[offset:] + keys[:offset]
        addrs_c = addrs[offset:] + addrs[:offset]
        sizes_c = sizes[offset:] + sizes[:offset]
        logger.info(f"keys_c:{keys_c}, addrs_c:{addrs_c}, sizes_c:{sizes_c}")
        self.m_store.get(keys_c, addrs_c, sizes_c)

    def _store_kv_tp_mismatch(self, req_meta: ReqMeta) -> None:
        """Store KV cache with TP mismatch by issuing strided puts.

        On the decode side this stores each (chunk, sub_idx) combination so the
        keys align with prefill's effective TP layout. Existing keys are
        skipped via a lookup, mirroring KVCacheStoreSendingThread._handle_request.
        """
        if self.kv_send_thread is None:
            return
        req_id = req_meta.req_id
        # Match the dedup gating in the send thread.
        if req_id not in self.kv_send_thread.stored_requests:  # type: ignore[attr-defined]
            return
        token_len = req_meta.token_len_chunk
        keys, addrs, sizes = self._build_tp_mismatch_keys_and_addrs(
            req_meta.block_hashes, req_meta.block_ids, token_len, mask_num=0
        )
        if not keys:
            self.kv_send_thread.dec_stored_request(req_id)  # type: ignore[attr-defined]
            return
        exists_states = self.kv_send_thread.lookup(keys)
        missing_indices = [i for i, exists in enumerate(exists_states) if not exists]
        if not missing_indices:
            self.kv_send_thread.dec_stored_request(req_id)  # type: ignore[attr-defined]
            return
        keys = [keys[i] for i in missing_indices]
        addrs = [addrs[i] for i in missing_indices]
        sizes = [sizes[i] for i in missing_indices]
        if req_meta.current_event is not None:
            req_meta.current_event.synchronize()
        logger.info(f"keys:{keys}, addrs:{addrs}, sizes:{sizes}")
        self.m_store.put(keys, addrs, sizes)
        logger.debug(
            "TP-mismatch stored %d sub-keys for request %s",
            len(keys),
            req_id,
        )
        self.kv_send_thread.dec_stored_request(req_id)  # type: ignore[attr-defined]

    def get_finished(self, finished_req_ids: set[str], meta: AscendConnectorMetadata) -> tuple[set[str], set[str]]:
        done_sending = (
            self.get_and_clear_finished_requests(
                finished_req_ids,
                meta,  # type: ignore[union-attr]
            )
            if self.kv_role in ["kv_producer", "kv_both"] or self.consumer_is_to_put
            else set()
        )

        done_recving = (
            self.kv_recv_thread.get_and_clear_finished_requests(  # type: ignore[union-attr]
            )
            if self.load_async
            else set()
        )

        logger.debug(
            "Number of completed KV cache send requests: %d, receive requests: %d, tp_rank:%d",
            len(done_sending),
            len(done_recving),
            self.tp_rank,
        )
        return done_sending, done_recving

    def get_and_clear_finished_requests(self, finished_req_ids, meta: AscendConnectorMetadata) -> set[str]:
        finished_sending = set()
        for req_id in meta.preempted_req_ids:
            self.kv_send_thread.delete_finished_stored_request(  # type: ignore[union-attr]
                req_id
            )
        for req_id in self.kv_send_thread.stored_requests.copy(  # type: ignore[union-attr]
        ):
            if (
                self.kv_send_thread.stored_requests[  # type: ignore[union-attr]
                    req_id
                ]
                == 0
                and req_id in self.finished_store_req
            ):
                self.finished_store_req.remove(req_id)
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(  # type: ignore[union-attr]
                    req_id
                )

        for req_id in finished_req_ids:
            req_remain_jobs = self.kv_send_thread.stored_requests.get(  # type: ignore[union-attr]
                req_id
            )
            if req_remain_jobs == 0:
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(  # type: ignore[union-attr]
                    req_id
                )
            elif req_remain_jobs is not None:
                self.finished_store_req.add(req_id)

        return finished_sending

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
            return 0
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

            # Under TP mismatch the keys live in the effective_tp_size namespace
            # (= max(local_tp, peer_tp)). Otherwise falls back to local_tp.
            check_tp_size = self.effective_tp_size
            multi_tp_keys = keys[:]
            for i in range(1, min(check_tp_size, self.num_kv_head)):
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
                for i in range(min(check_tp_size, self.num_kv_head) * self.pp_size)
            ]
            index = self.find_min_first_non_one_index(multi_tp_values)
            if index != -1:
                return starts[index]
        # all tokens where found, return the maximal end
        except Exception as e:
            logger.error(f"Remote connection failed in contains: {e}")
            return 0
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
