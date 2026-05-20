import dataclasses
import importlib
import logging
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
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerMultiBlockReqMeta,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import (
    ExternalCachedBlockPool,
    MooncakeStoreCoordinator,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    _resolve_block_sizes_compat,
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
        kv_cache_config: "KVCacheConfig | None" = None,
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

        # HMA-aware block sizes:
        # - ``scheduler_block_size`` is the LCM of per-group block sizes (used
        #   by the engine scheduler and our ``ReqMeta.token_len_chunk`` math);
        # - ``hash_block_size`` is the granularity at which the engine
        #   produced ``Request.block_hashes`` (GCD or override).
        # Single-group / legacy: both equal ``cache_config.block_size *
        # pcp_size * dcp_size`` — bit-identical to the pre-HMA formula.
        self.kv_cache_config = kv_cache_config
        scheduler_block_size, hash_block_size = _resolve_block_sizes_compat(
            vllm_config, kv_cache_config
        )
        self.block_size = scheduler_block_size
        self.hash_block_size = hash_block_size
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

        # HMA: a per-group ChunkedTokenDatabase. Each database knows its
        # own group block_size and is keyed by ``group_id`` (embedded in
        # KeyMetadata) so its emitted PoolKey can be distinguished from
        # other groups. ``token_database`` (singular) aliases group 0 so
        # legacy single-group code paths keep working unchanged.
        self.kv_cache_groups: list[KVCacheGroupSpec] = self._build_kv_cache_groups(
            kv_cache_config, scheduler_block_size
        )
        self.token_databases: list[ChunkedTokenDatabase] = []
        for group_idx, group in enumerate(self.kv_cache_groups):
            per_group_metadata = dataclasses.replace(self.metadata, group_id=group_idx)
            self.token_databases.append(
                ChunkedTokenDatabase(
                    per_group_metadata,
                    group.kv_cache_spec.block_size,
                    partitions,
                    hash_block_size=hash_block_size,
                )
            )
        self.token_database = self.token_databases[0]

        self.coord: MooncakeStoreCoordinator | None = None
        if kv_cache_config is not None and len(kv_cache_config.kv_cache_groups) > 1:
            self.coord = MooncakeStoreCoordinator(
                kv_cache_groups=kv_cache_config.kv_cache_groups,
                scheduler_block_size=scheduler_block_size,
                hash_block_size=hash_block_size,
            )

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

    @staticmethod
    def _build_kv_cache_groups(
        kv_cache_config: "KVCacheConfig | None",
        scheduler_block_size: int,
    ) -> list["KVCacheGroupSpec"]:
        """Return the real kv_cache_groups when HMA is on; otherwise a
        synthetic 1-group fallback so the rest of the worker can iterate
        groups uniformly. The synthetic group's spec is unused beyond its
        block_size (we never allocate from it).
        """
        if kv_cache_config is not None:
            return list(kv_cache_config.kv_cache_groups)
        synthetic_spec = FullAttentionSpec(
            block_size=scheduler_block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float16,
        )
        return [KVCacheGroupSpec(layer_names=[], kv_cache_spec=synthetic_spec)]

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
        # HMA: partition per-layer registration by kv-cache group so each
        # group's ChunkedTokenDatabase only sees its own layers' base
        # addresses *and* a block_len/num_blocks computed from its own
        # tensors (heterogeneous group shapes — e.g. Mamba state vs.
        # full-attn KV — would otherwise corrupt pointer math). Without
        # coord (single-group / legacy), keep the original bit-identical
        # registration loop.
        if self.coord is not None:
            group_base_addrs: list[list[int]] = [[] for _ in self.kv_cache_groups]
            group_block_lens: list[list[int]] = [[] for _ in self.kv_cache_groups]
            layer_to_group: dict[str, int] = {
                layer: g_idx
                for g_idx, g in enumerate(self.kv_cache_groups)
                for layer in g.layer_names
            }
            for layer_name, cache_or_caches in kv_caches.items():
                g_idx = layer_to_group.get(layer_name)
                if g_idx is None:
                    # Layer not bound to any group (e.g. eagle/draft) — skip.
                    continue
                for cache in cache_or_caches:
                    base_addr = cache.data_ptr()
                    block_shape = cache.shape[-block_rank:]
                    per_block_bytes = cache.element_size() * math.prod(block_shape)
                    per_group_num_blocks = cache.shape[0]
                    region_len = per_group_num_blocks * per_block_bytes
                    group_base_addrs[g_idx].append(base_addr)
                    group_block_lens[g_idx].append(per_block_bytes)
                    self.kv_caches_base_addr.append(base_addr)
                    ptrs.append(base_addr)
                    lengths.append(region_len)
            for g_idx, db in enumerate(self.token_databases):
                db.set_kv_caches_base_addr(group_base_addrs[g_idx])
                db.set_block_len(group_block_lens[g_idx] or self.block_len)
        else:
            for cache_or_caches in kv_caches.values():
                # Normalize to always be a list of caches
                for i, cache in enumerate(cache_or_caches, 0):
                    base_addr = cache.data_ptr()
                    region_len = self.num_blocks * self.block_len[i % length]
                    self.kv_caches_base_addr.append(base_addr)
                    ptrs.append(base_addr)
                    lengths.append(region_len)
            self.token_database.set_kv_caches_base_addr(self.kv_caches_base_addr)
            self.token_database.set_block_len(self.block_len)

        self.m_store.register_buffer(ptrs, lengths)

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
                    self.token_databases,
                    self.block_size,
                    self.tp_rank,
                    self.dcp_size,
                    self.put_step,
                    self.kv_role,
                    ready_event_sending,
                    self.enable_kv_events,
                    coord=self.coord,
                )
                self.kv_send_thread.start()
            if self.load_async:
                ready_event = threading.Event()
                self.kv_recv_thread = KVCacheStoreRecvingThread(
                    self.m_store,
                    self.token_databases,
                    self.block_size,
                    self.tp_rank,
                    self.dcp_size,
                    ready_event,
                    coord=self.coord,
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
                    addr_list = []
                    size_list = []
                    key_list = []
                    mask_num = request.load_spec.vllm_cached_tokens // self.block_size * self.block_size
                    # HMA: only chunks that this group's spec populates
                    # locally need to be fetched. Without coord (single-group
                    # legacy), every chunk is "in" — load_masks degenerate to
                    # all-True per group.
                    if self.coord is not None:
                        load_masks = self.coord.load_mask(request.block_hashes, token_len)
                    else:
                        load_masks = None
                    for g_idx, db in enumerate(self.token_databases):
                        group_block_ids = request.block_ids[g_idx]
                        group_mask = load_masks[g_idx] if load_masks is not None else None
                        for chunk_idx, (start, end, key) in enumerate(
                            db.process_tokens(token_len, request.block_hashes, mask_num)
                        ):
                            if group_mask is not None and (
                                chunk_idx >= len(group_mask) or not group_mask[chunk_idx]
                            ):
                                continue
                            addr, size, _ = db.prepare_value(start, end, group_block_ids)
                            key_list.append(key.to_string())
                            addr_list.append(addr)
                            size_list.append(size)
                    if not key_list:
                        continue
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
                logger.debug("Retrieved %s tokens", num_retrieved_tokens)

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
            # Layerwise is single-group only (HMA forbids layerwise at the
            # connector). ``block_ids`` is a 1-tuple — flatten to legacy
            # ``list[int]`` so LayerMultiBlockReqMeta stays untouched.
            flat_block_ids = request.block_ids[0]
            for layer_id, keys_multi_chunk in enumerate(keys):
                if not first_flag:
                    is_finish = self.get_event.wait(timeout=3)  # try---cache
                    if not is_finish:
                        logger.info("Layerwise get failed")
                self.get_event.clear()
                req_meta = LayerMultiBlockReqMeta(
                    request.req_id, keys_multi_chunk, starts, ends, flat_block_ids, layer_id
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
        logger.debug("Retrieved %s out of %s out of total %s tokens", retrieved_tokens, num_required_tokens, token_len)

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
            # Layerwise is single-group; flatten the 1-tuple block_ids.
            flat_block_ids = request.block_ids[0]
            for layer_id, keys_multi_chunk in enumerate(keys):
                req_meta = LayerMultiBlockReqMeta(
                    request.req_id,
                    keys_multi_chunk,
                    starts,
                    ends,
                    flat_block_ids,
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

        if logger.isEnabledFor(logging.DEBUG):
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

        HMA: query each group's keys individually, build a ``(group_id, hash)``
        exists set, then ask the coordinator for the longest **lcm-aligned**
        prefix where every group has a hit — this is the true number of
        usable cached tokens. Legacy single-group: identical to the
        pre-HMA scan.

        :param tokens: the input tokens, with shape [seq_len]
        :return: An int indicating how many prefix tokens are cached.
        """
        end = 0
        try:
            # Single-group / legacy fast path: bit-identical to pre-HMA.
            if self.coord is None:
                keys: list[str] = []
                starts: list[int] = []
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
                return end

            # HMA path: ask each group's database for its per-chunk keys,
            # call ``m_store.exists`` in one batch, then build the exists
            # set keyed by ``(group_id, raw_hash_bytes)`` and ask the
            # coordinator for the convergent lcm-aligned hit length.
            return self._lookup_hma(token_len, block_hashes, use_layerwise)
        except Exception as e:
            logger.error("Remote connection failed in contains: %s", e)
            return 0

    def _lookup_hma(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        use_layerwise: bool,
    ) -> int:
        """HMA worker-side lookup: ask the coordinator for the longest
        lcm-aligned prefix where every group has its chunk present in the
        external store.
        """
        assert self.coord is not None
        all_keys: list[str] = []
        # Parallel list — for each query (one per chunk per group) keep
        # the ``(group_id, hash_bytes)`` we'll insert into the exists set
        # when the corresponding ``exists`` reply is True.
        per_query: list[tuple[int, bytes]] = []
        for g_idx, db in enumerate(self.token_databases):
            for _start, _end, key in db.process_tokens(token_len, block_hashes):
                hash_bytes = bytes.fromhex(key.chunk_hash)
                per_query.append((g_idx, hash_bytes))
                if use_layerwise:
                    for item in key.split_layers(self.num_layers):
                        all_keys.append(item.to_string())
                else:
                    all_keys.append(key.to_string())
        if not all_keys:
            return 0
        res = self.m_store.exists(all_keys)
        if use_layerwise:
            res = self.check_all_layers_exists(res, self.num_layers)
        exists: set[tuple[int, bytes]] = set()
        for i, gid_hash in enumerate(per_query):
            if res[i] == 1:
                exists.add(gid_hash)
        pool = ExternalCachedBlockPool(exists)
        _, hit_length = self.coord.find_longest_cache_hit(
            block_hashes, token_len, pool
        )
        return hit_length

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

        HMA: like ``lookup`` but cross-checks TP/PP rank replicas. A chunk
        is "present" only when every rank-replica reports it. The HMA
        coordinator then computes the longest lcm-aligned prefix where
        every group has a chunk present.
        """
        end = 0
        try:
            # Single-group / legacy fast path.
            if self.coord is None:
                keys: list[str] = []
                starts: list[int] = []
                for start, end, key in self.token_database.process_tokens(token_len, block_hashes):
                    if use_layerwise:
                        keys_multi_layer = key.split_layers(self.num_layers)
                        for item in keys_multi_layer:
                            keys.append(item.to_string())
                    else:
                        keys.append(key.to_string())
                    starts.append(start)
                multi_tp_keys = self._tp_pp_broadcast_keys(keys)
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
                return end

            # HMA path.
            return self._lookup_scheduler_hma(token_len, block_hashes, use_layerwise)
        except Exception as e:
            logger.error("Remote connection failed in contains: %s", e)
            return 0

    def _tp_pp_broadcast_keys(self, keys: list[str]) -> list[str]:
        """Replicate ``keys`` across the (head_or_tp, pp) rank product so the
        scheduler-side check verifies that **every** worker rank has the
        chunk. Layout matches the original pre-HMA grouping:
        ``min(tp_size, num_kv_head) * pp_size`` replicas, contiguous.
        """
        multi_tp_keys = list(keys)
        for i in range(1, min(self.tp_size, self.num_kv_head)):
            for item in keys:
                multi_tp_keys.append(
                    item.replace("@head_or_tp_rank:0", f"@head_or_tp_rank:{i}", 1)
                )
        pp_base_keys = multi_tp_keys.copy()
        for i in range(1, self.pp_size):
            for item in pp_base_keys:
                multi_tp_keys.append(
                    item.replace("@pp_rank:0", f"@pp_rank:{i}", 1)
                )
        return multi_tp_keys

    def _lookup_scheduler_hma(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        use_layerwise: bool,
    ) -> int:
        """For each group: emit per-chunk keys, broadcast across TP/PP rank
        replicas, intersect presence (a chunk counts only if every replica
        reports it), then collect ``(group_id, hash_bytes)`` for present
        chunks. Coordinator turns that exists set into an lcm-aligned hit
        length consistent across groups.
        """
        assert self.coord is not None
        replicas = min(self.tp_size, self.num_kv_head) * self.pp_size
        exists: set[tuple[int, bytes]] = set()

        # Per-group, independent batched query. Done per group rather than
        # one big batch to keep replica-vs-chunk indexing local.
        for g_idx, db in enumerate(self.token_databases):
            chunk_hashes: list[bytes] = []
            base_keys: list[str] = []
            for _start, _end, key in db.process_tokens(token_len, block_hashes):
                chunk_hashes.append(bytes.fromhex(key.chunk_hash))
                if use_layerwise:
                    for item in key.split_layers(self.num_layers):
                        base_keys.append(item.to_string())
                else:
                    base_keys.append(key.to_string())
            if not base_keys:
                continue
            replicated = self._tp_pp_broadcast_keys(base_keys)
            res = self.m_store.exists(replicated)
            if use_layerwise:
                res = self.check_all_layers_exists(res, self.num_layers)
            # ``res`` now has length ``replicas * n_chunks`` in chunk units
            # (post layer-collapse); replica-major ordering.
            n_chunks = len(chunk_hashes)
            if len(res) < replicas * n_chunks:
                # Defensive: backend returned short result, treat as miss.
                continue
            for chunk_idx in range(n_chunks):
                if all(
                    res[r * n_chunks + chunk_idx] == 1 for r in range(replicas)
                ):
                    exists.add((g_idx, chunk_hashes[chunk_idx]))

        if not exists:
            return 0
        pool = ExternalCachedBlockPool(exists)
        _, hit_length = self.coord.find_longest_cache_hit(
            block_hashes, token_len, pool
        )
        return hit_length

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
