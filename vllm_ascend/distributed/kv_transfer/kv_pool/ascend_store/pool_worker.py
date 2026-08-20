from __future__ import annotations

import importlib
import math
import os
import threading
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor

import torch
import vllm.envs as envs
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
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerMultiBlockReqMeta,
    ReqMeta,
    get_block_hashes,
    get_cache_family_granularity,
    infer_group_cache_families,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import (
    AscendStoreCoordinator,
    ExternalCachedBlockPool,
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
        kv_cache_config: KVCacheConfig | None = None,
    ):
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.kv_cache_config = kv_cache_config
        hf_text_config = getattr(model_config, "hf_text_config", None)
        hf_config = getattr(model_config, "hf_config", hf_text_config)
        self.hf_config = hf_text_config or hf_config
        self.compress_ratios = getattr(hf_text_config, "compress_ratios", None)
        if self.compress_ratios is None:
            self.compress_ratios = getattr(hf_config, "compress_ratios", None)
        self.use_compress = self.compress_ratios is not None
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
        self.use_hybrid = self._uses_hybrid_kv_cache(vllm_config, kv_cache_config)
        self.original_block_size = self._infer_group_block_sizes(vllm_config, kv_cache_config)
        cp_scale = self.pcp_size * self.dcp_size
        self.grouped_block_size = [block_size * cp_scale for block_size in self.original_block_size]
        requested_hash_block_size = vllm_config.cache_config.hash_block_size
        if not isinstance(requested_hash_block_size, int):
            requested_hash_block_size = None
        self.hash_block_size = (
            requested_hash_block_size if requested_hash_block_size is not None else min(self.original_block_size)
        ) * cp_scale
        for group_block_size in self.grouped_block_size:
            assert group_block_size % self.hash_block_size == 0, "block_size must be divisible by hash_block_size"
        self.block_size = self.grouped_block_size[0]
        self.lcm_block_size = math.lcm(*self.grouped_block_size)
        self.num_kv_cache_groups = len(self.grouped_block_size)
        self.kv_cache_group_families = self._infer_group_families()
        self.group_uses_align_state = self._infer_group_uses_align_state()
        self.cache_transfer_granularity = self._infer_cache_transfer_granularity()
        if self.use_layerwise and self.num_kv_cache_groups > 1:
            raise NotImplementedError("AscendStore layerwise mode does not yet support hybrid KV cache groups.")

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

        spec_cfg = getattr(vllm_config, "speculative_config", None)
        use_eagle = bool(
            spec_cfg.use_eagle() if spec_cfg is not None and callable(getattr(spec_cfg, "use_eagle", None)) else False
        )
        kv_cache_groups = (
            list(kv_cache_config.kv_cache_groups) if kv_cache_config is not None and self.use_hybrid else None
        )
        self.token_database = ChunkedTokenDatabase(
            self.metadata,
            self.grouped_block_size,
            partitions,
            use_hybrid=self.use_hybrid,
            hash_block_size=self.hash_block_size,
            kv_cache_groups=kv_cache_groups,
            alignment_tokens=self.cache_transfer_granularity,
            retention_interval=getattr(envs, "VLLM_PREFIX_CACHE_RETENTION_INTERVAL", None),
            use_eagle=use_eagle,
        )
        self.cache_coordinator = self._build_cache_coordinator(vllm_config)
        self.token_database.set_cache_coordinator(self.cache_coordinator)

        backend = backend_map.get(self.backend.lower())
        assert backend is not None
        backend_path = backend.get("path")
        backend_name = backend.get("name")
        assert backend_path is not None and backend_name is not None
        backend_module = importlib.import_module(backend_path)
        real_backend = getattr(backend_module, backend_name)

        backend_kwargs = {}
        if self.backend.lower() in {"mooncake", "memcache"}:
            # DSV4 exposes compress_ratios; only use lazy store init for this
            # compressed-model path.
            backend_kwargs["lazy_init"] = self.use_compress
        self.m_store = real_backend(  # type: ignore[misc]
            parallel_config,
            **backend_kwargs,
        )
        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = False
        if kv_event_config and kv_event_config.enable_kv_cache_events:
            self.enable_kv_events = True

        self.kv_send_thread: KVTransferThread | None = None
        self.kv_recv_thread: KVTransferThread | None = None

        self.finished_store_req: set[str] = set()

        self.lookup_group_parallel = os.getenv("VLLM_ASCEND_LOOKUP_GROUP_PARALLEL", "0") == "1"
        try:
            self.lookup_group_parallel_workers = int(
                os.getenv("VLLM_ASCEND_LOOKUP_GROUP_PARALLEL_WORKERS", "6")
            )
        except ValueError:
            self.lookup_group_parallel_workers = 6
        self.lookup_group_parallel_workers = max(1, self.lookup_group_parallel_workers)
        self.lookup_early_stop = os.getenv("VLLM_ASCEND_LOOKUP_EARLY_STOP", "0") == "1"
        self.lookup_reachable_mask = os.getenv("VLLM_ASCEND_LOOKUP_REACHABLE_MASK", "0") == "1"
        self.lookup_full_guard = os.getenv("VLLM_ASCEND_LOOKUP_FULL_GUARD", "0") == "1"

    def _build_cache_coordinator(self, vllm_config: VllmConfig) -> AscendStoreCoordinator | None:
        if self.kv_cache_config is None or not self.use_hybrid:
            return None
        speculative_config = getattr(vllm_config, "speculative_config", None)
        use_eagle_fn = getattr(speculative_config, "use_eagle", None)
        use_eagle = bool(use_eagle_fn()) if callable(use_eagle_fn) else False
        retention_interval = getattr(envs, "VLLM_PREFIX_CACHE_RETENTION_INTERVAL", None)
        if not isinstance(retention_interval, int):
            retention_interval = None
        return AscendStoreCoordinator(
            list(self.kv_cache_config.kv_cache_groups),
            scheduler_block_size=self.cache_transfer_granularity,
            hash_block_size=self.hash_block_size,
            group_block_sizes=self.grouped_block_size,
            group_cache_families=self.kv_cache_group_families,
            use_eagle=use_eagle,
            retention_interval=retention_interval,
        )

    def _infer_group_families(self) -> list[str]:
        kv_cache_groups = self.kv_cache_config.kv_cache_groups if self.kv_cache_config is not None else None
        return infer_group_cache_families(kv_cache_groups, self.compress_ratios, self.hf_config)

    def _infer_group_block_sizes(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig | None,
    ) -> list[int]:
        if kv_cache_config is None or not self.use_hybrid:
            return [vllm_config.cache_config.block_size]

        block_sizes: list[int] = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            block_sizes.append(kv_cache_spec.block_size)
        return block_sizes

    def _infer_group_uses_align_state(self) -> list[bool]:
        if self.kv_cache_config is None:
            return [False]

        group_uses_align_state: list[bool] = []
        for group in self.kv_cache_config.kv_cache_groups:
            kv_cache_spec = group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                specs = [kv_cache_spec.kv_cache_specs[layer_name] for layer_name in group.layer_names]
            else:
                specs = [kv_cache_spec]
            group_uses_align_state.append(
                any(
                    isinstance(spec, MambaSpec) and getattr(spec, "mamba_cache_mode", None) == "align" for spec in specs
                )
            )
        return group_uses_align_state

    def _get_group_block_size(self, group_id: int) -> int:
        if group_id >= len(self.grouped_block_size):
            return self.grouped_block_size[0]
        return self.grouped_block_size[group_id]

    @staticmethod
    def _get_group_family(families: list[str], group_id: int) -> str:
        if group_id >= len(families):
            return "default"
        return families[group_id]

    def _infer_cache_transfer_granularity(self) -> int:
        granularities = [self.lcm_block_size]
        for group_id in range(self.num_kv_cache_groups):
            granularities.append(
                get_cache_family_granularity(
                    self._get_group_block_size(group_id),
                    self._get_group_family(self.kv_cache_group_families, group_id),
                )
            )
        return math.lcm(*granularities)

    @staticmethod
    def _uses_hybrid_kv_cache(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None) -> bool:
        if kv_cache_config is None:
            return False
        if getattr(vllm_config.scheduler_config, "disable_hybrid_kv_cache_manager", False):
            return False
        return len(kv_cache_config.kv_cache_groups) > 1 and any(
            not isinstance(group.kv_cache_spec, FullAttentionSpec) for group in kv_cache_config.kv_cache_groups
        )

    @staticmethod
    def _as_cache_tuple(cache_or_caches) -> tuple[torch.Tensor, ...]:
        if isinstance(cache_or_caches, torch.Tensor):
            return (cache_or_caches,)
        return tuple(cache_or_caches)

    def _get_cache_block_metadata(self, cache: torch.Tensor) -> tuple[int, int, int, int]:
        tensor_num_blocks = cache.shape[0]
        assert tensor_num_blocks % self.num_blocks == 0, (
            "The external block size must be an integer multiple of the kernel block size."
        )
        block_size_scale = tensor_num_blocks // self.num_blocks
        block_len = cache[0].numel() * cache.element_size() * block_size_scale
        block_stride = cache.stride(0) * cache.element_size() * block_size_scale
        region_len = (self.num_blocks - 1) * block_stride + block_len if self.num_blocks else 0
        return block_len, block_stride, region_len, block_size_scale

    @staticmethod
    def _get_storage_key(cache: torch.Tensor) -> int:
        try:
            return cache.untyped_storage().data_ptr()
        except AttributeError:
            return cache.storage().data_ptr()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache_tuple = self._as_cache_tuple(first_kv_cache_tuple)
        first_kv_cache = first_kv_cache_tuple[0]

        self.num_blocks = (
            self.kv_cache_config.num_blocks if self.kv_cache_config is not None else first_kv_cache.shape[0]
        )
        logger.info("num_blocks: %s", self.num_blocks)
        self.block_len = []
        self.block_stride = []
        for cache in first_kv_cache_tuple:
            block_len, block_stride, _, _ = self._get_cache_block_metadata(cache)
            logger.info("block_shape: %s", cache.shape[1:])
            self.block_len.append(block_len)
            self.block_stride.append(block_stride)

        logger.info(
            "Registering KV_Caches. use_mla: %s, use_sparse: %s, shape %s",
            self.use_mla,
            self.use_sparse,
            first_kv_cache.shape,
        )

        self.kv_caches = kv_caches
        self.kv_caches_base_addr = []
        self.group_kv_caches_base_addr: dict[int, list[int]] = {}
        self.group_block_len: dict[int, list[int]] = {}
        self.group_block_stride: dict[int, list[int]] = {}
        self.group_kv_cache_families: dict[int, str] = {
            group_id: self._get_group_family(self.kv_cache_group_families, group_id)
            for group_id in range(self.num_kv_cache_groups)
        }
        self.group_num_layers: dict[int, int] = {}
        registered_regions: dict[int, tuple[int, int]] = {}
        for cache_or_caches in kv_caches.values():
            for cache in self._as_cache_tuple(cache_or_caches):
                base_addr = cache.data_ptr()
                _, _, region_len, _ = self._get_cache_block_metadata(cache)
                if not isinstance(region_len, int):
                    region_len = 0
                self.kv_caches_base_addr.append(base_addr)
                storage_key = self._get_storage_key(cache)
                start = base_addr
                end = base_addr + region_len
                if storage_key in registered_regions:
                    old_start, old_end = registered_regions[storage_key]
                    registered_regions[storage_key] = (min(old_start, start), max(old_end, end))
                else:
                    registered_regions[storage_key] = (start, end)

        ptrs = [start for start, _ in registered_regions.values()]
        lengths = [end - start for start, end in registered_regions.values()]

        if self.kv_cache_config is not None and self.use_hybrid:
            for group_id, group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
                group_addrs: list[int] = []
                group_block_lens: list[int] = []
                group_block_strides: list[int] = []
                seen_group_ptrs: set[int] = set()
                for layer_name in group_spec.layer_names:
                    cache_or_caches = kv_caches[layer_name]
                    for cache in self._as_cache_tuple(cache_or_caches):
                        base_addr = cache.data_ptr()
                        if base_addr in seen_group_ptrs:
                            continue
                        block_len, block_stride, _, _ = self._get_cache_block_metadata(cache)
                        group_addrs.append(base_addr)
                        group_block_lens.append(block_len)
                        group_block_strides.append(block_stride)
                        seen_group_ptrs.add(base_addr)
                self.group_kv_caches_base_addr[group_id] = group_addrs
                self.group_block_len[group_id] = group_block_lens
                self.group_block_stride[group_id] = group_block_strides
                self.group_num_layers[group_id] = len(group_spec.layer_names)

        self.m_store.register_buffer(ptrs, lengths)
        self.token_database.set_kv_caches_base_addr(self.kv_caches_base_addr)
        self.token_database.set_block_len(self.block_len)
        self.token_database.set_block_stride(self.block_stride)
        self.token_database.set_group_buffers(
            self.group_kv_caches_base_addr,
            self.group_block_len,
            self.group_block_stride,
            cache_role="kv",
            group_cache_families=self.group_kv_cache_families,
            group_num_layers=self.group_num_layers,
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
                )
                self.kv_send_thread.start()
            if self.load_async:
                ready_event = threading.Event()
                self.kv_recv_thread = KVCacheStoreRecvingThread(
                    self.m_store, self.token_database, self.block_size, self.tp_rank, self.dcp_size, ready_event
                )
                self.kv_recv_thread.start()
                ready_event.wait()

    def start_load_kv(self, metadata: AscendConnectorMetadata):
        self.current_layer = 0
        self.layerwise_retrievers = []
        logger.debug("KV pool worker start_load_kv requests=%d", len(metadata.requests))
        for request in metadata.requests:
            load_spec = request.load_spec
            if load_spec is None or not load_spec.can_load:  # load =0
                logger.debug(
                    "KV pool worker skip get req=%s reason=%s",
                    request.req_id,
                    "no_load_spec" if load_spec is None else f"can_load={load_spec.can_load}",
                )
                continue
            request.skip_null_blocks_by_group = self.group_uses_align_state
            load_group_ids = request.kv_cache_group_ids or [0]
            token_len = request.token_len_chunk
            if (load_spec.kvpool_cached_tokens % self.cache_transfer_granularity != 0) and (
                load_spec.kvpool_cached_tokens == token_len - 1
            ):
                token_len = request.load_spec.kvpool_cached_tokens + 1
            else:
                token_len = request.load_spec.kvpool_cached_tokens
            request.load_spec.token_len = token_len
            logger.debug(
                "KV pool worker prepare get req=%s token_len_chunk=%d get_token_len=%d "
                "vllm_cached=%d kvpool_cached=%d groups=%s load_async=%s",
                request.req_id,
                request.token_len_chunk,
                token_len,
                load_spec.vllm_cached_tokens,
                load_spec.kvpool_cached_tokens,
                load_group_ids,
                self.load_async,
            )
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
                    load_masks = self.token_database.load_mask(request.block_hashes, token_len)
                    for group_id in load_group_ids:
                        block_ids = request.block_ids_by_group[group_id]
                        group_block_size = self.grouped_block_size[group_id]
                        mask_num = request.load_spec.vllm_cached_tokens // group_block_size * group_block_size
                        skip_null = (
                            group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]
                        )
                        for start, end, key, block_id in self.token_database.process_tokens_with_block_ids(
                            token_len,
                            request.block_hashes,
                            block_ids,
                            mask_num,
                            kv_cache_group_id=group_id,
                            skip_null_blocks=skip_null,
                        ):
                            if not self.token_database.mask_allows_chunk(load_masks, group_id, start):
                                continue
                            addr, size, _ = self.token_database.prepare_value(
                                start,
                                end,
                                block_ids,
                                kv_cache_group_id=group_id,
                                block_id=block_id,
                            )
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
                    import time as _kvtrace_time
                    logger.info(
                        "KVTRACE req=%s stage=kvpool_get_prepare elapsed_ms=0.000 "
                        "token_len=%d vllm_cached=%d kvpool_cached=%d keys=%d groups=%s",
                        request.req_id, token_len,
                        load_spec.vllm_cached_tokens, load_spec.kvpool_cached_tokens,
                        len(key_list_c), load_group_ids
                    )
                    _kvtrace_t0 = _kvtrace_time.perf_counter()
                    self.m_store.get(key_list_c, addr_list_c, size_list_c)
                    _kvtrace_get_ms = (_kvtrace_time.perf_counter() - _kvtrace_t0) * 1000
                    logger.info(
                        "KVTRACE req=%s stage=kvpool_get_backend elapsed_ms=%.3f "
                        "mode=sync keys=%d",
                        request.req_id, _kvtrace_get_ms, len(key_list_c)
                    )

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
        has_save_request = False
        for request in connector_metadata.requests:
            can_save = request.can_save
            if can_save is None or not can_save:
                continue
            current_event = torch.npu.Event()
            current_event.record()
            break

        _kvtrace_wait_events = []
        _kvtrace_per_request_wait = getattr(
            self.kv_send_thread, "_per_request_save_wait", False  # type: ignore[union-attr]
        )
        for request in connector_metadata.requests:
            can_save = request.can_save
            if can_save is None or not can_save:
                continue

            request.skip_null_blocks_by_group = self.group_uses_align_state
            request.current_event = current_event
            self.kv_send_thread.add_stored_request(  # type: ignore[union-attr]
                request.req_id
            )
            if _kvtrace_per_request_wait:
                event = self.kv_send_thread.prepare_stored_request_done_event(  # type: ignore[union-attr]
                    request.req_id
                )
                if event is not None:
                    _kvtrace_wait_events.append(event)
            self.kv_send_thread.add_request(  # type: ignore[union-attr]
                request,
            )
            has_save_request = True

        if has_save_request:
            # vLLM expects wait_for_save() to make stores visible before the
            # request is reported as finished. Without this barrier a following
            # identical prompt can lookup before Mooncake put() has completed.
            import time as _kvtrace_time
            _kvtrace_save_count = sum(1 for r in connector_metadata.requests if r.can_save)
            _kvtrace_qsize_before = self.kv_send_thread.request_queue.qsize()  # type: ignore[union-attr]
            _kvtrace_t0 = _kvtrace_time.perf_counter()
            if _kvtrace_per_request_wait:
                for event in _kvtrace_wait_events:
                    event.wait()
                _kvtrace_join_ms = (_kvtrace_time.perf_counter() - _kvtrace_t0) * 1000
                _kvtrace_qsize_after = self.kv_send_thread.request_queue.qsize()  # type: ignore[union-attr]
                logger.info(
                    "KVTRACE stage=kvpool_wait_for_save_request elapsed_ms=%.3f "
                    "save_requests=%d queue_size_before=%d queue_size_after=%d events=%d",
                    _kvtrace_join_ms, _kvtrace_save_count,
                    _kvtrace_qsize_before, _kvtrace_qsize_after,
                    len(_kvtrace_wait_events)
                )
            else:
                self.kv_send_thread.request_queue.join()  # type: ignore[union-attr]
                _kvtrace_join_ms = (_kvtrace_time.perf_counter() - _kvtrace_t0) * 1000
                _kvtrace_qsize_after = self.kv_send_thread.request_queue.qsize()  # type: ignore[union-attr]
                logger.info(
                    "KVTRACE stage=kvpool_wait_for_save_join elapsed_ms=%.3f "
                    "save_requests=%d queue_size_before=%d queue_size_after=%d",
                    _kvtrace_join_ms, _kvtrace_save_count,
                    _kvtrace_qsize_before, _kvtrace_qsize_after
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
                    request.req_id, keys_multi_chunk, starts, ends, request.block_ids_by_group, layer_id
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
        logger.debug(
            "Retrieved %s out of %s out of total %s tokens",
            retrieved_tokens,
            num_required_tokens,
            token_len,
        )

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
                    request.block_ids_by_group,
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

        logger.debug(
            "Number of completed KV cache send requests: %d, receive requests: %d, tp_rank:%d",
            len(done_sending),
            len(done_recving),
            self.tp_rank,
        )
        return done_sending, done_recving

    def ensure_store_initialized(self) -> None:
        ensure_initialized = getattr(self.m_store, "ensure_initialized", None)
        if ensure_initialized is not None:
            ensure_initialized()

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
        kv_cache_group_ids: list[int] | None = None,
        use_layerwise: bool = False,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.
        :param tokens: the input tokens, with shape [seq_len]
        :return: An int indicating how many prefix tokens are cached.
        """
        try:
            hits = []
            kv_cache_group_ids = kv_cache_group_ids or [0]
            coordinator_hit = self._lookup_with_coordinator(
                token_len,
                block_hashes,
                kv_cache_group_ids,
                use_layerwise,
                include_all_ranks=False,
            )
            if coordinator_hit is not None:
                return coordinator_hit
            for group_id in kv_cache_group_ids:
                end = 0
                keys = []
                starts = []
                ends = []
                for start, end, key in self.token_database.process_tokens(
                    token_len,
                    block_hashes,
                    kv_cache_group_id=group_id,
                ):
                    if use_layerwise:
                        keys_multi_layer = key.split_layers(self.num_layers)
                        for item in keys_multi_layer:
                            keys.append(item.to_string())
                    else:
                        keys.append(key.to_string())
                    starts.append(start)
                    ends.append(end)

                if not keys:
                    hits.append(0)
                    continue

                res = self.m_store.exists(keys)  # type: ignore[assignment]

                if use_layerwise:
                    res = self.check_all_layers_exists(res, self.num_layers)
                if group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]:
                    hit_end = 0
                    for index in range(len(ends) - 1, -1, -1):
                        if (
                            res[index] == 1  # type: ignore[index]
                            and ends[index] % self.cache_transfer_granularity == 0
                        ):
                            hit_end = ends[index]
                            break
                else:
                    hit_end = end
                    for index, value in enumerate(res):  # type: ignore[arg-type]
                        if value != 1:
                            hit_end = 0
                            for hit_index in range(index, 0, -1):
                                if starts[hit_index] % self.cache_transfer_granularity == 0:
                                    hit_end = starts[hit_index]
                                    break
                            break
                hits.append(hit_end)
        except Exception as e:
            logger.error("Remote connection failed in contains: %s", e)
            return 0
        return min(hits) if hits else 0

    def _get_group_num_kv_heads(self, group_id: int) -> int:
        if self.use_mla or self.use_sparse:
            return 1
        if group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]:
            return 1
        return self.num_kv_head

    def get_group_tp_size(self, kv_cache_group_id: int) -> int:
        if kv_cache_group_id < len(self.group_uses_align_state) and self.group_uses_align_state[kv_cache_group_id]:
            return self.tp_size
        return min(self.tp_size, self._get_group_num_kv_heads(kv_cache_group_id))

    @staticmethod
    def _replace_key_field(key: str, field: str, value: int) -> str:
        marker = f"@{field}:"
        start = key.find(marker)
        if start < 0:
            return key
        value_start = start + len(marker)
        value_end = key.find("@", value_start)
        if value_end < 0:
            value_end = len(key)
        return f"{key[:value_start]}{value}{key[value_end:]}"

    @staticmethod
    def _chunk_hash_to_bytes(chunk_hash: BlockHash | str) -> bytes:
        if isinstance(chunk_hash, str):
            if len(chunk_hash) == 64:
                try:
                    return bytes.fromhex(chunk_hash)
                except ValueError:
                    pass
            return chunk_hash.encode("utf-8")
        return bytes(chunk_hash)

    def _expand_lookup_key_variants(self, key: str, group_id: int, include_all_ranks: bool) -> list[str]:
        if not include_all_ranks:
            return [key]
        variants: list[str] = []
        group_tp_size = self.get_group_tp_size(group_id)
        for tp_rank in range(group_tp_size):
            tp_key = self._replace_key_field(key, "head_or_tp_rank", tp_rank)
            for pp_rank in range(self.pp_size):
                variants.append(self._replace_key_field(tp_key, "pp_rank", pp_rank))
        return variants

    def _lookup_coordinator_group(
        self,
        group_id: int,
        token_len: int,
        block_hashes: list[BlockHash],
        include_all_ranks: bool,
        hbm_hit_tokens: int = 0,
    ) -> tuple[int, set[tuple[int, bytes]], float, float, float]:
        """Build/exist one KV cache group for coordinator lookup.

        The return value is detached from shared state so callers can run
        groups in parallel without changing lookup semantics.
        """
        import time as _kvtrace_time

        group_exists: set[tuple[int, bytes]] = set()
        _kvtrace_t_kb = _kvtrace_time.perf_counter()
        keys: list[str] = []
        chunk_hashes: list[BlockHash | str] = []
        variant_counts: list[int] = []
        _kvtrace_app_ms = 0.0
        _kvtrace_raw_count = 0
        base_bs = self.token_database.get_block_size(group_id)
        cf = self.token_database.group_cache_families.get("kv", {}).get(group_id, "default")
        ebs = get_cache_family_granularity(base_bs, cf)
        lookup_start = hbm_hit_tokens // ebs * ebs
        for _, _, key_string, chunk_hash in self.token_database.process_token_key_strings(
            token_len,
            block_hashes,
            mask_num=lookup_start,
            kv_cache_group_id=group_id,
        ):
            _kvtrace_raw_count += 1
            _kvtrace_t_a = _kvtrace_time.perf_counter()
            variants = self._expand_lookup_key_variants(key_string, group_id, include_all_ranks)
            keys.extend(variants)
            chunk_hashes.append(chunk_hash)
            variant_counts.append(len(variants))
            _kvtrace_app_ms += (_kvtrace_time.perf_counter() - _kvtrace_t_a) * 1000
        _kvtrace_iter_ms = (_kvtrace_time.perf_counter() - _kvtrace_t_kb) * 1000 - _kvtrace_app_ms
        _kvtrace_str_ms = 0.0

        _kvtrace_kb_ms = (_kvtrace_time.perf_counter() - _kvtrace_t_kb) * 1000
        _kvtrace_raw_keys = len(chunk_hashes)
        token_chunks = (token_len + ebs - 1) // ebs if ebs > 0 else 0
        skipped_chunks = min(token_chunks, lookup_start // ebs) if ebs > 0 else 0
        expected_lookup_chunks = max(0, token_chunks - skipped_chunks)
        logger.info(
            "KVTRACE stage=kvpool_lookup_group_key_build group=%d "
            "elapsed_ms=%.3f raw_keys=%d variant_keys=%d token_len=%d "
            "hbm_hit_tokens=%d lookup_start=%d effective_block_size=%d "
            "token_chunks=%d skipped_chunks=%d expected_lookup_chunks=%d",
            group_id, _kvtrace_kb_ms, _kvtrace_raw_keys, len(keys), token_len,
            hbm_hit_tokens, lookup_start, ebs,
            token_chunks, skipped_chunks, expected_lookup_chunks
        )
        logger.info(
            "KVTRACE stage=kvpool_6d_group_mask group=%d "
            "hbm_hit_tokens=%d lookup_start=%d effective_block_size=%d "
            "skipped_chunks=%d expected_lookup_chunks=%d built_raw_keys=%d built_variant_keys=%d",
            group_id, hbm_hit_tokens, lookup_start, ebs,
            skipped_chunks, expected_lookup_chunks, _kvtrace_raw_keys, len(keys)
        )
        logger.info(
            "KVTRACE stage=kvpool_kb_detail group=%d "
            "raw_count=%d iter_ms=%.3f str_ms=%.3f app_ms=%.3f total_ms=%.3f "
            "compress_ratios=%s grouped_block_size=%s",
            group_id, _kvtrace_raw_count,
            _kvtrace_iter_ms, _kvtrace_str_ms, _kvtrace_app_ms, _kvtrace_kb_ms,
            getattr(self, "compress_ratios", None),
            self.grouped_block_size[group_id] if hasattr(self, "grouped_block_size") else "N/A"
        )

        if not keys:
            logger.info(
                "KVTRACE stage=kvpool_6d_group_skip_external group=%d reason=no_keys "
                "hbm_hit_tokens=%d lookup_start=%d effective_block_size=%d "
                "token_chunks=%d skipped_chunks=%d expected_lookup_chunks=%d",
                group_id, hbm_hit_tokens, lookup_start, ebs,
                token_chunks, skipped_chunks, expected_lookup_chunks
            )
            return group_id, group_exists, _kvtrace_kb_ms, 0.0, 0.0

        _kvtrace_t0 = _kvtrace_time.perf_counter()
        res = self.m_store.exists(keys)  # type: ignore[assignment]
        _kvtrace_exist_ms = (_kvtrace_time.perf_counter() - _kvtrace_t0) * 1000
        _kvtrace_exists_count = sum(1 for v in res if v == 1) if res else 0
        logger.info(
            "KVTRACE stage=kvpool_exist_backend elapsed_ms=%.3f "
            "group=%d keys=%d exists_count=%d missing_count=%d",
            _kvtrace_exist_ms, group_id, len(keys),
            _kvtrace_exists_count, len(keys) - _kvtrace_exists_count
        )
        _kvtrace_t_hc = _kvtrace_time.perf_counter()
        offset = 0
        for chunk_hash, count in zip(chunk_hashes, variant_counts, strict=True):
            values = res[offset : offset + count]  # type: ignore[index]
            if values and all(value == 1 for value in values):
                group_exists.add((group_id, self._chunk_hash_to_bytes(chunk_hash)))
            offset += count

        _kvtrace_hc_ms = (_kvtrace_time.perf_counter() - _kvtrace_t_hc) * 1000
        logger.info(
            "KVTRACE stage=kvpool_lookup_group_hit_calc group=%d "
            "elapsed_ms=%.3f hit_chunks=%d/%d token_len=%d",
            group_id, _kvtrace_hc_ms,
            len(group_exists), len(chunk_hashes), token_len
        )
        logger.debug(
            "KV pool coordinator lookup group=%d token_len=%d keys=%d exists_chunks=%d/%d sample_keys=%s",
            group_id,
            token_len,
            len(keys),
            len(group_exists),
            len(chunk_hashes),
            keys[:3],
        )
        return group_id, group_exists, _kvtrace_kb_ms, _kvtrace_exist_ms, _kvtrace_hc_ms

    def _lookup_with_coordinator(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int],
        use_layerwise: bool,
        include_all_ranks: bool,
        hbm_hit_tokens: int = 0,
    ) -> int | None:
        if self.cache_coordinator is None or use_layerwise:
            return None
        if sorted(kv_cache_group_ids) != list(range(self.num_kv_cache_groups)):
            return None

        import time as _kvtrace_time
        _kvtrace_total_keybuild = 0.0
        _kvtrace_total_exist = 0.0
        _kvtrace_total_hitcalc = 0.0
        exists: set[tuple[int, bytes]] = set()
        # 6D: Add HBM-hit blocks to exists without RPC (aligned per group)
        logger.info(
            "KVTRACE stage=kvpool_6d_lookup_start token_len=%d hbm_hit_tokens=%d "
            "groups=%s use_layerwise=%s include_all_ranks=%s block_hashes=%d",
            token_len, hbm_hit_tokens, kv_cache_group_ids,
            use_layerwise, include_all_ranks, len(block_hashes)
        )
        if hbm_hit_tokens > 0:
            for group_id in kv_cache_group_ids:
                base_bs = self.token_database.get_block_size(group_id)
                cf = self.token_database.group_cache_families.get("kv", {}).get(group_id, "default")
                ebs = get_cache_family_granularity(base_bs, cf)
                lookup_start = hbm_hit_tokens // ebs * ebs
                grouped = get_block_hashes(block_hashes, ebs, self.token_database.hash_block_size)
                injected = 0
                if grouped:
                    num_hbm_chunks = lookup_start // ebs
                    for cid in range(min(num_hbm_chunks, len(grouped))):
                        exists.add((group_id, self._chunk_hash_to_bytes(grouped[cid])))
                        injected += 1
                token_chunks = (token_len + ebs - 1) // ebs if ebs > 0 else 0
                skipped_chunks = min(token_chunks, lookup_start // ebs) if ebs > 0 else 0
                logger.info(
                    "KVTRACE stage=kvpool_6d_hbm_inject group=%d "
                    "base_block_size=%d cache_family=%s effective_block_size=%d "
                    "hbm_hit_tokens=%d lookup_start=%d token_chunks=%d grouped_hashes=%d "
                    "skipped_chunks=%d injected_chunks=%d exists_size=%d",
                    group_id, base_bs, cf, ebs, hbm_hit_tokens, lookup_start,
                    token_chunks, len(grouped) if grouped else 0,
                    skipped_chunks, injected, len(exists)
                )
        lookup_masks = None
        if self.lookup_reachable_mask:
            aligned_lookup_len = token_len
            lcm_block_size = getattr(self.cache_coordinator, "lcm_block_size", 1)
            if lcm_block_size > 1:
                aligned_lookup_len = ((token_len + lcm_block_size - 1) // lcm_block_size) * lcm_block_size
            lookup_masks = self.cache_coordinator.lookup_mask(aligned_lookup_len)
            logger.info(
                "KVTRACE stage=kvpool_lookup_mask_start token_len=%d aligned_token_len=%d "
                "lcm_block_size=%d mask_kinds=%s",
                token_len, aligned_lookup_len, lcm_block_size,
                [
                    "all" if mask is None else f"{sum(1 for item in mask if item)}/{len(mask)}"
                    for mask in lookup_masks
                ],
            )

        lookup_limit = token_len
        full_guard_group_ids: set[int] = set()
        if self.lookup_full_guard and not self.lookup_group_parallel:
            for spec, group_ids, _ in getattr(self.cache_coordinator, "attention_groups", []):
                if isinstance(spec, FullAttentionSpec):
                    full_guard_group_ids.update(
                        group_id for group_id in group_ids if group_id in kv_cache_group_ids
                    )
            if full_guard_group_ids:
                _kvtrace_full_bound = token_len
                _kvtrace_full_keybuild = 0.0
                _kvtrace_full_exist = 0.0
                _kvtrace_full_hitcalc = 0.0
                _kvtrace_full_raw = 0
                _kvtrace_full_variant = 0
                first_missing_group = -1
                first_missing_start_global = -1
                for group_id in sorted(full_guard_group_ids):
                    _kvtrace_t_kb = _kvtrace_time.perf_counter()
                    keys: list[str] = []
                    chunk_hashes: list[BlockHash | str] = []
                    variant_counts: list[int] = []
                    starts: list[int] = []
                    _kvtrace_app_ms = 0.0
                    base_bs = self.token_database.get_block_size(group_id)
                    cf = self.token_database.group_cache_families.get("kv", {}).get(group_id, "default")
                    ebs = get_cache_family_granularity(base_bs, cf)
                    lookup_start = hbm_hit_tokens // ebs * ebs
                    lookup_mask = None
                    if lookup_masks is not None and group_id < len(lookup_masks):
                        lookup_mask = lookup_masks[group_id]
                    skipped_by_mask = 0
                    seen_chunks = 0
                    for start_idx, _, key_string, chunk_hash in self.token_database.process_token_key_strings(
                        token_len,
                        block_hashes,
                        mask_num=lookup_start,
                        kv_cache_group_id=group_id,
                    ):
                        seen_chunks += 1
                        if lookup_mask is not None:
                            chunk_idx = start_idx // base_bs if base_bs > 0 else 0
                            if chunk_idx >= len(lookup_mask) or not lookup_mask[chunk_idx]:
                                skipped_by_mask += 1
                                continue
                        _kvtrace_t_a = _kvtrace_time.perf_counter()
                        variants = self._expand_lookup_key_variants(key_string, group_id, include_all_ranks)
                        keys.extend(variants)
                        chunk_hashes.append(chunk_hash)
                        variant_counts.append(len(variants))
                        starts.append(start_idx)
                        _kvtrace_app_ms += (_kvtrace_time.perf_counter() - _kvtrace_t_a) * 1000
                    _kvtrace_kb_ms = (_kvtrace_time.perf_counter() - _kvtrace_t_kb) * 1000
                    _kvtrace_full_keybuild += _kvtrace_kb_ms
                    _kvtrace_full_raw += len(chunk_hashes)
                    _kvtrace_full_variant += len(keys)
                    group_exists: set[tuple[int, bytes]] = set()
                    if keys:
                        _kvtrace_t0 = _kvtrace_time.perf_counter()
                        res = self.m_store.exists(keys)  # type: ignore[assignment]
                        _kvtrace_exist_ms = (_kvtrace_time.perf_counter() - _kvtrace_t0) * 1000
                        _kvtrace_full_exist += _kvtrace_exist_ms
                        _kvtrace_t_hc = _kvtrace_time.perf_counter()
                        offset = 0
                        first_missing_start = -1
                        for start, chunk_hash, count in zip(starts, chunk_hashes, variant_counts, strict=True):
                            values = res[offset : offset + count]  # type: ignore[index]
                            if values and all(value == 1 for value in values):
                                item = (group_id, self._chunk_hash_to_bytes(chunk_hash))
                                exists.add(item)
                                group_exists.add(item)
                            elif first_missing_start < 0:
                                first_missing_start = start
                            offset += count
                        _kvtrace_hc_ms = (_kvtrace_time.perf_counter() - _kvtrace_t_hc) * 1000
                        _kvtrace_full_hitcalc += _kvtrace_hc_ms
                        if first_missing_start < 0:
                            group_bound = token_len
                        else:
                            group_bound = (first_missing_start // self.cache_transfer_granularity) * self.cache_transfer_granularity
                            group_bound = max(group_bound, min(max(hbm_hit_tokens, 0), token_len))
                        if first_missing_start >= 0 and (
                            first_missing_group < 0 or group_bound < _kvtrace_full_bound
                        ):
                            first_missing_group = group_id
                            first_missing_start_global = first_missing_start
                        logger.info(
                            "KVTRACE stage=kvpool_lookup_full_guard_group group=%d "
                            "key_build_ms=%.3f exist_ms=%.3f hit_calc_ms=%.3f "
                            "raw_keys=%d variant_keys=%d exists_count=%d missing_count=%d "
                            "seen_chunks=%d skipped_by_mask=%d lookup_start=%d bound=%d token_len=%d "
                            "first_missing_start=%d",
                            group_id, _kvtrace_kb_ms, _kvtrace_exist_ms, _kvtrace_hc_ms,
                            len(chunk_hashes), len(keys), len(group_exists),
                            len(chunk_hashes) - len(group_exists), seen_chunks, skipped_by_mask,
                            lookup_start, group_bound, token_len, first_missing_start,
                        )
                    else:
                        group_bound = token_len
                        logger.info(
                            "KVTRACE stage=kvpool_lookup_full_guard_group group=%d "
                            "key_build_ms=%.3f exist_ms=0.000 hit_calc_ms=0.000 "
                            "raw_keys=0 variant_keys=0 exists_count=0 missing_count=0 "
                            "seen_chunks=%d skipped_by_mask=%d lookup_start=%d bound=%d token_len=%d",
                            group_id, _kvtrace_kb_ms, seen_chunks, skipped_by_mask,
                            lookup_start, group_bound, token_len,
                        )
                    _kvtrace_full_bound = min(_kvtrace_full_bound, group_bound)

                lookup_limit = min(token_len, _kvtrace_full_bound)
                _kvtrace_total_keybuild += _kvtrace_full_keybuild
                _kvtrace_total_exist += _kvtrace_full_exist
                _kvtrace_total_hitcalc += _kvtrace_full_hitcalc
                logger.info(
                    "KVTRACE stage=kvpool_lookup_full_guard_summary groups=%s "
                    "lookup_limit=%d token_len=%d hbm_hit_tokens=%d raw_keys=%d "
                    "variant_keys=%d key_build_ms=%.3f exist_ms=%.3f hit_calc_ms=%.3f",
                    sorted(full_guard_group_ids), lookup_limit, token_len, hbm_hit_tokens,
                    _kvtrace_full_raw, _kvtrace_full_variant, _kvtrace_full_keybuild,
                    _kvtrace_full_exist, _kvtrace_full_hitcalc,
                )
                if lookup_limit <= min(max(hbm_hit_tokens, 0), token_len):
                    hit_length = min(max(hbm_hit_tokens, 0), token_len)
                    total_ms = _kvtrace_total_keybuild + _kvtrace_total_exist + _kvtrace_total_hitcalc
                    logger.info(
                        "KVTRACE stage=kvpool_lookup_full_guard_stop reason=full_attention_bound "
                        "hit_tokens=%d lookup_limit=%d hbm_hit_tokens=%d token_len=%d "
                        "first_missing_group=%d first_missing_start=%d total_ms=%.3f",
                        hit_length, lookup_limit, hbm_hit_tokens, token_len,
                        first_missing_group, first_missing_start_global, total_ms,
                    )
                    logger.info(
                        "KVTRACE stage=kvpool_lookup_breakdown token_len=%d "
                        "key_build_ms=%.3f exist_ms=%.3f hit_calc_ms=%.3f "
                        "coord_hit_ms=0.000 total_ms=%.3f hit_tokens=%d groups=%s full_guard=True stopped=True",
                        token_len, _kvtrace_total_keybuild, _kvtrace_total_exist,
                        _kvtrace_total_hitcalc, total_ms, hit_length, kv_cache_group_ids,
                    )
                    return hit_length

        if self.lookup_group_parallel:
            _kvtrace_t_parallel = _kvtrace_time.perf_counter()
            max_workers = min(self.lookup_group_parallel_workers, len(kv_cache_group_ids))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self._lookup_coordinator_group,
                        group_id,
                        token_len,
                        block_hashes,
                        include_all_ranks,
                        hbm_hit_tokens,
                    )
                    for group_id in kv_cache_group_ids
                ]
                group_results = [future.result() for future in futures]
            for _, group_exists, keybuild_ms, exist_ms, hitcalc_ms in group_results:
                exists.update(group_exists)
                _kvtrace_total_keybuild += keybuild_ms
                _kvtrace_total_exist += exist_ms
                _kvtrace_total_hitcalc += hitcalc_ms
            _kvtrace_parallel_ms = (_kvtrace_time.perf_counter() - _kvtrace_t_parallel) * 1000
            logger.info(
                "KVTRACE stage=kvpool_lookup_group_parallel elapsed_ms=%.3f "
                "workers=%d groups=%s key_build_sum_ms=%.3f exist_sum_ms=%.3f "
                "hit_calc_sum_ms=%.3f exists_size=%d",
                _kvtrace_parallel_ms, max_workers, kv_cache_group_ids,
                _kvtrace_total_keybuild, _kvtrace_total_exist,
                _kvtrace_total_hitcalc, len(exists)
            )
            _kvtrace_t_coord = _kvtrace_time.perf_counter()
            _, hit_length = self.cache_coordinator.find_longest_cache_hit(
                block_hashes,
                token_len,
                ExternalCachedBlockPool(exists),
                apply_eagle=False,
            )
            _kvtrace_coord_ms = (_kvtrace_time.perf_counter() - _kvtrace_t_coord) * 1000
            logger.info(
                "KVTRACE stage=kvpool_lookup_breakdown token_len=%d "
                "key_build_ms=%.3f exist_ms=%.3f hit_calc_ms=%.3f "
                "coord_hit_ms=%.3f total_ms=%.3f hit_tokens=%d groups=%s parallel=True parallel_wall_ms=%.3f",
                token_len,
                _kvtrace_total_keybuild, _kvtrace_total_exist,
                _kvtrace_total_hitcalc, _kvtrace_coord_ms,
                _kvtrace_parallel_ms + _kvtrace_coord_ms,
                hit_length, kv_cache_group_ids, _kvtrace_parallel_ms
            )
            logger.debug(
                "KV pool coordinator lookup final token_len=%d groups=%s hit=%d parallel=True",
                token_len,
                kv_cache_group_ids,
                hit_length,
            )
            return hit_length

        for group_id in kv_cache_group_ids:
            if group_id in full_guard_group_ids:
                logger.info(
                    "KVTRACE stage=kvpool_lookup_group_skip_prequeried group=%d lookup_limit=%d token_len=%d",
                    group_id, lookup_limit, token_len,
                )
                continue
            _kvtrace_t_kb = _kvtrace_time.perf_counter()
            keys: list[str] = []
            chunk_hashes: list[BlockHash | str] = []
            variant_counts: list[int] = []
            _kvtrace_iter_ms = 0.0
            _kvtrace_str_ms = 0.0
            _kvtrace_app_ms = 0.0
            _kvtrace_raw_count = 0
            base_bs = self.token_database.get_block_size(group_id)
            cf = self.token_database.group_cache_families.get("kv", {}).get(group_id, "default")
            ebs = get_cache_family_granularity(base_bs, cf)
            lookup_start = hbm_hit_tokens // ebs * ebs
            lookup_mask = None
            if lookup_masks is not None and group_id < len(lookup_masks):
                lookup_mask = lookup_masks[group_id]
            _kvtrace_lookup_mask_skipped = 0
            _kvtrace_lookup_mask_seen = 0
            for start_idx, _, key_string, chunk_hash in self.token_database.process_token_key_strings(
                token_len,
                block_hashes,
                mask_num=lookup_start,
                kv_cache_group_id=group_id,
                max_num=lookup_limit,
            ):
                _kvtrace_lookup_mask_seen += 1
                if lookup_mask is not None:
                    chunk_idx = start_idx // base_bs if base_bs > 0 else 0
                    if chunk_idx >= len(lookup_mask) or not lookup_mask[chunk_idx]:
                        _kvtrace_lookup_mask_skipped += 1
                        continue
                _kvtrace_raw_count += 1
                _kvtrace_t_a = _kvtrace_time.perf_counter()
                variants = self._expand_lookup_key_variants(key_string, group_id, include_all_ranks)
                keys.extend(variants)
                chunk_hashes.append(chunk_hash)
                variant_counts.append(len(variants))
                _kvtrace_app_ms += (_kvtrace_time.perf_counter() - _kvtrace_t_a) * 1000
            _kvtrace_iter_ms = (_kvtrace_time.perf_counter() - _kvtrace_t_kb) * 1000 - _kvtrace_app_ms
            _kvtrace_str_ms = 0.0

            _kvtrace_kb_ms = (_kvtrace_time.perf_counter() - _kvtrace_t_kb) * 1000
            _kvtrace_total_keybuild += _kvtrace_kb_ms
            _kvtrace_raw_keys = len(chunk_hashes)
            token_chunks = (lookup_limit + ebs - 1) // ebs if ebs > 0 else 0
            skipped_chunks = min(token_chunks, lookup_start // ebs) if ebs > 0 else 0
            expected_lookup_chunks = max(0, token_chunks - skipped_chunks)
            logger.info(
                "KVTRACE stage=kvpool_lookup_group_key_build group=%d "
                "elapsed_ms=%.3f raw_keys=%d variant_keys=%d token_len=%d "
                "hbm_hit_tokens=%d lookup_start=%d lookup_limit=%d effective_block_size=%d "
                "token_chunks=%d skipped_chunks=%d expected_lookup_chunks=%d",
                group_id, _kvtrace_kb_ms, _kvtrace_raw_keys, len(keys), token_len,
                hbm_hit_tokens, lookup_start, lookup_limit, ebs,
                token_chunks, skipped_chunks, expected_lookup_chunks
            )
            if self.lookup_reachable_mask:
                logger.info(
                    "KVTRACE stage=kvpool_lookup_reachable_mask group=%d "
                    "seen_chunks=%d skipped_by_mask=%d built_raw_keys=%d "
                    "mask_kind=%s token_len=%d hbm_hit_tokens=%d lookup_start=%d",
                    group_id, _kvtrace_lookup_mask_seen,
                    _kvtrace_lookup_mask_skipped, _kvtrace_raw_keys,
                    "all" if lookup_mask is None else f"{sum(1 for item in lookup_mask if item)}/{len(lookup_mask)}",
                    token_len, hbm_hit_tokens, lookup_start,
                )
            logger.info(
                "KVTRACE stage=kvpool_6d_group_mask group=%d "
                "hbm_hit_tokens=%d lookup_start=%d lookup_limit=%d effective_block_size=%d "
                "skipped_chunks=%d expected_lookup_chunks=%d built_raw_keys=%d built_variant_keys=%d",
                group_id, hbm_hit_tokens, lookup_start, lookup_limit, ebs,
                skipped_chunks, expected_lookup_chunks, _kvtrace_raw_keys, len(keys)
            )
            logger.info(
                "KVTRACE stage=kvpool_kb_detail group=%d "
                "raw_count=%d iter_ms=%.3f str_ms=%.3f app_ms=%.3f total_ms=%.3f "
                "compress_ratios=%s grouped_block_size=%s",
                group_id, _kvtrace_raw_count,
                _kvtrace_iter_ms, _kvtrace_str_ms, _kvtrace_app_ms, _kvtrace_kb_ms,
                getattr(self, "compress_ratios", None),
                self.grouped_block_size[group_id] if hasattr(self, "grouped_block_size") else "N/A"
            )

            if not keys:
                logger.info(
                    "KVTRACE stage=kvpool_6d_group_skip_external group=%d reason=no_keys "
                    "hbm_hit_tokens=%d lookup_start=%d lookup_limit=%d effective_block_size=%d "
                    "token_chunks=%d skipped_chunks=%d expected_lookup_chunks=%d",
                    group_id, hbm_hit_tokens, lookup_start, lookup_limit, ebs,
                    token_chunks, skipped_chunks, expected_lookup_chunks
                )
                continue
            _kvtrace_t0 = _kvtrace_time.perf_counter()
            res = self.m_store.exists(keys)  # type: ignore[assignment]
            _kvtrace_exist_ms = (_kvtrace_time.perf_counter() - _kvtrace_t0) * 1000
            _kvtrace_total_exist += _kvtrace_exist_ms
            _kvtrace_exists_count = sum(1 for v in res if v == 1) if res else 0
            logger.info(
                "KVTRACE stage=kvpool_exist_backend elapsed_ms=%.3f "
                "group=%d keys=%d exists_count=%d missing_count=%d",
                _kvtrace_exist_ms, group_id, len(keys),
                _kvtrace_exists_count, len(keys) - _kvtrace_exists_count
            )
            _kvtrace_t_hc = _kvtrace_time.perf_counter()
            offset = 0
            for chunk_hash, count in zip(chunk_hashes, variant_counts, strict=True):
                values = res[offset : offset + count]  # type: ignore[index]
                if values and all(value == 1 for value in values):
                    exists.add((group_id, self._chunk_hash_to_bytes(chunk_hash)))
                offset += count

            _kvtrace_hc_ms = (_kvtrace_time.perf_counter() - _kvtrace_t_hc) * 1000
            _kvtrace_total_hitcalc += _kvtrace_hc_ms
            logger.info(
                "KVTRACE stage=kvpool_lookup_group_hit_calc group=%d "
                "elapsed_ms=%.3f hit_chunks=%d/%d token_len=%d",
                group_id, _kvtrace_hc_ms,
                sum(1 for g, _ in exists if g == group_id),
                len(chunk_hashes), token_len
            )
            logger.debug(
                "KV pool coordinator lookup group=%d token_len=%d keys=%d exists_chunks=%d/%d sample_keys=%s",
                group_id,
                token_len,
                len(keys),
                sum(1 for group, _ in exists if group == group_id),
                len(chunk_hashes),
                keys[:3],
            )
            if self.lookup_early_stop and chunk_hashes and expected_lookup_chunks > 0:
                first_chunk_exists = (
                    group_id,
                    self._chunk_hash_to_bytes(chunk_hashes[0]),
                ) in exists
                if not first_chunk_exists:
                    hit_length = min(max(hbm_hit_tokens, 0), token_len)
                    total_ms = (
                        _kvtrace_total_keybuild
                        + _kvtrace_total_exist
                        + _kvtrace_total_hitcalc
                    )
                    logger.info(
                        "KVTRACE stage=kvpool_lookup_early_stop group=%d "
                        "reason=first_external_chunk_missing hit_tokens=%d "
                        "hbm_hit_tokens=%d lookup_start=%d effective_block_size=%d "
                        "processed_groups=%s skipped_groups=%s key_build_ms=%.3f "
                        "exist_ms=%.3f hit_calc_ms=%.3f total_ms=%.3f",
                        group_id, hit_length, hbm_hit_tokens, lookup_start, ebs,
                        kv_cache_group_ids[:kv_cache_group_ids.index(group_id) + 1],
                        kv_cache_group_ids[kv_cache_group_ids.index(group_id) + 1:],
                        _kvtrace_total_keybuild, _kvtrace_total_exist,
                        _kvtrace_total_hitcalc, total_ms,
                    )
                    logger.info(
                        "KVTRACE stage=kvpool_lookup_breakdown token_len=%d "
                        "key_build_ms=%.3f exist_ms=%.3f hit_calc_ms=%.3f "
                        "coord_hit_ms=0.000 total_ms=%.3f hit_tokens=%d groups=%s early_stop=True",
                        token_len, _kvtrace_total_keybuild, _kvtrace_total_exist,
                        _kvtrace_total_hitcalc, total_ms, hit_length, kv_cache_group_ids,
                    )
                    return hit_length

        _kvtrace_t_coord = _kvtrace_time.perf_counter()
        _, hit_length = self.cache_coordinator.find_longest_cache_hit(
            block_hashes,
            token_len,
            ExternalCachedBlockPool(exists),
            apply_eagle=False,
        )
        _kvtrace_coord_ms = (_kvtrace_time.perf_counter() - _kvtrace_t_coord) * 1000
        logger.info(
            "KVTRACE stage=kvpool_lookup_breakdown token_len=%d "
            "key_build_ms=%.3f exist_ms=%.3f hit_calc_ms=%.3f "
            "coord_hit_ms=%.3f total_ms=%.3f hit_tokens=%d groups=%s full_guard=%s lookup_limit=%d",
            token_len,
            _kvtrace_total_keybuild, _kvtrace_total_exist,
            _kvtrace_total_hitcalc, _kvtrace_coord_ms,
            _kvtrace_total_keybuild + _kvtrace_total_exist + _kvtrace_total_hitcalc + _kvtrace_coord_ms,
            hit_length, kv_cache_group_ids, self.lookup_full_guard, lookup_limit
        )
        logger.debug(
            "KV pool coordinator lookup final token_len=%d groups=%s hit=%d",
            token_len,
            kv_cache_group_ids,
            hit_length,
        )
        return hit_length

    def lookup_scheduler(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int] | None = None,
        use_layerwise: bool = False,
        hbm_hit_tokens: int = 0,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.
        :param tokens: the input tokens, with shape [seq_len]
        :return: An int indicating how many prefix tokens are cached.
        """
        try:
            hits = []
            kv_cache_group_ids = kv_cache_group_ids or [0]
            coordinator_hit = self._lookup_with_coordinator(
                token_len,
                block_hashes,
                kv_cache_group_ids,
                use_layerwise,
                include_all_ranks=True,
                hbm_hit_tokens=hbm_hit_tokens,
            )
            if coordinator_hit is not None:
                return coordinator_hit
            for group_id in kv_cache_group_ids:
                end = 0
                keys = []
                starts = []
                ends = []
                for start, end, key in self.token_database.process_tokens(
                    token_len,
                    block_hashes,
                    kv_cache_group_id=group_id,
                ):
                    if use_layerwise:
                        keys_multi_layer = key.split_layers(self.num_layers)
                        for item in keys_multi_layer:
                            keys.append(item.to_string())
                    else:
                        keys.append(key.to_string())
                    starts.append(start)
                    ends.append(end)

                if not keys:
                    hits.append(0)
                    continue

                multi_tp_keys = keys[:]
                group_tp_size = self.get_group_tp_size(group_id)
                for i in range(1, group_tp_size):
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
                    for i in range(group_tp_size * self.pp_size)
                ]
                first_missing = self.find_min_first_non_one_index(multi_tp_values)
                logger.debug(
                    "KV pool lookup request token_len=%d group=%d keys=%d multi_tp_keys=%d "
                    "exists_count=%d/%d first_missing=%d exists_sample=%s sample_keys=%s",
                    token_len,
                    group_id,
                    len(keys),
                    len(multi_tp_keys),
                    sum(1 for value in res if value == 1),  # type: ignore[union-attr]
                    len(res),
                    first_missing,
                    list(res[: min(12, len(res))]),  # type: ignore[index]
                    multi_tp_keys[:3],
                )
                if group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]:
                    exists_by_block = [all(values[idx] == 1 for values in multi_tp_values) for idx in range(num_block)]
                    hit_end = 0
                    for index in range(num_block - 1, -1, -1):
                        if exists_by_block[index] and ends[index] % self.cache_transfer_granularity == 0:
                            hit_end = ends[index]
                            break
                    hits.append(hit_end)
                else:
                    index = first_missing
                    if index == -1:
                        hits.append(end)
                    else:
                        hit_end = 0
                        for hit_index in range(index, 0, -1):
                            if starts[hit_index] % self.cache_transfer_granularity == 0:
                                hit_end = starts[hit_index]
                                break
                        hits.append(hit_end)
                logger.debug(
                    "KV pool scheduler lookup group=%d keys=%d hit=%d token_len=%d",
                    group_id,
                    len(keys),
                    hits[-1],
                    token_len,
                )
        except Exception as e:
            logger.error("Remote connection failed in contains: %s", e)
            return 0
        logger.debug(
            "KV pool scheduler lookup final token_len=%d groups=%s hits=%s result=%d",
            token_len,
            kv_cache_group_ids,
            hits,
            min(hits) if hits else 0,
        )
        return min(hits) if hits else 0

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
