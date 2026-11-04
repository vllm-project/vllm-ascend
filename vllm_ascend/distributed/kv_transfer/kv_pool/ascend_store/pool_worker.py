from __future__ import annotations

import importlib
import math
import threading
import time
from collections.abc import Callable, Sequence

import torch
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pcp_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.core.kv_cache_placement import map_kvpp_layers_to_owners
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    backend_map,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import AscendStoreCoordinator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_cache import (
    DSparkPrefixCache,
    DSparkPrefixKeys,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreBatch,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
    record_failed_blocks,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import (
    KVCacheStoreLayerSendingThread,
    _circular_shift,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    get_physical_layer_index,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    ReqMeta,
    get_block_hashes,
    get_group_cache_family,
    infer_cache_transfer_granularity,
    infer_cacheable_group_ids,
    infer_group_block_sizes,
    infer_group_cache_families,
    infer_tp_mismatch_info,
    uses_hybrid_kv_cache,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metrics import (
    AscendStoreKVConnectorStats,
)
from vllm_ascend.distributed.utils import (
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
)


class KVPoolWorker:
    # The main class for the cache engine.

    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwise: bool,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.vllm_config = vllm_config
        self.use_kvpp = KVPPConfig.from_vllm_config(vllm_config).size > 1
        self.kv_cache_config = kv_cache_config
        hf_text_config = getattr(model_config, "hf_text_config", None)
        hf_config = getattr(model_config, "hf_config", hf_text_config)
        self.hf_config = hf_text_config or hf_config
        self.compress_ratios = getattr(hf_text_config, "compress_ratios", None)
        self.max_model_len = model_config.max_model_len
        if self.compress_ratios is None:
            self.compress_ratios = getattr(hf_config, "compress_ratios", None)
        self.use_compress = self.compress_ratios is not None
        self.dp_rank = parallel_config.data_parallel_rank

        self._init_parallelism_info(model_config, parallel_config)
        self._init_kv_transfer_config(vllm_config, extra_config, use_layerwise, kv_cache_config)
        self._init_key_head_config(model_config, parallel_config)
        self._init_metadata(model_config, vllm_config, extra_config)
        self._init_backend(parallel_config, extra_config)
        self._init_kv_events(vllm_config)
        self._init_state_vars()
        self._dspark_prefix_keys: DSparkPrefixKeys | None = None
        self.dspark_prefix_cache: DSparkPrefixCache | None = None
        if not use_layerwise:
            self._init_cache_layout()
        self._kv_stats = AscendStoreKVConnectorStats()
        self._kv_stats_lock = threading.Lock()

    def _init_parallelism_info(self, model_config, parallel_config) -> None:
        self.local_rank = envs.LOCAL_RANK

        self.use_mla = False
        if hasattr(model_config, "use_mla") and isinstance(model_config.use_mla, bool) and model_config.use_mla:
            self.use_mla = True
        self.use_sparse = hasattr(model_config.hf_text_config, "index_topk")

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = get_pp_group().rank_in_group if self.pp_size > 1 else 0

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.kvpp_rank = self.pcp_rank * self.tp_size + self.tp_rank
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0
        self.model_name = model_config.model.split("/")[-1]

        # Global layer addressing for layerwise keys under pipeline parallel:
        # get_num_layers returns the per-stage layer count, so record the
        # stage's global layer offset and the model-wide total layers used by
        # the layerwise key space.
        self.pp_layer_offset = 0
        self.total_layers = None
        try:
            start, _ = model_config.get_layers_start_end_indices(parallel_config)
            self.pp_layer_offset = start
            self.total_layers = model_config.get_total_num_hidden_layers()
        except AttributeError:
            # Older vLLM without PP layer-index APIs: fall back to the
            # non-PP layout (offset 0, total == local).
            self.pp_layer_offset = 0
            self.total_layers = None
        except Exception as e:
            logger.warning(
                "Failed to get PP layer indices (pp_size=%d), falling back to local layer addressing: %s",
                self.pp_size,
                e,
            )
            self.pp_layer_offset = 0
            self.total_layers = None

    def _init_kv_transfer_config(self, vllm_config, extra_config, use_layerwise, kv_cache_config) -> None:
        self._extra_config = extra_config
        self.use_layerwise = use_layerwise
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.load_async = extra_config.get("load_async", False)
        self._save_disabled = threading.Event()
        self._closed = False
        self._invalid_block_ids: set[int] = set()
        self._failed_requests: set[str] = set()
        self._invalid_block_ids_lock = threading.Lock()
        self._retired_store_req_ids: set[str] = set()
        self.consumer_is_to_put = extra_config.get("consumer_is_to_put", False)
        self.save_decode_cache = extra_config.get("save_decode_cache", False)
        self.can_put = self.kv_role in ("kv_producer", "kv_both") or self.consumer_is_to_put or self.save_decode_cache
        self.backend = extra_config.get("backend", "mooncake")
        self.backend_name = self.backend.lower()
        kv_cache_groups = kv_cache_config.kv_cache_groups if kv_cache_config is not None else None
        self.use_hybrid = self._compute_use_hybrid(vllm_config, kv_cache_groups)
        self._attention_saved_layers: set[int] = set()
        self.use_mamba = self._uses_mamba_kv_cache(self.use_hybrid, kv_cache_config)
        speculative_config = getattr(vllm_config, "speculative_config", None)
        use_eagle_fn = getattr(speculative_config, "use_eagle", None)
        self.use_eagle = use_eagle_fn() is True if callable(use_eagle_fn) else False
        self.original_block_size = infer_group_block_sizes(vllm_config.cache_config.block_size, kv_cache_groups)
        self.cacheable_group_ids = infer_cacheable_group_ids(kv_cache_groups)
        cacheable_block_sizes = [self.original_block_size[i] for i in self.cacheable_group_ids]
        self.grouped_block_size = [block_size * self.dcp_size for block_size in self.original_block_size]
        requested_hash_block_size = vllm_config.cache_config.prefix_match_unit
        if not isinstance(requested_hash_block_size, int):
            requested_hash_block_size = None
        self.hash_block_size = (
            requested_hash_block_size if requested_hash_block_size is not None else min(cacheable_block_sizes)
        ) * self.dcp_size
        for group_id in self.cacheable_group_ids:
            group_block_size = self.grouped_block_size[group_id]
            assert group_block_size % self.hash_block_size == 0, "block_size must be divisible by hash_block_size"
        self.block_size = self.grouped_block_size[0]
        self.lcm_block_size = math.lcm(*(self.grouped_block_size[i] for i in self.cacheable_group_ids))
        self.num_kv_cache_groups = len(self.grouped_block_size)
        self.kv_cache_group_families = infer_group_cache_families(kv_cache_groups, self.compress_ratios, self.hf_config)
        self.group_uses_align_state = self._infer_group_uses_align_state()
        self.cache_transfer_granularity = infer_cache_transfer_granularity(
            self.grouped_block_size, self.lcm_block_size, self.cacheable_group_ids
        )
        self.h2d_stagger_us = int(extra_config.get("h2d_stagger_us", 0))

        logger.info(
            "use_hybrid: %s, use_mamba: %s, num_kv_cache_groups: %s, hash_block_size: %s, lcm_block_size: %s",
            self.use_hybrid,
            self.use_mamba,
            self.num_kv_cache_groups,
            self.hash_block_size,
            self.lcm_block_size,
        )

    def _compute_use_hybrid(self, vllm_config, kv_cache_groups) -> bool:
        return uses_hybrid_kv_cache(vllm_config.scheduler_config, kv_cache_groups)

    def _init_key_head_config(self, model_config, parallel_config) -> None:
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
        if self.use_kvpp:
            # Every owner saves all blocks of its layer shard, including its MTP replica.
            self.head_or_tp_rank = self.kvpp_rank
            self.put_step = 1
        self.my_key_index = (
            self.pcp_rank * self.dcp_size * (self.tp_size // self.put_step)
            + self.dcp_rank * (self.tp_size // self.put_step)
            + self.head_or_tp_rank
        )
        self.num_ranks_per_layer = self.pcp_size * self.dcp_size * (self.tp_size // self.put_step)

        extra_cfg = self._extra_config
        tp_mismatch_info = infer_tp_mismatch_info(
            self.kv_role,
            extra_cfg,
            self.tp_size,
            self.num_kv_head,
            self.use_mla,
        )
        self.peer_tp_size = tp_mismatch_info.peer_tp_size
        self.effective_tp_size = tp_mismatch_info.effective_tp_size
        self.tp_mismatch = tp_mismatch_info.enabled
        if self.tp_mismatch:
            if self.use_sparse:
                raise ValueError(
                    f"TP mismatch (local_tp={self.tp_size}, peer_tp={self.peer_tp_size}) "
                    "is not supported with sparse KV layouts (use_sparse=True). "
                    "Strided I/O requires uniform block_len across all cache entries."
                )
            if self.use_hybrid and not self.use_layerwise:
                raise NotImplementedError(
                    f"TP mismatch (local_tp={self.tp_size}, peer_tp={self.peer_tp_size}) "
                    "is not yet supported with hybrid KV cache layouts (e.g. DSV4). "
                    "The strided I/O path assumes a single dense KV group."
                )
            self.local_heads_per_rank = tp_mismatch_info.local_heads_per_rank
            self.effective_heads_per_rank = tp_mismatch_info.effective_heads_per_rank
            self.num_sub_keys = tp_mismatch_info.num_sub_keys
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
            self.local_heads_per_rank = tp_mismatch_info.local_heads_per_rank
            self.effective_heads_per_rank = tp_mismatch_info.effective_heads_per_rank
            self.num_sub_keys = tp_mismatch_info.num_sub_keys

    def _init_metadata(self, model_config, vllm_config, extra_config) -> None:
        partitions = None
        if self.kv_role == "kv_consumer" and self.can_put:
            num_hidden_layers = model_config.hf_text_config.num_hidden_layers
            partition_list_str = extra_config.get("prefill_pp_layer_partition", None)
            prefill_pp_size = int(extra_config.get("prefill_pp_size", 1))

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

        self.metadata: list[KeyMetadata] = []
        for group_id in range(self.num_kv_cache_groups):
            # the mamba kv_heads is not same with the full attention, can't share the cache data
            group_tp_rank = self.tp_rank if self.group_uses_align_state[group_id] else self.head_or_tp_rank
            self.metadata.append(
                KeyMetadata(
                    model_config.model.rstrip("/").split("/")[-1],
                    group_tp_rank,
                    self.dcp_rank,
                    self.pp_rank,
                    group_id,
                )
            )

        self.token_database = ChunkedTokenDatabase(
            self.metadata, self.grouped_block_size, partitions, hash_block_size=self.hash_block_size
        )
        self.cache_coordinator = self._build_cache_coordinator(vllm_config)
        self.token_database.cache_coordinator = self.cache_coordinator

    def _init_backend(self, parallel_config, extra_config) -> None:
        backend = backend_map.get(self.backend.lower())
        assert backend is not None
        backend_path = backend.get("path")
        backend_name = backend.get("name")
        assert backend_path is not None and backend_name is not None
        backend_module = importlib.import_module(backend_path)
        real_backend = getattr(backend_module, backend_name)

        backend_kwargs = {}
        # lazy_init is requested only for compressed models. Each backend owns
        # the hardware/protocol checks that decide whether it can honor it.
        backend_kwargs["lazy_init"] = self.use_compress
        # The connector's extra_config (with MultiConnector the child's own
        # config, not the top-level one) carries the QoS the backends inject.
        backend_kwargs["extra_config"] = extra_config
        self.m_store = real_backend(  # type: ignore[misc]
            parallel_config,
            **backend_kwargs,
        )

    def _init_kv_events(self, vllm_config) -> None:
        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = False
        if kv_event_config and kv_event_config.enable_kv_cache_events:
            self.enable_kv_events = True

    def _init_state_vars(self) -> None:
        self.kv_send_thread: KVTransferThread | None = None
        self.kv_recv_thread: KVTransferThread | None = None
        self._previous_save_batch: KVCacheStoreBatch | None = None
        self._transfer_threads_started = False
        self.external_slot_release_waiter: Callable[[int], None] | None = None

    def _init_cache_layout(self) -> None:
        self.physical_layer_to_group_layers: dict[int, list[tuple[int, int]]] = {}
        self._global_to_local_layer: dict[int, int] = {}
        if self.kv_cache_config is None:
            return
        physical_layers = {
            self._extract_physical_layer_index(layer_name)
            for group in self.kv_cache_config.kv_cache_groups
            for layer_name in group.layer_names
        }
        self._global_to_local_layer = {
            global_layer: local_layer for local_layer, global_layer in enumerate(sorted(physical_layers))
        }
        self.num_layers = max(self.num_layers, len(physical_layers))
        if self.num_kv_cache_groups > 1:
            for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups):
                if group_id not in self.cacheable_group_ids:
                    continue
                local_layers = sorted(
                    {
                        self._global_to_local_layer[self._extract_physical_layer_index(name)]
                        for name in group.layer_names
                    }
                )
                for layer_index, local_layer in enumerate(local_layers):
                    self.physical_layer_to_group_layers.setdefault(local_layer, []).append((group_id, layer_index))

    def _group_pp_byte_offset(self, group_id: int) -> int:
        """Keep GVA allocation and transfer origins on the same PP layout."""
        if getattr(self, "pp_size", 1) == 1 or getattr(self, "use_block_key_layerwise", False):
            return 0
        block_lens = self.group_block_len.get(group_id) or []
        num_layers = int(self.group_num_layers.get(group_id, 0))
        if not block_lens or num_layers <= 0:
            return 0
        per_layer = sum(block_lens) // num_layers
        return int(getattr(self, "layerwise_key_layer_offset", 0)) * per_layer

    def shutdown(self) -> None:
        if self._closed:
            return
        threads = [thread for thread in (self.kv_send_thread, self.kv_recv_thread) if thread is not None]
        for thread in threads:
            thread.stop()
        for thread in threads:
            if thread.is_alive():
                thread.join()
        self.m_store.close()
        self._closed = True

    def set_external_slot_release_waiter(self, waiter: Callable[[int], None]) -> bool:
        return False

    def _start_kv_transfer_threads(self) -> None:
        if self._transfer_threads_started:
            return

        if self.can_put:
            ready_event_sending = threading.Event()
            self.kv_send_thread = KVCacheStoreSendingThread(
                self.m_store,
                self.token_database,
                self.grouped_block_size,
                self.tp_rank,
                self.tp_size,
                # KVPP owners hold different layers, not PCP replicas.
                0 if self.use_kvpp else self.pcp_rank,
                1 if self.use_kvpp else self.pcp_size,
                self.dcp_size,
                self.put_step,
                self.kv_role,
                ready_event_sending,
                self.group_uses_align_state,
                self.enable_kv_events,
                worker=self if self.tp_mismatch else None,
            )
            self.kv_send_thread.start()
            ready_event_sending.wait()
        if self.load_async:
            ready_event = threading.Event()
            self.kv_recv_thread = KVCacheStoreRecvingThread(
                self.m_store,
                self.token_database,
                self.grouped_block_size,
                self.tp_rank,
                self.tp_size,
                self.dcp_size,
                ready_event,
                invalid_block_ids=self._invalid_block_ids,
                invalid_block_ids_lock=self._invalid_block_ids_lock,
                worker=self if self.tp_mismatch else None,
                record_operation=self._record_kv_connector_operation,
            )
            self.kv_recv_thread.start()
            ready_event.wait()
        self._transfer_threads_started = True

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
            self.kv_cache_config.kv_cache_groups,
            scheduler_block_size=self.cache_transfer_granularity,
            hash_block_size=self.hash_block_size,
            group_block_sizes=self.grouped_block_size,
            group_cache_families=self.kv_cache_group_families,
            use_eagle=use_eagle,
            retention_interval=retention_interval,
        )

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

    @staticmethod
    def _uses_mamba_kv_cache(use_hybrid: bool, kv_cache_config: KVCacheConfig | None):
        if not use_hybrid or kv_cache_config is None:
            return False
        for group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                if any(isinstance(spec, MambaSpec) for spec in kv_cache_spec.kv_cache_specs.values()):
                    return True
            elif isinstance(kv_cache_spec, MambaSpec):
                return True
        return False

    @staticmethod
    def _as_cache_tuple(cache_or_caches) -> tuple[torch.Tensor, ...]:
        if isinstance(cache_or_caches, torch.Tensor):
            cache_or_caches = (cache_or_caches,)
        # NoPE MLA exposes an empty RoPE view whose data_ptr() is zero.
        return tuple(cache for cache in cache_or_caches if cache.numel())

    def _get_cache_block_metadata(self, cache: torch.Tensor) -> tuple[int, int, int, int]:
        tensor_num_blocks = cache.shape[0]
        assert tensor_num_blocks % self.num_blocks == 0, (
            "The external block size must be an integer multiple of the kernel block size."
        )
        block_size_scale = tensor_num_blocks // self.num_blocks
        block_len = cache[0].numel() * cache.element_size() * block_size_scale
        block_stride = cache.stride(0) * cache.element_size() * block_size_scale
        # Registration covers the physical span, including gaps between kernel
        # blocks inside one scheduler block and non-contiguous inner views.
        region_len = (
            (1 + sum((size - 1) * stride for size, stride in zip(cache.shape, cache.stride()))) * cache.element_size()
            if cache.numel()
            else 0
        )
        return block_len, block_stride, region_len, block_size_scale

    def _get_cache_transfer_segments(self, cache: torch.Tensor) -> list[tuple[int, int, int]]:
        """Describe full-block pool transfers without crossing kernel-page gaps.

        Both bulk and layerwise transfers must preserve physical-block gaps.
        Partial layerwise transfers also retain each segment's token interval.
        """
        block_len, block_stride, _, scale = self._get_cache_block_metadata(cache)
        physical_bytes = block_len // scale
        physical_stride = block_stride // scale
        if scale == 1 or physical_stride == physical_bytes:
            return [(cache.data_ptr(), block_len, block_stride)]
        if not cache[0].is_contiguous():
            raise ValueError("Segmented KV pool transfer requires contiguous data inside each physical block.")
        return [(cache.data_ptr() + index * physical_stride, physical_bytes, block_stride) for index in range(scale)]

    def _get_combined_cache_transfer_segments(
        self, caches: tuple[torch.Tensor, ...], group_id: int
    ) -> list[tuple[int, int, int]] | None:
        """Coalesce adjacent MLA components for complete bulk pages only.

        The packed object becomes physical-page major. All readers and writers
        must use this layout and a fresh namespace when upgrading an old pool.
        Layerwise/partial transfers retain component-major segment metadata.
        """
        if self.use_layerwise or len(caches) != 2:
            return None
        extra_config = getattr(self, "_extra_config", None)
        if not isinstance(extra_config, dict) or extra_config.get("discard_partial_chunks", True) is not True:
            return None
        if extra_config.get("save_decode_cache", False) or getattr(self, "tp_mismatch", False):
            return None
        if getattr(self, "dcp_size", 1) != 1 or getattr(self, "use_kvpp", False):
            return None
        block_sizes = getattr(self, "grouped_block_size", ())
        granularity = getattr(self, "cache_transfer_granularity", 0)
        if group_id >= len(block_sizes) or not granularity or granularity % block_sizes[group_id]:
            return None
        align_state = getattr(self, "group_uses_align_state", ())
        if group_id < len(align_state) and align_state[group_id]:
            return None
        first, second = caches
        if (
            first.ndim != 4
            or second.ndim != 4
            or first.shape[:-1] != second.shape[:-1]
            or first.dtype != second.dtype
            or first.device != second.device
            or self._get_storage_key(first) != self._get_storage_key(second)
            or not first[0].is_contiguous()
            or not second[0].is_contiguous()
        ):
            return None
        first_len, block_stride, _, scale = self._get_cache_block_metadata(first)
        second_len, second_stride, _, second_scale = self._get_cache_block_metadata(second)
        if scale <= 1 or scale != second_scale or block_stride != second_stride:
            return None
        first_bytes = first_len // scale
        page_bytes = (first_len + second_len) // scale
        page_stride = block_stride // scale
        if second.data_ptr() != first.data_ptr() + first_bytes or page_stride < page_bytes:
            return None
        if page_stride == page_bytes:
            return [(first.data_ptr(), first_len + second_len, block_stride)]
        return [(first.data_ptr() + index * page_stride, page_bytes, block_stride) for index in range(scale)]

    @staticmethod
    def _get_storage_key(cache: torch.Tensor) -> int:
        try:
            return cache.untyped_storage().data_ptr()
        except AttributeError:
            return cache.storage().data_ptr()

    def _extract_physical_layer_index(self, layer_name: str) -> int:
        base_layers = getattr(
            self.hf_config,
            "num_hidden_layers",
            self.num_layers,
        )
        return get_physical_layer_index(layer_name, base_layers)

    def _global_group_alloc_size(self, group_id: int) -> int:
        # GLOBAL region size: per-layer bytes x TOTAL model layers.
        # Under PP each stage sees a LOCAL group view (non-uniform splits make
        # the per-stage layer counts differ) but all stages plus the decode
        # side share ONE region per pool key, so the region must cover the
        # union of all writers. vllm groups layers by identical cache spec,
        # so per-layer bytes are uniform within a group.
        gbl = self.group_block_len.get(group_id) or []
        if not gbl:
            return self.page_size_bytes
        # Backward compatibility: when total_layers is not set (unit tests,
        # partial init), fall back to the LOCAL sum (pre-PP behavior).
        total_layers = getattr(self, "total_layers", None)
        if not total_layers or not isinstance(total_layers, int):
            return sum(gbl)
        n_local = int(self.group_num_layers.get(group_id, 0))
        if n_local <= 0:
            return sum(gbl)
        per_layer = (sum(gbl) + n_local - 1) // n_local
        n_global = max(total_layers, int(self.num_layers), n_local)
        # Physical cache layers (for example MTP) can extend past the target
        # model's last layer. Preserve the existing byte origin, but cover the
        # complete local payload instead of sizing from target layers alone.
        shard_bytes = max(per_layer * n_global, self._group_pp_byte_offset(group_id) + sum(gbl))
        # DCP shard-major layout: ranks sharing head_or_tp_rank (put_step>1,
        # e.g. MLA) hold different context shards of the same logical block
        # and share ONE region, so the region must cover all shards. The
        # per-shard stride is the group byte total aligned up to the GVA
        # hugepage size: hybrid DSA groups have non-uniform per-layer bytes,
        # and an unaligned stride would yield misaligned GVA addresses that
        # SDMA rejects. With put_step == 1 every rank owns a distinct region
        # key and no shard separation is needed.
        if self.put_step > 1 and self.dcp_size > 1:
            gva_align = 2 * 1024 * 1024
            shard_stride = (shard_bytes + gva_align - 1) // gva_align * gva_align
            return shard_stride * self.dcp_size
        return shard_bytes

    def _infer_cache_group_metadata(self, group_id: int, layer_names: list[str]):
        group_addrs: list[int] = []
        group_block_lens: list[int] = []
        group_block_strides: list[int] = []
        group_cache_entry_segments: list[tuple[int, int]] = []
        layer_names_by_physical: dict[int, list[str]] = {}
        for layer_name in layer_names:
            phys = self._extract_physical_layer_index(layer_name)
            phys = self._global_to_local_layer.get(phys, phys)
            layer_names_by_physical.setdefault(phys, []).append(layer_name)

        layer_cache_entry_offsets = [0]
        for phys in sorted(layer_names_by_physical):
            if getattr(self, "use_kvpp", False):
                self.physical_layer_to_group_layers.setdefault(phys, []).append(
                    (group_id, len(layer_cache_entry_offsets) - 1)
                )
            for layer_name in sorted(layer_names_by_physical[phys]):
                cache_or_caches = self.kv_caches[layer_name]
                caches = self._as_cache_tuple(cache_or_caches)
                combined = self._get_combined_cache_transfer_segments(caches, group_id)
                cache_segments = (
                    [combined]
                    if combined is not None
                    else [self._get_cache_transfer_segments(cache) for cache in caches]
                )
                for segments in cache_segments:
                    for segment_index, (base_addr, block_len, block_stride) in enumerate(segments):
                        group_cache_entry_segments.append((segment_index, len(segments)))
                        group_addrs.append(base_addr)
                        group_block_lens.append(block_len)
                        group_block_strides.append(block_stride)
                if combined is not None:
                    logger.info(
                        "KVPool coalesced bulk cache layer=%s group=%d descriptors=%d->%d",
                        layer_name,
                        group_id,
                        2 * (caches[0].shape[0] // self.num_blocks),
                        len(combined),
                    )
            layer_cache_entry_offsets.append(len(group_addrs))
        self.group_kv_caches_base_addr[group_id] = group_addrs
        self.group_block_len[group_id] = group_block_lens
        self.group_block_stride[group_id] = group_block_strides
        self.group_layer_cache_entry_offsets[group_id] = layer_cache_entry_offsets
        self.group_cache_entry_segments[group_id] = group_cache_entry_segments
        self.group_num_layers[group_id] = len(layer_names_by_physical)

    def _align_kv_ptrs(self, registered_regions: dict[int, tuple[int, int]]):
        """
        In hybrid scenario, where a KVCacheTensor is shared by multiple layers,
        but sometimes, layers cannot be evenly distributed among multiple groups,
        the layers sharing the KVCacheTensor may not completely occupy all the space of the KVCacheTensor.
        This results in the calculated start address not being the previously aligned address.
        Therefore, we down-align the start address to meet the 2MB alignment requirement.
        """
        if not self.use_hybrid:
            return
        alignment = 2 * 1024 * 1024
        for storage_key in registered_regions:
            start, end = registered_regions[storage_key]
            new_start = start // alignment * alignment
            # Because the addresses of raw tensors are aligned to 2MB,
            # all shared sub-tensors, when aligned downwards, should theoretically not exceed the address bounds.
            assert new_start >= storage_key, "invalid kv cache tensor, raw tensor ptr must be align to 2MB"
            registered_regions[storage_key] = (new_start, end)

    def _register_dspark_prefix_cache(self, draft_caches: dict[str, torch.Tensor]) -> None:
        """Register draft caches when the layerwise worker enables prefix reuse."""

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        draft_names = getattr(self, "_dspark_draft_layer_names", ())
        draft_caches = {name: cache for name, cache in kv_caches.items() if name in draft_names}
        if draft_names:
            kv_caches = {name: cache for name, cache in kv_caches.items() if name not in draft_names}
        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache_tuple = self._as_cache_tuple(first_kv_cache_tuple)
        first_kv_cache = first_kv_cache_tuple[0]

        self.num_blocks = (
            self.kv_cache_config.num_blocks if self.kv_cache_config is not None else first_kv_cache.shape[0]
        )
        self._register_dspark_prefix_cache(draft_caches)
        logger.info("num_blocks: %s", self.num_blocks)
        self.block_len = []
        self.block_stride = []
        for cache in first_kv_cache_tuple:
            block_len, block_stride, _, _ = self._get_cache_block_metadata(cache)
            logger.info("block_shape: %s", cache.shape[1:])
            self.block_len.append(block_len)
            self.block_stride.append(block_stride)

        self.group_kv_caches_base_addr: dict[int, list[int]] = {}
        self.group_block_len: dict[int, list[int]] = {}
        self.group_block_stride: dict[int, list[int]] = {}
        self.group_layer_cache_entry_offsets: dict[int, list[int]] = {}
        self.group_cache_entry_segments: dict[int, list[tuple[int, int]]] = {}
        self.kv_caches = kv_caches
        if self.use_kvpp:
            self.physical_layer_to_group_layers.clear()
            owners = map_kvpp_layers_to_owners(self.vllm_config, kv_caches.keys())
            kv_caches = {
                name: caches for name, caches in kv_caches.items() if owners.get(name) in (None, self.kvpp_rank)
            }
            self.kv_caches = kv_caches
        self.group_kv_cache_families: dict[int, str] = {
            group_id: get_group_cache_family(self.kv_cache_group_families, group_id)
            for group_id in range(self.num_kv_cache_groups)
        }
        self.group_num_layers: dict[int, int] = {}

        logger.info(
            "Registering KV_Caches. use_mla: %s, use_sparse: %s, shape %s",
            self.use_mla,
            self.use_sparse,
            first_kv_cache.shape,
        )

        self.kv_caches_base_addr = []

        registered_regions: dict[int, tuple[int, int]] = {}
        registration_caches = (
            {**kv_caches, **draft_caches} if getattr(self, "dspark_prefix_cache", None) is not None else kv_caches
        )
        for cache_or_caches in registration_caches.values():
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

        self._align_kv_ptrs(registered_regions)
        ptrs = [start for start, _ in registered_regions.values()]
        lengths = [end - start for start, end in registered_regions.values()]

        if self.kv_cache_config is not None and self.use_hybrid:
            for group_id, group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
                layer_names = [name for name in group_spec.layer_names if name not in draft_names]
                if self.use_kvpp:
                    layer_names = [name for name in layer_names if name in kv_caches]
                self._infer_cache_group_metadata(group_id, layer_names)
        else:
            self._infer_cache_group_metadata(0, list(kv_caches.keys()))

        # group_num_layers is computed from the actual kv_caches dict which
        # includes ALL attention layers (main + MTP). For single-group models,
        # sum(group_num_layers.values()) equals the physical layer count
        # (including MTP). For multi-group models, it counts (group, layer)
        # pairs which is NOT the physical layer count — keep the original
        # num_layers (physical layers) in that case.
        original_num_layers = self.num_layers
        new_num_layers = sum(self.group_num_layers.values())
        if not self.use_kvpp and self.num_kv_cache_groups == 1 and new_num_layers != original_num_layers:
            self.num_layers = new_num_layers
            logger.info(
                "KVPoolWorker: updated num_layers %d -> %d (includes MTP/spec-decode draft layers).",
                original_num_layers,
                self.num_layers,
            )

        self.page_size_bytes = sum(self.block_len)
        self.token_database.set_group_buffers(
            self.group_kv_caches_base_addr,
            self.group_block_len,
            self.group_block_stride,
            cache_role="kv",
            group_cache_families=self.group_kv_cache_families,
            group_num_layers=self.group_num_layers,
            group_layer_cache_entry_offsets=self.group_layer_cache_entry_offsets,
            group_cache_entry_segments=self.group_cache_entry_segments,
        )

        if self.tp_mismatch:
            first_cache = self._as_cache_tuple(next(iter(kv_caches.values())))[0]
            self.elem_size = first_cache.element_size()
            self.head_dim = first_cache.shape[-1]
            # block_len[0] = block_size * num_kv_head_per_local_rank * head_dim * elem_size
            self.per_token_bytes = self.group_block_len[0][0] // self.block_size
            self.sub_size_bytes = self.effective_heads_per_rank * self.head_dim * self.elem_size
            logger.info(
                "TP mismatch strided I/O: per_token_bytes=%d, sub_size_bytes=%d",
                self.per_token_bytes,
                self.sub_size_bytes,
            )

        # Register buffers before starting the transfer threads.
        self.m_store.register_buffer(ptrs, lengths)
        self._start_kv_transfer_threads()

    def start_load_kv(self, metadata: AscendConnectorMetadata):
        self.current_layer = 0
        logger.debug("KV pool worker start_load_kv requests=%d", len(metadata.requests))
        if len(metadata.requests) == 0:
            return
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
                token_len = load_spec.kvpool_cached_tokens + 1
            else:
                token_len = load_spec.kvpool_cached_tokens
            load_spec.token_len = token_len
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
            if self.load_async:
                self.kv_recv_thread.add_request(  # type: ignore[union-attr]
                    request,
                )
                continue

            if self.tp_mismatch:
                # TP mismatch is restricted to non-hybrid, single-group KV.
                group_block_size = self.grouped_block_size[0]
                mask_num = load_spec.vllm_cached_tokens // group_block_size * group_block_size
                loaded = self._load_kv_tp_mismatch(
                    request.block_hashes,
                    request.block_ids_by_group[0],
                    token_len,
                    mask_num,
                )
                if loaded is False:
                    self._record_failed_request(request)
                continue

            addr_list = []
            size_list = []
            key_list = []
            block_id_list: list[int] = []
            load_masks = self.token_database.load_mask(request.block_hashes, token_len)
            for group_id in load_group_ids:
                if group_id >= len(request.block_ids_by_group):
                    continue
                block_ids = request.block_ids_by_group[group_id]
                group_block_size = self.grouped_block_size[group_id]
                mask_num = load_spec.vllm_cached_tokens // group_block_size * group_block_size
                skip_null = group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]

                def chunk_filter(start: int, group_id=group_id, load_masks=load_masks) -> bool:
                    return self.token_database.mask_allows_chunk(load_masks, group_id, start)

                for (
                    start,
                    end,
                    key,
                    _block_hash,
                    block_id,
                ) in self.token_database.process_token_key_strings_with_block_ids(
                    token_len,
                    request.block_hashes,
                    block_ids,
                    mask_num,
                    kv_cache_group_id=group_id,
                    skip_null_blocks=skip_null,
                    chunk_filter=chunk_filter,
                ):
                    addr, size, block_id = self.token_database.prepare_value(
                        start,
                        end,
                        block_ids,
                        kv_cache_group_id=group_id,
                        block_id=block_id,
                    )
                    key_list.append(key)
                    addr_list.append(addr)
                    size_list.append(size)
                    block_id_list.append(block_id)
            if not key_list:
                continue
            key_list_c = _circular_shift(key_list, self.tp_rank % len(key_list))
            addr_list_c = _circular_shift(addr_list, self.tp_rank % len(addr_list))
            size_list_c = _circular_shift(size_list, self.tp_rank % len(size_list))
            block_id_list_c = _circular_shift(block_id_list, self.tp_rank % len(block_id_list))
            logger.debug(
                "KV pool worker calls backend get request=%s token_len=%d groups=%s keys=%d sample_keys=%s",
                request.req_id,
                token_len,
                load_group_ids,
                len(key_list_c),
                key_list_c[:3],
            )
            load_get_start = time.perf_counter()
            ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
            self._record_kv_connector_operation(
                "load_get",
                time.perf_counter() - load_get_start,
                len(key_list_c),
            )
            if self.backend_name == "memcache" and (
                ret is None or len(ret) != len(key_list_c) or any(r != 0 for r in ret)
            ):
                self._record_failed_request(request)
                continue
            if ret is not None and any(r != 0 for r in ret):
                missing_block_ids = record_failed_blocks(
                    block_id_list_c,
                    ret,
                )
                if len(request.block_ids_by_group) == 1:
                    self._invalid_block_ids.update(missing_block_ids)
                elif missing_block_ids:
                    logger.error(
                        "KV load failed for hybrid request %s. "
                        "Skip invalid-block fallback to avoid scheduler crash. "
                        "failed_blocks=%s",
                        request.req_id,
                        missing_block_ids,
                    )
            elif ret is None:
                missing_block_ids = record_failed_blocks(
                    block_id_list_c,
                    [1] * len(block_id_list_c),
                )
                if len(request.block_ids_by_group) == 1:
                    self._invalid_block_ids.update(missing_block_ids)
                elif missing_block_ids:
                    logger.error(
                        "KV load failed for hybrid request %s. "
                        "Skip invalid-block fallback to avoid scheduler crash. "
                        "failed_blocks=%s",
                        request.req_id,
                        missing_block_ids,
                    )
            logger.debug(
                "KV pool worker backend get returned request=%s token_len=%d groups=%s keys=%d",
                request.req_id,
                token_len,
                load_group_ids,
                len(key_list_c),
            )

    def _record_kv_connector_operation(self, operation: str, duration_seconds: float, num_keys: int) -> None:
        with self._kv_stats_lock:
            self._kv_stats.record_operation(operation, duration_seconds, num_keys)

    def get_stats(self) -> AscendStoreKVConnectorStats | None:
        with self._kv_stats_lock:
            if self._kv_stats.is_empty():
                return None
            stats = self._kv_stats
            self._kv_stats = AscendStoreKVConnectorStats()
            return stats

    def _record_invalid_blocks_ids(self, block_ids: list[int]) -> None:
        if not block_ids:
            return
        with self._invalid_block_ids_lock:
            self._invalid_block_ids.update(block_ids)

    def _record_failed_request(self, request: ReqMeta) -> None:
        request.can_save = False
        if self.num_kv_cache_groups > 1:
            with self._invalid_block_ids_lock:
                self._failed_requests.add(request.req_id)
        else:
            self._record_invalid_blocks_ids([block_id for block_id in request.block_ids if block_id > 0])
        self._save_disabled.set()

    def get_failed_requests(self) -> set[str]:
        with self._invalid_block_ids_lock:
            failed = self._failed_requests.copy()
            self._failed_requests.clear()
        for thread in (self.kv_send_thread, self.kv_recv_thread):
            if thread is not None:
                failed.update(thread.get_and_clear_failed_requests())
        return failed

    def get_block_ids_with_load_errors(self) -> set[int]:
        with self._invalid_block_ids_lock:
            invalid_blocks = self._invalid_block_ids.copy()
            self._invalid_block_ids.clear()
        return invalid_blocks

    def handle_released_saves(self, released_req_ids: set[str] | None) -> None:
        """Fence only the saves of requests whose blocks this step may reuse.

        Finished and preempted requests' blocks are freed immediately by the
        scheduler; before this step's forward can reallocate those blocks,
        every queued save chunk of exactly those requests must have drained.
        Requests whose blocks were already released in earlier steps were
        drained by that step's fence and never queue new puts, so they need
        no further waiting here.
        """
        send_thread = self.kv_send_thread
        if send_thread is None:
            return
        send_thread.raise_if_failed()
        if isinstance(send_thread, KVCacheStoreSendingThread):
            send_thread.wait_for_requests_saved(released_req_ids)
        # Layerwise saves are already synchronized at each step's final layer.

    def wait_for_save(self, connector_metadata: AscendConnectorMetadata) -> None:
        current_event = None
        assert self.kv_send_thread is not None
        send_thread = self.kv_send_thread
        if not isinstance(send_thread, KVCacheStoreSendingThread):
            raise TypeError(
                f"Non-layerwise KV save requires KVCacheStoreSendingThread, but got {type(send_thread).__name__}"
            )
        requests: list[ReqMeta] = []

        for request in connector_metadata.requests:
            can_save = request.can_save
            if can_save is None or not can_save:
                continue
            if current_event is None:
                current_event = torch.npu.Event()
                current_event.record()
            request.skip_null_blocks_by_group = self.group_uses_align_state
            request.current_event = current_event
            requests.append(request)

        if not requests:
            return

        self._previous_save_batch = send_thread.add_save_batch(requests)

    def _make_sub_key_str(self, base_key: str, effective_rank: int) -> str:
        """Rewrite ``@head_or_tp_rank:<local>`` in serialized ``base_key`` to ``<effective_rank>``.

        Under TP mismatch, both sides address the pool at the effective_tp_size
        namespace rather than the local TP rank.
        """
        return self._replace_key_field(base_key, "head_or_tp_rank", effective_rank)

    def _build_strided_addrs(self, block_id: int, token_count: int, sub_idx: int) -> tuple[list[int], list[int]]:
        """Build per-token (addr, size) pairs into local KV cache memory for one
        sub-key inside one block.

        KV cache layout: [num_block, block_size, num_kv_head_per_local_rank, head_dim].
        Heads of consecutive tokens are interleaved with token position, so a
        sub-slice of heads requires one transfer per token. Block stepping uses
        ``block_stride`` because the kernel may pad between blocks.
        """
        head_offset_bytes = sub_idx * self.sub_size_bytes
        addrs: list[int] = []
        sizes: list[int] = []
        # tp_mismatch is restricted to a single dense KV group -> group 0.
        group_addrs = self.group_kv_caches_base_addr[0]
        group_block_len = self.group_block_len[0]
        group_block_stride = self.group_block_stride[0]
        for base_addr, entry_block_len, entry_block_stride in zip(
            group_addrs, group_block_len, group_block_stride, strict=True
        ):
            entry_per_token_bytes = entry_block_len // self.block_size
            block_base = base_addr + block_id * entry_block_stride
            for t in range(token_count):
                addrs.append(block_base + t * entry_per_token_bytes + head_offset_bytes)
                sizes.append(self.sub_size_bytes)
        return addrs, sizes

    def _build_tp_mismatch_keys_and_addrs(
        self,
        block_hashes: list,
        block_ids: list[int],
        token_len: int,
        mask_num: int = 0,
        shard_rank: int | None = None,
        shard_size: int | None = None,
        save_start_token: int = 0,
    ) -> tuple[list[str], list[list[int]], list[list[int]], list[int]]:
        """Walk chunks x sub-keys; emit (keys, addrs, sizes, block_ids) for backend put/get.

        Each key represents one (chunk, sub_idx) pair. Its addrs/sizes cover all
        layer-entries x all tokens in the chunk, addressed at the head-slice
        owned by sub_idx within this rank's local cache.
        """
        all_keys: list[str] = []
        all_addrs: list[list[int]] = []
        all_sizes: list[list[int]] = []
        all_block_ids: list[int] = []
        for start, end, base_key, _block_hash, block_id in self.token_database.process_token_key_strings_with_block_ids(
            token_len,
            block_hashes,
            block_ids,
            mask_num=mask_num,
            shard_rank=shard_rank,
            shard_size=shard_size,
        ):
            if end <= save_start_token:
                continue
            token_count = end - start
            for sub_idx in range(self.num_sub_keys):
                effective_rank = self.tp_rank * self.num_sub_keys + sub_idx
                addrs, sizes = self._build_strided_addrs(block_id, token_count, sub_idx)
                all_keys.append(self._make_sub_key_str(base_key, effective_rank))
                all_addrs.append(addrs)
                all_sizes.append(sizes)
                all_block_ids.append(block_id)
        return all_keys, all_addrs, all_sizes, all_block_ids

    def _load_kv_tp_mismatch(
        self,
        block_hashes: list,
        block_ids: list[int],
        token_len: int,
        mask_num: int,
    ) -> bool | None:
        keys, addrs, sizes, key_block_ids = self._build_tp_mismatch_keys_and_addrs(
            block_hashes, block_ids, token_len, mask_num
        )
        if not keys:
            return None
        offset = self.tp_rank % len(keys)
        keys_c = keys[offset:] + keys[:offset]
        addrs_c = addrs[offset:] + addrs[:offset]
        sizes_c = sizes[offset:] + sizes[:offset]
        block_ids_c = key_block_ids[offset:] + key_block_ids[:offset]
        logger.debug(
            "KV pool worker tp_mismatch get keys=%d sample_keys=%s",
            len(keys_c),
            keys_c[:3],
        )
        load_get_start = time.perf_counter()
        ret = self.m_store.get(keys_c, addrs_c, sizes_c)
        self._record_kv_connector_operation(
            "load_get",
            time.perf_counter() - load_get_start,
            len(keys_c),
        )
        if self.backend_name == "memcache" and (ret is None or len(ret) != len(keys_c) or any(r != 0 for r in ret)):
            return False
        if ret is not None and any(r != 0 for r in ret):
            missing_block_ids = record_failed_blocks(block_ids_c, ret)
            with self._invalid_block_ids_lock:
                self._invalid_block_ids.update(missing_block_ids)
        elif ret is None:
            missing_block_ids = record_failed_blocks(block_ids_c, [1] * len(block_ids_c))
            with self._invalid_block_ids_lock:
                self._invalid_block_ids.update(missing_block_ids)
        logger.debug(
            "KV pool worker tp_mismatch get returned keys=%d",
            len(keys_c),
        )
        return None

    def _store_kv_tp_mismatch(self, req_meta: ReqMeta) -> bool:
        send_thread = self.kv_send_thread
        if send_thread is None:
            logger.error("Cannot save TP-mismatch KV without a sending thread")
            return False
        assert isinstance(send_thread, KVCacheStoreSendingThread)
        req_id = req_meta.req_id
        if not send_thread.is_live_store_job(req_meta):
            return False
        token_len = req_meta.token_len_chunk
        block_ids = req_meta.block_ids_by_group[0]
        save_start_token = send_thread.get_saved_offset(req_id)
        keys, addrs, sizes, _ = self._build_tp_mismatch_keys_and_addrs(
            req_meta.block_hashes,
            block_ids,
            token_len,
            mask_num=0,
            shard_rank=self.pcp_rank,
            shard_size=self.pcp_size,
            save_start_token=save_start_token,
        )
        if not keys:
            return True
        logical_key_count = len(keys)
        partitions = getattr(self.token_database, "partitions", None)
        if self.kv_role == "kv_consumer" and partitions is not None and len(partitions) > 1:
            # TP-mismatch keys still represent logical Decode blocks here.
            # Expand them before lookup so a partial Prefill-PP write retries
            # only the missing physical partition rather than treating rank 0
            # as proof that the whole block exists.
            keys, addrs, sizes = send_thread._decode_adaptor_prefill_pp(
                keys,
                addrs,
                sizes,
            )
        if not keys or len(keys) % logical_key_count != 0:
            logger.error("Prefill-PP adaptor expanded %s TP keys to an invalid count %s", logical_key_count, len(keys))
            return False
        physical_keys_per_logical_key = len(keys) // logical_key_count
        physical_keys_per_block = self.num_sub_keys * physical_keys_per_logical_key
        exists_states = send_thread.lookup(keys)
        if len(exists_states) != len(keys):
            logger.error(
                "Invalid TP-mismatch lookup result count for request %s: expected=%d actual=%d",
                req_id,
                len(keys),
                len(exists_states),
            )
            return False
        missing_indices = [i for i, exists in enumerate(exists_states) if not exists]
        if not missing_indices:
            return True
        missing_keys = [keys[i] for i in missing_indices]
        missing_addrs = [addrs[i] for i in missing_indices]
        missing_sizes = [sizes[i] for i in missing_indices]
        if req_meta.current_event is not None:
            req_meta.current_event.synchronize()
        logger.debug(
            "KV pool worker tp_mismatch put req=%s keys=%d sample_keys=%s",
            req_id,
            len(missing_keys),
            missing_keys[:3],
        )
        put_result = self.m_store.put(missing_keys, missing_addrs, missing_sizes)
        if isinstance(put_result, list):
            if len(put_result) != len(missing_keys):
                logger.error(
                    "KV store backend returned %s results for %s TP-mismatch keys", len(put_result), len(missing_keys)
                )
                return False
            put_failed = not all(put_result)
        else:
            put_failed = put_result is False

        if self.enable_kv_events:
            successful_missing_indices = (
                {physical_idx for physical_idx, succeeded in zip(missing_indices, put_result, strict=True) if succeeded}
                if isinstance(put_result, list)
                else (set() if put_failed else set(missing_indices))
            )
            completed_block_indices = {
                block_idx
                for block_idx in {index // physical_keys_per_block for index in missing_indices}
                if all(
                    exists_states[index] or index in successful_missing_indices
                    for index in range(
                        block_idx * physical_keys_per_block,
                        min((block_idx + 1) * physical_keys_per_block, len(keys)),
                    )
                )
            }
            event_block_size = (
                req_meta.original_block_size[0]
                if isinstance(req_meta.original_block_size, list)
                else req_meta.original_block_size
            )
            stored_events: list[BlockStored] = []
            group_hashes = get_block_hashes(
                req_meta.block_hashes,
                self.block_size,
                self.token_database.hash_block_size,
            )
            # Match the block selection and ordering used to build PUT keys.
            # PCP ranks own disjoint blocks, including when retrying a suffix.
            store_chunks = self.token_database.process_token_key_strings_with_block_ids(
                token_len,
                req_meta.block_hashes,
                block_ids,
                mask_num=0,
                shard_rank=self.pcp_rank,
                shard_size=self.pcp_size,
            )
            for relative_idx, (start, end, _base_key, _hash, _block_id) in enumerate(
                chunk for chunk in store_chunks if chunk[1] > save_start_token
            ):
                if relative_idx not in completed_block_indices:
                    continue
                block_idx = start // self.block_size
                if block_idx >= len(group_hashes):
                    continue
                block_hash = maybe_convert_block_hash(group_hashes[block_idx])
                parent_hash = maybe_convert_block_hash(group_hashes[block_idx - 1]) if block_idx > 0 else None
                token_ids = send_thread.get_event_token_ids(req_meta, start, end)
                stored_events.append(
                    BlockStored(
                        block_hashes=[block_hash],
                        parent_block_hash=parent_hash,
                        token_ids=token_ids,
                        block_size=event_block_size,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                    )
                )
            if stored_events:
                send_thread.update_kv_event(stored_events)
        if put_failed:
            qualifier = "partially " if isinstance(put_result, list) else ""
            logger.error("KV store backend %sfailed to put TP-mismatch request %s", qualifier, req_id)
            return False
        return True

    def get_finished(self, finished_req_ids: set[str], meta: AscendConnectorMetadata) -> tuple[set[str], set[str]]:
        if self.kv_send_thread is not None:
            send_thread = self.kv_send_thread
            for req_id in meta.preempted_req_ids:
                if isinstance(send_thread, (KVCacheStoreSendingThread, KVCacheStoreLayerSendingThread)):
                    send_thread.delete_finished_stored_request(req_id)
                if isinstance(send_thread, KVCacheStoreSendingThread):
                    send_thread.reset_saved_request(req_id)
                    self._retired_store_req_ids.discard(req_id)
            if isinstance(send_thread, KVCacheStoreSendingThread):
                for req_id in finished_req_ids:
                    if send_thread.is_stored_request(req_id):
                        self._retired_store_req_ids.add(req_id)
                    else:
                        send_thread.reset_saved_request(req_id)
                        self._retired_store_req_ids.discard(req_id)
                for req_id in self._retired_store_req_ids.copy():
                    if not send_thread.is_stored_request(req_id):
                        send_thread.reset_saved_request(req_id)
                        self._retired_store_req_ids.discard(req_id)
            self.kv_send_thread.discard_finished_requests(meta.preempted_req_ids)
            # Saves complete synchronously in wait_for_save(), so the scheduler
            # never waits for a request-level finished_sending notification.
            self.kv_send_thread.get_and_clear_finished_requests()
            done_sending: set[str] = set()
        else:
            done_sending = set()

        done_recving = set()
        if self.kv_recv_thread is not None:
            self.kv_recv_thread.discard_finished_requests(meta.preempted_req_ids)
            if self.load_async and not self.use_layerwise:
                done_recving = self.kv_recv_thread.get_and_clear_finished_requests(meta.loading_req_ids)

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

    def reset_store(self) -> None:
        """Drain queued puts, then wipe the backend.

        Join first so a put already on the send thread cannot land after
        ``remove_all`` and repopulate the pool with the previous weights.
        """
        if self.kv_send_thread is not None:
            self.kv_send_thread.request_queue.join()
        self.m_store.remove_all()

    def _build_lookup_keys(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        group_id: int,
    ) -> tuple[list[str], list[int], list[int]]:
        keys: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        for start, end, key_string, _ in self.token_database.process_token_key_strings(
            token_len, block_hashes, kv_cache_group_id=group_id
        ):
            keys.append(key_string)
            starts.append(start)
            ends.append(end)
        return keys, starts, ends

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
            coordinator_hit = (
                None
                if use_layerwise
                else self._lookup_with_coordinator(
                    token_len,
                    block_hashes,
                    kv_cache_group_ids,
                    include_all_ranks=False,
                )
            )
            if coordinator_hit is not None:
                return coordinator_hit
            for group_id in kv_cache_group_ids:
                keys, starts, ends = self._build_lookup_keys(token_len, block_hashes, group_id)

                if not keys:
                    hits.append(0)
                    continue

                res = self.m_store.exists(keys)  # type: ignore[assignment]

                if use_layerwise:
                    res = self.check_all_layers_exists(
                        res,
                        getattr(self, "layerwise_key_layers", 0) or self.num_layers,
                    )
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
                    hit_end = ends[-1]
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
            logger.error(
                "Remote connection failed in get_common_prefix_length. type=%s, error=%s. "
                "Check network and remote store.",
                type(e).__name__,
                e,
            )
            return 0
        return min(hits) if hits else 0

    def _get_group_num_kv_heads(self, group_id: int) -> int:
        if self.use_mla or self.use_sparse:
            return 1
        if group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]:
            return 1
        return self.num_kv_head

    def get_group_tp_size(self, kv_cache_group_id: int):
        if self.use_kvpp:
            return self.tp_size * self.pcp_size
        if self.tp_mismatch:
            return self.effective_tp_size
        if self.group_uses_align_state[kv_cache_group_id]:
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

    def _expand_lookup_keys_by_rank(self, keys: list[str], group_id: int) -> list[str]:
        # PCP replicas share keys; expand only the physical KV partitions.
        expanded: list[str] = []
        num_head_or_tp_ranks = self.get_group_tp_size(group_id)
        # Keep each rank shard's block/layer keys contiguous to match
        # lookup_scheduler()'s [rank_shard][block] result slicing.
        for pp_rank in range(self.pp_size):
            for dcp_rank in range(self.dcp_size):
                for head_or_tp_rank in range(num_head_or_tp_ranks):
                    for key in keys:
                        rank_key = self._replace_key_field(key, "dcp", dcp_rank)
                        rank_key = self._replace_key_field(rank_key, "head_or_tp_rank", head_or_tp_rank)
                        expanded.append(self._replace_key_field(rank_key, "pp_rank", pp_rank))
        return expanded

    def _expand_lookup_key_variants(self, key: str, group_id: int, include_all_ranks: bool) -> list[str]:
        if not include_all_ranks:
            return [key]
        return self._expand_lookup_keys_by_rank([key], group_id)

    def _lookup_with_coordinator(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int],
        include_all_ranks: bool,
        hbm_hit_tokens: int = 0,
    ) -> int | None:
        if self.cache_coordinator is None:
            return None
        if sorted(kv_cache_group_ids) != list(range(self.num_kv_cache_groups)):
            return None
        coordinator = self.cache_coordinator

        def query_group_hits(
            group_id: int,
            group_block_hashes: Sequence[BlockHash | str],
            lookup_mask: Sequence[bool] | None,
        ) -> list[BlockHash | str]:
            group_block_size = coordinator.group_effective_block_sizes[group_id]
            present: list[BlockHash | str] = []
            lookup_start = hbm_hit_tokens // group_block_size * group_block_size

            keys: list[str] = []
            chunk_hashes: list[BlockHash | str] = []
            variant_counts: list[int] = []
            for _, _, key_string, chunk_hash in self.token_database.process_token_key_strings(
                token_len,
                block_hashes,
                mask_num=lookup_start,
                kv_cache_group_id=group_id,
                chunk_mask=lookup_mask,
            ):
                variants = self._expand_lookup_key_variants(key_string, group_id, include_all_ranks)
                keys.extend(variants)
                chunk_hashes.append(chunk_hash)
                variant_counts.append(len(variants))

            if keys:
                res = self.m_store.exists(keys)  # type: ignore[assignment]
                offset = 0
                for chunk_hash, count in zip(chunk_hashes, variant_counts, strict=True):
                    values = res[offset : offset + count]  # type: ignore[index]
                    if values and all(value == 1 for value in values):
                        present.append(chunk_hash)
                    offset += count

            logger.debug(
                "KV pool coordinator lookup group=%d token_len=%d keys=%d exists_chunks=%d sample_keys=%s",
                group_id,
                token_len,
                len(keys),
                len(present),
                keys[:3],
            )
            return present

        return coordinator.find_reachable_hit_tokens(
            block_hashes,
            token_len,
            query_group_hits,
            log_context="KV pool coordinator lookup",
            hbm_hit_tokens=hbm_hit_tokens,
        )

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
            hits: list[list[int]] = []
            max_hit_position = self.max_model_len
            kv_cache_group_ids = kv_cache_group_ids or [0]
            coordinator_hit = (
                None
                if use_layerwise
                else self._lookup_with_coordinator(
                    token_len,
                    block_hashes,
                    kv_cache_group_ids,
                    include_all_ranks=True,
                    hbm_hit_tokens=hbm_hit_tokens,
                )
            )
            if coordinator_hit is not None:
                return coordinator_hit
            for group_id in kv_cache_group_ids:
                keys, starts, ends = self._build_lookup_keys(token_len, block_hashes, group_id)

                if not keys:
                    return 0

                multi_tp_keys = self._expand_lookup_keys_by_rank(keys, group_id)
                num_ranks = len(multi_tp_keys) // len(keys)
                res = self.m_store.exists(multi_tp_keys)  # type: ignore[assignment]
                num_block = len(keys)
                if use_layerwise:
                    res = self.check_all_layers_exists(
                        res,
                        getattr(self, "layerwise_key_layers", 0) or self.num_layers,
                    )
                    num_block = len(keys) // (getattr(self, "layerwise_key_layers", 0) or self.num_layers)
                multi_tp_values = [
                    res[i * num_block : (i + 1) * num_block]  # type: ignore[index]
                    for i in range(num_ranks)
                ]
                logger.debug(
                    "KV pool lookup request token_len=%d group=%d keys=%d multi_tp_keys=%d "
                    "exists_count=%d/%d exists_sample=%s sample_keys=%s",
                    token_len,
                    group_id,
                    len(keys),
                    len(multi_tp_keys),
                    sum(1 for value in res if value == 1),  # type: ignore[union-attr]
                    len(res),
                    list(res[: min(12, len(res))]),  # type: ignore[index]
                    multi_tp_keys[:3],
                )
                if group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]:
                    group_hits = self.find_all_discontinuous_hit_positions(
                        multi_tp_values, ends, num_block, max_hit_position, self.cache_transfer_granularity
                    )
                else:
                    group_hits = self.find_all_continuous_hit_positions(
                        multi_tp_values, ends, num_block, max_hit_position, self.cache_transfer_granularity
                    )
                if not group_hits:
                    return 0
                max_hit_position = min(max_hit_position, group_hits[-1])
                hits.append(group_hits)
                logger.debug(
                    "KV pool scheduler lookup group=%d keys=%d hit=%d token_len=%d",
                    group_id,
                    len(keys),
                    max_hit_position,
                    token_len,
                )
        except Exception as e:
            logger.error(
                "Remote connection failed in lookup. type=%s, error=%s. Check network and remote store.",
                type(e).__name__,
                e,
            )
            return 0
        final_hits = self._max_intersection_hit_position(hits)
        logger.debug(
            "KV pool scheduler lookup final token_len=%d groups=%s hit=%d",
            token_len,
            kv_cache_group_ids,
            final_hits,
        )
        return final_hits

    @staticmethod
    def _max_intersection_hit_position(hits: list[list[int]]) -> int:
        """
        For all attention groups, treat the position of the maximum common hit as the final hit position
        """
        if not hits:
            return 0
        common_elements = set(hits[0]).intersection(*hits[1:])
        if not common_elements:
            return 0
        return max(common_elements)

    def check_all_layers_exists(self, res: list[int], num_layers: int) -> list[int]:
        total_chunks = len(res) // num_layers
        result = []

        for chunk_idx in range(total_chunks):
            start = chunk_idx * num_layers
            end = start + num_layers
            chunk = res[start:end]
            result.append(1 if all(x == 1 for x in chunk) else 0)

        return result

    @staticmethod
    def find_all_discontinuous_hit_positions(
        arr, ends, num_blocks: int, max_hit_position: int, cache_transfer_granularity: int
    ) -> list[int]:
        """
        For mamba attn, there will be some uncached null blocks, we just collect all hit positions,
        and use the last position as final hit position
        """
        hits: list[int] = []
        for i in range(num_blocks):
            if ends[i] > max_hit_position:
                break
            if all(row[i] == 1 for row in arr):
                if ends[i] % cache_transfer_granularity == 0:
                    hits.append(ends[i])
        return hits

    @staticmethod
    def find_all_continuous_hit_positions(
        arr, ends, num_blocks: int, max_hit_position: int, cache_transfer_granularity: int
    ) -> list[int]:
        hits: list[int] = []
        for i in range(num_blocks):
            if ends[i] > max_hit_position:
                break
            if all(row[i] == 1 for row in arr):
                if ends[i] % cache_transfer_granularity == 0:
                    hits.append(ends[i])
            else:
                break
        return hits

    def get_kv_events(self) -> list[BlockStored]:
        if self.enable_kv_events and self.kv_send_thread is not None:
            # collect store kv events form sending thread
            events = self.kv_send_thread.get_kv_events()
            return events
        return []

    def build_connector_worker_meta(self) -> AscendStoreKVConnectorWorkerMetadata | None:
        if isinstance(self.kv_send_thread, KVCacheStoreSendingThread):
            if ce := self.kv_send_thread.get_completed_events():
                return AscendStoreKVConnectorWorkerMetadata(ce)
        return None
