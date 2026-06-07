from __future__ import annotations

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
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.cpu_binding import (
    bind_thread_to_cpus,
    get_cpu_binding_rank,
    get_memcache_client_cpus,
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
    get_cache_family_granularity,
    infer_group_cache_families,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreKeyLayerRecvingThread,
    KVCacheStoreKeyLayerSendingThread,
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
    _circular_shift,
    record_failed_blocks,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_config import (
    get_layerwise_config,
)
from vllm_ascend.memcache_comm_fence import (
    get_attention_compute_start_gate,
    reset_attention_compute_start_gate,
)

_shared_layer_transfer_events: list[threading.Event] | None = None


def get_shared_layer_transfer_events() -> list[threading.Event] | None:
    return _shared_layer_transfer_events


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
        self.kv_cache_config = kv_cache_config
        hf_text_config = getattr(model_config, "hf_text_config", None)
        hf_config = getattr(model_config, "hf_config", hf_text_config)
        self.hf_config = hf_text_config or hf_config
        self.compress_ratios = getattr(hf_text_config, "compress_ratios", None)
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
        self._init_layerwise_config()

    def _init_parallelism_info(self, model_config, parallel_config) -> None:
        self.local_rank = envs.LOCAL_RANK
        self.cpu_binding_rank = get_cpu_binding_rank(self.local_rank, parallel_config)

        self.use_mla = False
        if hasattr(model_config, "use_mla") and isinstance(model_config.use_mla, bool) and model_config.use_mla:
            self.use_mla = True
        self.use_sparse = hasattr(model_config.hf_text_config, "index_topk")

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = (parallel_config.rank // self.tp_size) % self.pp_size

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0

    def _init_kv_transfer_config(self, vllm_config, extra_config, use_layerwise, kv_cache_config) -> None:
        self._extra_config = extra_config
        self.use_layerwise = use_layerwise
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.load_async = extra_config.get("load_async", False)
        self._invalid_block_ids: set[int] = set()
        self._invalid_block_ids_lock = threading.Lock()
        self.consumer_is_to_put = extra_config.get("consumer_is_to_put", False)
        self.backend = extra_config.get("backend", "mooncake")
        self.backend_name = self.backend.lower()
        self.use_gva_layerwise = self.use_layerwise and self.backend_name == "memcache"
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
        self.h2d_stagger_us = int(extra_config.get("h2d_stagger_us", 0))
        self.h2d_stagger_group_size = int(extra_config.get("h2d_stagger_group_size", 0))
        self.h2d_stagger_dynamic_addrs_per_us = int(extra_config.get("h2d_stagger_dynamic_addrs_per_us", 0))
        self.h2d_stagger_max_us = int(extra_config.get("h2d_stagger_max_us", 0))
        self.layerwise_max_transfer_blocks = int(extra_config.get("layerwise_max_transfer_blocks", 0))
        self.layerwise_max_transfer_bytes = int(extra_config.get("layerwise_max_transfer_bytes", 0))

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
        self.my_key_index = (
            self.pcp_rank * self.dcp_size * (self.tp_size // self.put_step)
            + self.dcp_rank * (self.tp_size // self.put_step)
            + self.head_or_tp_rank
        )
        self.num_ranks_per_layer = self.pcp_size * self.dcp_size * (self.tp_size // self.put_step)

    def _init_metadata(self, model_config, vllm_config, extra_config) -> None:
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

        self.token_database = ChunkedTokenDatabase(
            self.metadata,
            self.grouped_block_size,
            partitions,
            use_hybrid=self.use_hybrid,
            hash_block_size=self.hash_block_size,
        )

    def _init_backend(self, parallel_config, extra_config) -> None:
        backend = backend_map.get(self.backend.lower())
        assert backend is not None
        backend_path = backend.get("path")
        backend_name = backend.get("name")
        assert backend_path is not None and backend_name is not None
        backend_module = importlib.import_module(backend_path)
        real_backend = getattr(backend_module, backend_name)

        if self.backend.lower() == "memcache":
            memcache_client_cpus = extra_config.get("memcache_client_cpus")
            if memcache_client_cpus is None:
                try:
                    memcache_client_cpus = get_memcache_client_cpus(self.cpu_binding_rank)
                except Exception as err:
                    logger.warning(
                        "Failed to get MemCache client CPUs for rank%d: %s",
                        self.cpu_binding_rank,
                        err,
                    )
            self.m_store = real_backend(  # type: ignore[misc]
                parallel_config,
                memcache_client_cpus,
                lazy_init=True,
            )
        else:
            backend_kwargs = {}
            if self.backend.lower() == "mooncake":
                backend_kwargs["lazy_init"] = self.use_compress
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
        self._registered_buffer_ptrs: list[int] = []
        self._registered_buffer_lengths: list[int] = []
        self._kv_caches_registered = False
        self._backend_initialized = False
        self._buffers_registered = False
        self._transfer_threads_started = False

    def _init_layerwise_config(self) -> None:
        self.layer_load_tasks: list[list[LayerTransferTask]] = [[] for i in range(self.num_layers)]
        self.layer_save_tasks: list[list[LayerTransferTask]] = [[] for i in range(self.num_layers)]
        self.layer_load_finished_events = None
        self.layer_save_finished_events = None
        self.layer_transfer_finished_events = None

        self.kv_transfer_thread_cpus: list[int] = []
        if get_ascend_config().enable_cpu_binding:
            try:
                self.kv_transfer_thread_cpus = get_memcache_client_cpus(self.cpu_binding_rank)
            except Exception as err:
                logger.warning(
                    "Failed to get MemCache CPUs for KV transfer threads in rank%d: %s",
                    self.cpu_binding_rank,
                    err,
                )

        self.next_layer_to_submit = 0
        layerwise_config = get_layerwise_config(self.num_layers, self._extra_config)
        self.layerwise_offload = layerwise_config.has_layer_reuse
        self.num_prefetch_layers = layerwise_config.num_prefetch_layers
        self.independent_layers = layerwise_config.independent_layers
        self.prefetch_layer_map = layerwise_config.prefetch_layer_map
        self.sync_save_events = None

        if self.use_gva_layerwise and self.layerwise_offload and self.kv_role == "kv_both":
            self.layer_transfer_finished_events = [threading.Event() for _ in range(self.num_layers)]
            globals()["_shared_layer_transfer_events"] = self.layer_transfer_finished_events

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
        if not self.use_gva_layerwise:
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

    def init_backend(self) -> None:
        if not self._backend_initialized:
            if hasattr(self.m_store, "init_store"):
                self.m_store.init_store()
            self._backend_initialized = True
        if not self._kv_caches_registered:
            return
        if not self._buffers_registered:
            self.m_store.register_buffer(
                self._registered_buffer_ptrs,
                self._registered_buffer_lengths,
            )
            self._buffers_registered = True
        self._start_kv_transfer_threads()

    def _start_kv_transfer_threads(self) -> None:
        if self._transfer_threads_started:
            return

        if self.use_layerwise:
            self.get_event = threading.Event()
            self.layer_load_finished_events = [threading.Event() for i in range(self.num_layers)]
            self.layer_save_finished_events = [threading.Event() for i in range(self.num_layers)]
            self.sync_save_events = [torch.npu.Event() for i in range(self.num_layers)]
            if self.use_gva_layerwise and self.kv_role in ["kv_producer", "kv_both"]:
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
                ready_event_sending.wait()
                self._bind_kv_transfer_thread(
                    self.kv_send_thread,
                    0,
                    "layerwise send",
                )
            elif self.kv_role in ["kv_producer", "kv_both"]:
                ready_event_sending = threading.Event()
                self.kv_send_thread = KVCacheStoreKeyLayerSendingThread(
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
                )
                self.kv_send_thread.start()
                ready_event_sending.wait()
            ready_event = threading.Event()
            if self.use_gva_layerwise:
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
            else:
                self.kv_recv_thread = KVCacheStoreKeyLayerRecvingThread(
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
            if self.use_gva_layerwise:
                self._bind_kv_transfer_thread(
                    self.kv_recv_thread,
                    1,
                    "layerwise recv",
                )
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
                ready_event_sending.wait()
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
        self._transfer_threads_started = True

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

        self.page_size_bytes = sum(self.block_len)
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
        self._registered_buffer_ptrs = ptrs
        self._registered_buffer_lengths = lengths
        self._kv_caches_registered = True
        if self._backend_initialized:
            self.init_backend()

    def start_load_kv(self, metadata: AscendConnectorMetadata):
        self.current_layer = 0
        self.layerwise_retrievers = []
        if self.use_layerwise:
            self.next_layer_to_submit = 0
            reset_attention_compute_start_gate()
        logger.debug("KV pool worker start_load_kv requests=%d", len(metadata.requests))
        if len(metadata.requests) == 0:
            return
        if self.use_layerwise:
            self.process_layer_data(metadata.requests)
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
            if self.load_async:
                self.kv_recv_thread.add_request(  # type: ignore[union-attr]
                    request,
                )
                continue

            addr_list = []
            size_list = []
            key_list = []
            block_id_list: list[int] = []
            for group_id in load_group_ids:
                block_ids = request.block_ids_by_group[group_id]
                group_block_size = self.grouped_block_size[group_id]
                mask_num = request.load_spec.vllm_cached_tokens // group_block_size * group_block_size
                skip_null = group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]
                for start, end, key, _ in self.token_database.process_tokens_with_block_ids(
                    token_len,
                    request.block_hashes,
                    block_ids,
                    mask_num,
                    kv_cache_group_id=group_id,
                    skip_null_blocks=skip_null,
                ):
                    addr, size, block_id = self.token_database.prepare_value(
                        start,
                        end,
                        block_ids,
                        kv_cache_group_id=group_id,
                    )
                    key_list.append(key.to_string())
                    addr_list.append(addr)
                    size_list.append(size)
                    block_id_list.append(block_id)
            if not key_list:
                continue
            key_list_c = _circular_shift(key_list, self.tp_rank % len(key_list))
            addr_list_c = _circular_shift(addr_list, self.tp_rank % len(addr_list))
            size_list_c = _circular_shift(size_list, self.tp_rank % len(size_list))
            block_id_list_c = _circular_shift(block_id_list, self.tp_rank % len(block_id_list))
            ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
            if ret is not None and any(r != 0 for r in ret):
                missing_block_ids = record_failed_blocks(block_id_list_c, ret)
                self._invalid_block_ids.update(missing_block_ids)
            elif ret is None:
                missing_block_ids = record_failed_blocks(block_id_list_c, [1] * len(block_id_list_c))
                self._invalid_block_ids.update(missing_block_ids)

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
                cached_tokens > 0 and cached_tokens % self.block_size == 0 and full_blocks < cached_full_blocks
            )
            if request.last_block_gva is not None and (
                cached_tokens % self.block_size != 0 or needs_last_block_at_boundary
            ):
                partial_block_index = (
                    cached_full_blocks if cached_tokens % self.block_size != 0 else cached_full_blocks - 1
                )
            else:
                partial_block_index = None
            if partial_block_index is not None and partial_block_index < load_start_block:
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

    def _build_shared_save_data(self) -> None:
        """Build shared block data once and attach to all layer save tasks.

        For GVA path (KVCacheStoreLayerSendingThread): pre-computes
        SharedBlockData via LayerBatchBuilder.build_shared().

        For Key path (KVCacheStoreKeyLayerSendingThread): pre-computes
        cached process_tokens via build_cached_process_tokens().
        """
        # Find the first non-empty layer task (all have identical block_ranges)
        first_task = None
        for layer_id in range(self.num_layers):
            if self.layer_save_tasks[layer_id]:
                first_task = self.layer_save_tasks[layer_id][0]
                break
        if first_task is None:
            return

        if isinstance(self.kv_send_thread, KVCacheStoreLayerSendingThread):
            shared = self.kv_send_thread.build_shared_data(first_task)
            if shared is not None:
                for layer_id in range(self.num_layers):
                    for task in self.layer_save_tasks[layer_id]:
                        task.shared_block_data = shared
        elif isinstance(self.kv_send_thread, KVCacheStoreKeyLayerSendingThread):
            cached = self.kv_send_thread.build_cached_process_tokens(first_task)
            if cached is not None:
                for layer_id in range(self.num_layers):
                    for task in self.layer_save_tasks[layer_id]:
                        task.cached_process_tokens = cached

    def process_layer_data(self, requests: list[ReqMeta]) -> None:
        if not requests:
            return
        for layer_id in range(self.num_layers):
            self._process_save_for_layer_batch(requests, layer_id)
        self._build_shared_save_data()
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

        submit_count = self.num_prefetch_layers if self.current_layer == 0 else 1
        submitted_layers = 0
        while submitted_layers < submit_count and self.next_layer_to_submit < self.num_layers:
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
        logger.debug(">>>>>>>>>>>>>>>>>>>> clear load layer %d", self.current_layer)
        self.layer_load_finished_events[self.current_layer].clear()

    def get_block_ids_with_load_errors(self) -> set[int]:
        with self._invalid_block_ids_lock:
            invalid_blocks = self._invalid_block_ids.copy()
            self._invalid_block_ids.clear()
        return invalid_blocks

    def save_kv_layer(self, connector_metadata: AscendConnectorMetadata) -> None:
        self.sync_save_events[self.current_layer].record()
        if self.layer_save_tasks[self.current_layer]:
            for block_range in self.layer_save_tasks[self.current_layer][0].block_ranges:
                self.kv_send_thread.add_stored_request(block_range.request.req_id)
            self.kv_send_thread.add_request(self.layer_save_tasks[self.current_layer])
        elif self.layerwise_offload:
            layer_task = LayerTransferTask(
                layer_id=self.current_layer,
                block_ranges=[],
            )
            self.layer_save_tasks[self.current_layer].append(layer_task)
            self.kv_send_thread.add_request(self.layer_save_tasks[self.current_layer])
        else:
            self.layer_save_finished_events[self.current_layer].set()
        if self.current_layer == self.num_layers - 1:
            is_finish = self.layer_save_finished_events[self.num_layers - 1].wait(timeout=10)
            if not is_finish:
                logger.info("Layerwise %d save wait timed out", self.current_layer)
            for layer_id in range(self.num_layers):
                if self.layer_save_finished_events[layer_id].is_set():
                    logger.debug(">>>>>>>>>>>>>>>>>>>> clear save layer %d", layer_id)
                    self.layer_save_finished_events[layer_id].clear()

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

        for request in connector_metadata.requests:
            can_save = request.can_save
            if can_save is None or not can_save:
                continue

            request.skip_null_blocks_by_group = self.group_uses_align_state
            request.current_event = current_event
            self.kv_send_thread.add_stored_request(  # type: ignore[union-attr]
                request.req_id
            )
            self.kv_send_thread.add_request(  # type: ignore[union-attr]
                request,
            )
            has_save_request = True

        if has_save_request:
            # vLLM expects wait_for_save() to make stores visible before the
            # request is reported as finished. Without this barrier a following
            # identical prompt can lookup before Mooncake put() has completed.
            self.kv_send_thread.request_queue.join()  # type: ignore[union-attr]

    def get_finished(self, finished_req_ids: set[str], meta: AscendConnectorMetadata) -> tuple[set[str], set[str]]:
        if self.kv_send_thread is not None:
            for req_id in meta.preempted_req_ids:
                self.kv_send_thread.delete_finished_stored_request(req_id)
            self.kv_send_thread.discard_finished_requests(meta.preempted_req_ids)
            if self.use_layerwise:
                self.kv_send_thread.get_and_clear_finished_requests()
                done_sending = set()
            else:
                stale_finished_req_ids = finished_req_ids - meta.delayed_free_req_ids
                self.kv_send_thread.discard_finished_requests(stale_finished_req_ids)
                done_sending = self.kv_send_thread.get_and_clear_finished_requests(meta.delayed_free_req_ids)
        else:
            done_sending = set()

        done_recving = set()
        if self.kv_recv_thread is not None:
            self.kv_recv_thread.discard_finished_requests(meta.preempted_req_ids)
            if self.load_async:
                done_recving = self.kv_recv_thread.get_and_clear_finished_requests(meta.loading_req_ids)

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
            kv_cache_group_ids = self._get_lookup_gate_group_ids(kv_cache_group_ids)
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

    def _get_lookup_gate_group_ids(self, kv_cache_group_ids: list[int]) -> list[int]:
        gate_group_ids = [group_id for group_id in kv_cache_group_ids if self._is_lookup_gate_group(group_id)]
        if not gate_group_ids:
            return kv_cache_group_ids
        if len(gate_group_ids) != len(kv_cache_group_ids):
            logger.debug(
                "KV pool lookup gates on groups %s, ignoring non-gate groups from %s",
                gate_group_ids,
                kv_cache_group_ids,
            )
        return gate_group_ids

    def _is_lookup_gate_group(self, group_id: int) -> bool:
        if group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]:
            return False
        cache_family = self._get_group_family(self.kv_cache_group_families, group_id)
        # DeepSeek V4 has a c128 compressed KV group. Its key stream is much
        # sparser than the dense KV groups, so using it as a strict gate makes
        # the whole request report 0 hit even when the loadable groups exist.
        if cache_family == "c128":
            return False
        # The DSV4 c4 group is currently written as a TP-sharded key stream in
        # this connector path. Runtime logs show only 32/128 keys visible for a
        # 16K prompt, so letting it gate the external pool prevents otherwise
        # complete c1 groups from loading. Keep pooling gate/load on complete
        # 128-token c1 KV groups until c4 storage is made fully discoverable.
        if cache_family != "c1":
            return False
        # In the DSV4 hybrid layout, some auxiliary groups use smaller logical
        # block sizes (for example 8/32). The Ascend kernels in this path are
        # fixed to the 128-token KV block shape, so those groups cannot be used
        # as external-pool gates or load targets for the 16K pooling path.
        return self._get_group_block_size(group_id) == self.block_size

    def _get_group_num_kv_heads(self, group_id: int) -> int:
        if self.use_mla or self.use_sparse:
            return 1
        if group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]:
            return 1
        return self.num_kv_head

    def lookup_scheduler(
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
            kv_cache_group_ids = self._get_lookup_gate_group_ids(kv_cache_group_ids)
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
                group_tp_size = min(self.tp_size, self._get_group_num_kv_heads(group_id))
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
