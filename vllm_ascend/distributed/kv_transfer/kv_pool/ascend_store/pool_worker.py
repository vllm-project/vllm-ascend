from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import importlib
import math
import threading
from collections.abc import Generator, Sequence
from dataclasses import dataclass

import torch
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
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
from vllm.v1.worker.utils import extract_layer_index

from vllm_ascend.attention.cache_layout import stable_model_fingerprint

from vllm_ascend.attention.cache_layout import stable_model_fingerprint

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
    ChunkedTokenDatabase,
    aggregate_c128_page_chunks,
    KeyMetadata,
    LayerMultiBlockReqMeta,
    ReqMeta,
    TransferChunkWithBlockId,
    get_block_hashes,
    get_cache_family_granularity,
    infer_group_cache_families,
    build_layerwise_store_key,
    resolve_hybrid_cache_c128_config,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_session import (
    ChunkStoreSession,
    PutBinding,
    PutClaim,
    RequestStoreLease,
    StoreCommitError,
    StorePutRegistry,
    StoreReadLeaseRegistry,
)
from vllm_ascend.distributed.kv_transfer.layer_workspace_fence import RequestChunkKey
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
    record_failed_blocks,
)
from vllm_ascend.distributed.utils import (
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
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


@dataclass(frozen=True)
class _LayerwiseRangeBlock:
    key: str
    start: int
    end: int
    block_id: int
    group_id: int = 0

@dataclass(frozen=True)
class _LayerwiseRangeComponent:
    base_addr: int
    block_len: int
    block_stride: int
    object_offset: int


@dataclass
class _LayerwiseRangeSessionState:
    request: ReqMeta
    session: ChunkStoreSession
    records: list[_LayerwiseRangeBlock]


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
        self.model_config = model_config
        hf_text_config = getattr(model_config, "hf_text_config", None)
        hf_config = getattr(model_config, "hf_config", hf_text_config)
        self.hf_config = hf_text_config or hf_config
        self.compress_ratios = getattr(hf_text_config, "compress_ratios", None)
        self.max_model_len = model_config.max_model_len
        if self.compress_ratios is None:
            self.compress_ratios = getattr(hf_config, "compress_ratios", None)
        self.use_compress = self.compress_ratios is not None
        self.dp_rank = parallel_config.data_parallel_rank
        self.use_mla = False
        if hasattr(model_config, "use_mla") and isinstance(model_config.use_mla, bool) and model_config.use_mla:
            self.use_mla = True
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.use_sparse = hasattr(model_config.hf_text_config, "index_topk")
        self.use_layerwise = use_layerwize
        self.use_layerwise_range = self.use_layerwise and bool(extra_config.get("use_layerwise_range", False))
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
        self._invalid_block_ids: set[int] = set()
        self._invalid_block_ids_lock = threading.Lock()
        self.consumer_is_to_put = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_put", False
        )
        self.backend = vllm_config.kv_transfer_config.kv_connector_extra_config.get("backend", "mooncake")
        self.use_hybrid = self._uses_hybrid_kv_cache(vllm_config, kv_cache_config)
        self.use_mamba = self._uses_mamba_kv_cache(self.use_hybrid, kv_cache_config)
        if self.use_layerwise_range and self.backend.lower() != "mooncake":
            raise NotImplementedError(
                "Layerwise Store range sessions currently require the Mooncake backend"
            )
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
        discard_partial_chunks = vllm_config.kv_transfer_config.get_from_extra_config(
            "discard_partial_chunks", True
        )
        self.hybrid_cache_c128_config = resolve_hybrid_cache_c128_config(
            vllm_config,
            use_layerwise=self.use_layerwise,
            group_block_sizes=self.grouped_block_size,
            group_cache_families=self.kv_cache_group_families,
            hash_block_size=self.hash_block_size,
            discard_partial_chunks=discard_partial_chunks,
        )
        self.cache_transfer_granularity = (
            self.hybrid_cache_c128_config.chunk_tokens
            if self.hybrid_cache_c128_config.enabled
            else self._infer_cache_transfer_granularity()
        )
        assert self.cache_transfer_granularity is not None
        if self.use_layerwise and self.num_kv_cache_groups > 1:
            raise NotImplementedError("AscendStore layerwise mode does not yet support hybrid KV cache groups.")

        logger.info(
            "use_hybrid: %s, use_mamba: %s, num_kv_cache_groups: %s, hash_block_size: %s, lcm_block_size: %s",
            self.use_hybrid,
            self.use_mamba,
            self.num_kv_cache_groups,
            self.hash_block_size,
            self.lcm_block_size,
        )
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

        self.metadata: list[KeyMetadata] = []
        for group_id in range(self.num_kv_cache_groups):
            # the mamba kv_heads is not same with the full attention, can't share the cache data
            group_tp_rank = self.tp_rank if self.group_uses_align_state[group_id] else self.head_or_tp_rank
            self.metadata.append(
                KeyMetadata(
                    model_config.model.rstrip("/").split("/")[-1],
                    group_tp_rank,
                    self.pcp_rank,
                    self.dcp_rank,
                    self.pp_rank,
                    group_id,
                )
            )

        self.token_database = ChunkedTokenDatabase(
            self.metadata,
            self.grouped_block_size,
            partitions,
            self.use_hybrid,
            self.hash_block_size,
            self.hybrid_cache_c128_config,
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
        self._range_execution_id = -1
        self._range_save_started = False
        self._range_read_plans: dict[
            str, tuple[ReqMeta, RequestStoreLease, list[_LayerwiseRangeBlock]]
        ] = {}
        self._range_layer_futures: list[Future[None]] = []
        self._range_finished_requests: set[str] = set()
        self._range_executor: ThreadPoolExecutor | None = None
        self._range_put_registry: StorePutRegistry | None = None
        self._range_read_registry: StoreReadLeaseRegistry | None = None
        self._range_components: dict[tuple[int, str], list[_LayerwiseRangeComponent]] = {}
        self._range_object_sizes: dict[int, int] = {}
        self._range_layout_fingerprint: str | None = None
        self._range_layout_version = 1
        self._range_request_leases: dict[tuple[str, int], RequestStoreLease] = {}
        # RequestTracker in vLLM 0.23 does not carry a generation field yet.
        # Keep the generation at the worker boundary so a recycled request id
        # cannot reuse an old Store lease or an old in-flight object.
        self._range_request_generations: dict[str, int] = {}
        self._next_range_request_generation = 0
        self._range_sessions_by_request: dict[str, _LayerwiseRangeSessionState] = {}
        self._range_load_layers_seen: set[str] = set()
        if self.use_layerwise_range:
            self._range_executor = ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="layerwise-store-range", initializer=self.m_store.set_device
            )
            self._range_put_registry = StorePutRegistry(self.m_store)
        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = False
        if kv_event_config and kv_event_config.enable_kv_cache_events:
            self.enable_kv_events = True

        self.kv_send_thread: KVTransferThread | None = None
        self.kv_recv_thread: KVTransferThread | None = None

        self.finished_store_req: set[str] = set()

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
            transfer_chunk_tokens=self.hybrid_cache_c128_config.chunk_tokens,
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
        return self.grouped_block_size[group_id]

    @staticmethod
    def _get_group_family(families: list[str], group_id: int) -> str:
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
    def _uses_mamba_kv_cache(use_hybrid: bool, kv_cache_config: KVCacheConfig | None):
        if not use_hybrid or kv_cache_config is None:
            return False
        return any([isinstance(g.kv_cache_spec, MambaSpec) for g in kv_cache_config.kv_cache_groups])

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

    @staticmethod
    def _align_up(value: int, alignment: int) -> int:
        return (value + alignment - 1) // alignment * alignment

    def _is_mtp_layer_name(self, layer_name: str) -> bool:
        from vllm_ascend.attention.offload_capability import is_speculative_cache_layer
        from vllm_ascend.ops.dsa_offload import get_dsa_config_value

        return is_speculative_cache_layer(
            layer_name,
            num_hidden_layers=get_dsa_config_value(
                self.model_config, "num_hidden_layers"
            ),
        )

    def _range_group_layer_names(self, group_id: int) -> list[str]:
        if self.kv_cache_config is None:
            return [name for name in self.kv_caches if not self._is_mtp_layer_name(name)]
        names = list(self.kv_cache_config.kv_cache_groups[group_id].layer_names)
        return [name for name in names if name in self.kv_caches and not self._is_mtp_layer_name(name)]

    def _init_layerwise_range_layout(self) -> None:
        if not self.use_layerwise_range or self._range_layout_fingerprint is not None:
            return
        payload: list[dict[str, object]] = []
        for group_id in range(self.num_kv_cache_groups):
            offset = 0
            layer_id = 0
            names = self._range_group_layer_names(group_id)
            for layer_name in names:
                cache_tuple = self._as_cache_tuple(self.kv_caches[layer_name])
                components: list[_LayerwiseRangeComponent] = []
                for component_id, cache in enumerate(cache_tuple):
                    block_len, block_stride, _, _ = self._get_cache_block_metadata(cache)
                    offset = self._align_up(offset, 64)
                    components.append(_LayerwiseRangeComponent(cache.data_ptr(), block_len, block_stride, offset))
                    payload.append({"group": group_id, "layer": layer_name, "layer_id": layer_id, "component": component_id, "block_len": block_len, "block_stride": block_stride, "offset": offset})
                    offset += block_len
                self._range_components[(group_id, layer_name)] = components
                layer_id += 1
            if names:
                self._range_object_sizes[group_id] = self._align_up(offset, 64)
        if not self._range_components:
            raise RuntimeError("Layerwise range Store has no non-MTP attention layers")
        self._range_layout_fingerprint = stable_model_fingerprint({"model": str(self.model_config.model), "groups": payload})
        logger.info("Initialized layerwise range layout groups=%s object_sizes=%s fingerprint=%s", sorted(self._range_object_sizes), self._range_object_sizes, self._range_layout_fingerprint)

    def _range_key(self, key, group_id: int) -> str:
        self._init_layerwise_range_layout()
        assert self._range_layout_fingerprint is not None
        family = self._get_group_family(self.kv_cache_group_families, group_id)
        return build_layerwise_store_key(key.key_metadata, key.chunk_hash, model_fingerprint=self._range_layout_fingerprint, cache_layout_version=self._range_layout_version, group_id=group_id, cache_family=family).to_string()

    def _range_records(self, request: ReqMeta, token_len: int, group_id: int) -> list[_LayerwiseRangeBlock]:
        if group_id >= len(request.block_ids_by_group):
            return []
        block_ids = request.block_ids_by_group[group_id]
        block_size = self.grouped_block_size[group_id]
        records: list[_LayerwiseRangeBlock] = []
        skip_null = bool(request.skip_null_blocks_by_group and request.skip_null_blocks_by_group[group_id])
        for start, end, key, block_id in self.token_database.process_tokens_with_block_ids(token_len, request.block_hashes, block_ids, kv_cache_group_id=group_id, skip_null_blocks=skip_null):
            if end - start != block_size:
                continue
            records.append(_LayerwiseRangeBlock(self._range_key(key, group_id), start, end, block_id, group_id))
        return records

    def _shard_range_save_records(
        self,
        request: ReqMeta,
        group_id: int,
        records: list[_LayerwiseRangeBlock],
    ) -> list[_LayerwiseRangeBlock]:
        """Select the unique TP writer for replicated KV heads.

        Keep this consistent with ``KVCacheStoreSendingThread``.  When the
        number of KV heads is smaller than TP, multiple TP workers generate
        the same Store key.  Only one of those workers may open a Mooncake put
        session for each block.  Loads are intentionally not sharded because
        every TP worker needs its local history workspace populated.
        """
        if (
            self.dcp_size > 1
            or request.disable_tp_key_sharding
            or self.group_uses_align_state[group_id]
        ):
            return records
        return records[self.tp_rank % self.put_step :: self.put_step]

    def _range_destination_for_layers(
        self,
        layer_names: Sequence[str],
        records: list[_LayerwiseRangeBlock],
    ):
        keys: list[str] = []
        ptrs: list[list[int]] = []
        sizes: list[list[int]] = []
        offsets: list[list[int]] = []
        for record in records:
            component_groups = [
                self._range_components.get((record.group_id, layer_name))
                for layer_name in layer_names
            ]
            if not any(component_groups):
                continue
            if not all(component_groups):
                raise RuntimeError(
                    "Layerwise range GET group crosses incompatible cache "
                    f"groups: layers={tuple(layer_names)!r}, "
                    f"group_id={record.group_id}"
                )
            components = [
                component
                for group in component_groups
                if group is not None
                for component in group
            ]
            keys.append(record.key)
            ptrs.append([c.base_addr + record.block_id * c.block_stride for c in components])
            sizes.append([c.block_len for c in components])
            offsets.append([c.object_offset for c in components])
        return keys, ptrs, sizes, offsets

    def _range_destination(
        self,
        layer_name: str,
        records: list[_LayerwiseRangeBlock],
    ):
        return self._range_destination_for_layers((layer_name,), records)

    def _normalize_range_request_identity(self, request: ReqMeta) -> None:
        """Attach a worker-local generation to vLLM 0.23 request metadata.

        RequestTracker in the pinned vLLM does not expose a generation.  A
        request id can nevertheless be recycled after completion, so the
        Store lease owner must not be just that id.  Explicit generations are
        accepted for newer vLLM callers but are checked against the worker
        mapping to reject stale metadata.
        """

        request_id = request.req_id
        supplied = request.request_generation
        current = self._range_request_generations.get(request_id)
        if current is None:
            if supplied > 0:
                generation = supplied
                self._next_range_request_generation = max(
                    generation, self._next_range_request_generation
                )
            else:
                self._next_range_request_generation += 1
                generation = self._next_range_request_generation
            self._range_request_generations[request_id] = generation
        else:
            if supplied > 0 and supplied != current:
                raise RuntimeError(
                    f"stale Store request generation for {request_id}: "
                    f"metadata={supplied}, active={current}"
                )
            generation = current
        request.request_generation = generation
        if request.chunk_id < 0:
            raise ValueError(f"invalid negative Store chunk id for {request_id}")

    def _drop_range_request_identity(self, request_id: str) -> None:
        """Forget only the active generation; keep the monotonic counter."""

        self._range_request_generations.pop(request_id, None)

    def _abort_range_read_plans(self) -> None:
        """Release all active Store read leases after a range-load failure."""
        owners = list(self._range_request_leases.items())
        self._range_request_leases.clear()
        self._range_read_plans.clear()
        for owner, lease in owners:
            try:
                lease.release()
            except BaseException:
                logger.exception("Failed to release Store lease after range load failure: %s", owner)
            self._drop_range_request_identity(owner[0])
        self._range_load_layers_seen.clear()

    def _get_range_lease(self, request: ReqMeta) -> RequestStoreLease:
        if self._range_read_registry is None:
            self._range_read_registry = StoreReadLeaseRegistry(self.m_store)
        owner = (request.req_id, request.request_generation)
        lease = self._range_request_leases.get(owner)
        if lease is None:
            lease = RequestStoreLease(owner, self._range_read_registry)
            self._range_request_leases[owner] = lease
        return lease

    def _infer_cache_group_metadata(self, group_id: int, layer_names: list[str]):
        group_addrs: list[int] = []
        group_block_lens: list[int] = []
        group_block_strides: list[int] = []
        group_tensors: list[torch.Tensor] = []
        group_block_size_scales: list[int] = []
        for layer_name in layer_names:
            cache_or_caches = self.kv_caches[layer_name]
            for cache in self._as_cache_tuple(cache_or_caches):
                base_addr = cache.data_ptr()
                block_len, block_stride, _, block_size_scale = self._get_cache_block_metadata(cache)
                group_addrs.append(base_addr)
                group_block_lens.append(block_len)
                group_block_strides.append(block_stride)
                group_tensors.append(cache)
                group_block_size_scales.append(block_size_scale)
        self.group_kv_caches_base_addr[group_id] = group_addrs
        self.group_block_len[group_id] = group_block_lens
        self.group_block_stride[group_id] = group_block_strides
        self.group_kv_cache_tensors[group_id] = group_tensors
        self.group_block_size_scales[group_id] = group_block_size_scales
        self.group_num_layers[group_id] = len(layer_names)

    def _create_c128_staging_buffers(self) -> tuple[list[int], list[int]]:
        """Create one reusable full-page staging value for every C128 tensor."""
        self.c128_staging_tensors: dict[int, list[torch.Tensor]] = {}
        config = self.hybrid_cache_c128_config
        if not config.enabled:
            return [], []
        assert config.c128_group_id is not None
        if config.c128_slots_per_page is None:
            raise RuntimeError("C128 slot geometry is missing from the hybrid cache configuration.")
        group_id = config.c128_group_id
        if group_id not in self.group_kv_cache_tensors:
            raise RuntimeError(f"C128 cache group {group_id} has no registered KV cache tensors.")
        staging_tensors: list[torch.Tensor] = []
        ptrs: list[int] = []
        lengths: list[int] = []
        for cache, block_size_scale in zip(
            self.group_kv_cache_tensors[group_id],
            self.group_block_size_scales[group_id],
            strict=True,
        ):
            first_page = cache.narrow(0, 0, block_size_scale)
            if not first_page.is_contiguous():
                raise ValueError("hybrid C128 transfer requires contiguous external KV cache pages.")
            if first_page.numel() % config.c128_slots_per_page != 0:
                raise ValueError(
                    "The C128 external page cannot be divided into "
                    f"{config.c128_slots_per_page} cache slots."
                )
            staging = torch.empty_like(first_page, memory_format=torch.contiguous_format)
            if not staging.is_contiguous():
                raise ValueError("hybrid C128 transfer requires a contiguous staging page.")
            staging_tensors.append(staging)
            ptrs.append(staging.data_ptr())
            lengths.append(staging.numel() * staging.element_size())
        self.c128_staging_tensors[group_id] = staging_tensors
        return ptrs, lengths

    def _get_c128_staging_value(self, group_id: int) -> tuple[list[int], list[int]]:
        staging_tensors = self.c128_staging_tensors.get(group_id)
        if not staging_tensors:
            raise RuntimeError(f"C128 staging buffers are not registered for cache group {group_id}.")
        return (
            [tensor.data_ptr() for tensor in staging_tensors],
            [tensor.numel() * tensor.element_size() for tensor in staging_tensors],
        )

    def _merge_c128_staging_chunk(self, group_id: int, chunk: TransferChunkWithBlockId) -> None:
        """Copy only the key-authoritative slots from staging into a target page."""
        slots_per_page = self.hybrid_cache_c128_config.c128_slots_per_page
        if slots_per_page is None:
            raise RuntimeError("C128 slot geometry is missing from the hybrid cache configuration.")
        if chunk.value_start < 0 or chunk.value_end > slots_per_page:
            raise ValueError(
                "Invalid C128 authoritative range "
                f"[{chunk.value_start}, {chunk.value_end}) for {slots_per_page} slots."
            )
        if chunk.value_start >= chunk.value_end:
            raise ValueError("The C128 authoritative range must not be empty.")

        target_tensors = self.group_kv_cache_tensors[group_id]
        block_size_scales = self.group_block_size_scales[group_id]
        staging_tensors = self.c128_staging_tensors[group_id]
        for target_cache, block_size_scale, staging in zip(
            target_tensors,
            block_size_scales,
            staging_tensors,
            strict=True,
        ):
            target_start = chunk.block_id * block_size_scale
            if target_start < 0 or target_start + block_size_scale > target_cache.shape[0]:
                raise ValueError(f"C128 target block id {chunk.block_id} is outside the KV cache allocation.")
            target_page = target_cache.narrow(0, target_start, block_size_scale)
            if not target_page.is_contiguous():
                raise ValueError("hybrid C128 transfer requires contiguous target pages.")
            if target_page.numel() != staging.numel():
                raise ValueError("C128 target and staging pages must have identical sizes.")
            if target_page.numel() % slots_per_page != 0:
                raise ValueError(
                    f"The C128 external page cannot be divided into {slots_per_page} cache slots."
                )

            elements_per_slot = target_page.numel() // slots_per_page
            element_offset = chunk.value_start * elements_per_slot
            element_count = (chunk.value_end - chunk.value_start) * elements_per_slot
            target_page.view(-1).narrow(0, element_offset, element_count).copy_(
                staging.view(-1).narrow(0, element_offset, element_count)
            )

        # copy_ is asynchronous on NPU. A single staging page is reused by the
        # next Mooncake get, so its authoritative range must be consumed first.
        if staging_tensors and staging_tensors[0].device.type == "npu":
            torch.npu.current_stream().synchronize()

    def _record_sync_load_failures(
        self,
        request: ReqMeta,
        block_ids: list[int],
        results: list[int] | None,
    ) -> None:
        if results is None:
            results = [1] * len(block_ids)
        if not any(result != 0 for result in results):
            return
        missing_block_ids = record_failed_blocks(block_ids, results)
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

    def _build_sync_load_group_plan(
        self,
        request: ReqMeta,
        load_group_ids: list[int],
    ) -> list[tuple[int, list[int], int, bool, bool]]:
        """Precompute group-level sync load state once per request."""
        load_spec = request.load_spec
        assert load_spec is not None
        is_c128_enabled = self.hybrid_cache_c128_config.enabled
        c128_mask_num = 0
        if is_c128_enabled:
            c128_chunk_tokens = self.hybrid_cache_c128_config.chunk_tokens
            assert c128_chunk_tokens is not None
            c128_mask_num = (
                load_spec.vllm_cached_tokens // c128_chunk_tokens * c128_chunk_tokens
            )

        group_plan: list[tuple[int, list[int], int, bool, bool]] = []
        for group_id in load_group_ids:
            if group_id >= len(request.block_ids_by_group):
                continue
            block_ids = request.block_ids_by_group[group_id]
            if is_c128_enabled:
                mask_num = c128_mask_num
            else:
                group_block_size = self.grouped_block_size[group_id]
                mask_num = (
                    load_spec.vllm_cached_tokens // group_block_size * group_block_size
                )
            skip_null = (
                group_id < len(self.group_uses_align_state)
                and self.group_uses_align_state[group_id]
            )
            is_c128_group = (
                is_c128_enabled and self.token_database.is_c128_group(group_id)
            )
            group_plan.append(
                (group_id, block_ids, mask_num, skip_null, is_c128_group)
            )
        return group_plan

    def _collect_sync_load_chunks(
        self,
        request: ReqMeta,
        token_len: int,
        load_group_ids: list[int],
    ) -> tuple[
        list[str],
        list[list[int]],
        list[list[int]],
        list[int],
        dict[int, list[TransferChunkWithBlockId]],
        int,
    ]:
        key_list: list[str] = []
        addr_list: list[list[int]] = []
        size_list: list[list[int]] = []
        block_id_list: list[int] = []
        c128_load_plan: dict[int, list[TransferChunkWithBlockId]] = {}
        c128_page_count = 0
        load_masks = self.token_database.load_mask(request.block_hashes, token_len)
        for (
            group_id,
            block_ids,
            mask_num,
            skip_null,
            is_c128_group,
        ) in self._build_sync_load_group_plan(request, load_group_ids):
            for chunk in self.token_database.process_transfer_chunks_with_block_ids(
                token_len,
                request.block_hashes,
                block_ids,
                mask_num,
                kv_cache_group_id=group_id,
                skip_null_blocks=skip_null,
            ):
                if not self.token_database.mask_allows_chunk(
                    load_masks, group_id, chunk.raw_start
                ):
                    continue
                if is_c128_group:
                    c128_load_plan.setdefault(group_id, []).append(chunk)
                    continue
                addr, size, block_id = self.token_database.prepare_transfer_value(
                    chunk,
                    block_ids,
                    kv_cache_group_id=group_id,
                )
                key_list.append(chunk.key.to_string())
                addr_list.append(addr)
                size_list.append(size)
                block_id_list.append(block_id)
        c128_page_count = sum(
            len(aggregate_c128_page_chunks(chunks, self.hybrid_cache_c128_config.c128_slots_per_page))
            for chunks in c128_load_plan.values()
        )
        return (
            key_list,
            addr_list,
            size_list,
            block_id_list,
            c128_load_plan,
            c128_page_count,
        )

    def _load_direct_sync_chunks(
        self,
        request: ReqMeta,
        token_len: int,
        load_group_ids: list[int],
        key_list: list[str],
        addr_list: list[list[int]],
        size_list: list[list[int]],
        block_id_list: list[int],
    ) -> None:
        rotation = self.tp_rank % len(key_list)
        key_list_c = key_list[rotation:] + key_list[:rotation]
        addr_list_c = addr_list[rotation:] + addr_list[:rotation]
        size_list_c = size_list[rotation:] + size_list[:rotation]
        block_id_list_c = block_id_list[rotation:] + block_id_list[:rotation]
        logger.debug(
            "KV pool worker calls backend get request=%s token_len=%d groups=%s keys=%d sample_keys=%s",
            request.req_id,
            token_len,
            load_group_ids,
            len(key_list_c),
            key_list_c[:3],
        )
        ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
        self._record_sync_load_failures(request, block_id_list_c, ret)

    def _load_c128_sync_chunks(
        self,
        request: ReqMeta,
        c128_load_plan: dict[int, list[TransferChunkWithBlockId]],
    ) -> None:
        """Load one complete physical C128 page per aggregated page key."""
        for group_id, chunks in c128_load_plan.items():
            page_chunks = aggregate_c128_page_chunks(chunks, self.hybrid_cache_c128_config.c128_slots_per_page)
            key_list: list[str] = []
            addr_list: list[list[int]] = []
            size_list: list[list[int]] = []
            block_id_list: list[int] = []
            for chunk in page_chunks:
                block_ids = list(chunk.block_ids) or [chunk.block_id]
                addr, size, block_id = self.token_database.prepare_transfer_value(
                    chunk,
                    block_ids,
                    kv_cache_group_id=group_id,
                )
                if not addr:
                    continue
                key_list.append(chunk.key.to_string())
                addr_list.append(addr)
                size_list.append(size)
                block_id_list.append(block_id)
            if not key_list:
                continue
            rotation = self.tp_rank % len(key_list)
            key_list = key_list[rotation:] + key_list[:rotation]
            addr_list = addr_list[rotation:] + addr_list[:rotation]
            size_list = size_list[rotation:] + size_list[:rotation]
            block_id_list = block_id_list[rotation:] + block_id_list[:rotation]
            ret = self.m_store.get(key_list, addr_list, size_list)
            self._record_sync_load_failures(request, block_id_list, ret)

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

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache_tuple = self._as_cache_tuple(first_kv_cache_tuple)
        first_kv_cache = first_kv_cache_tuple[0]

        self.num_blocks = (
            self.kv_cache_config.num_blocks if self.kv_cache_config is not None else first_kv_cache.shape[0]
        )
        logger.info("num_blocks: %s", self.num_blocks)
        self.group_kv_caches_base_addr: dict[int, list[int]] = {}
        self.group_block_len: dict[int, list[int]] = {}
        self.group_block_stride: dict[int, list[int]] = {}
        self.group_kv_cache_tensors: dict[int, list[torch.Tensor]] = {}
        self.group_block_size_scales: dict[int, list[int]] = {}
        self.kv_caches = kv_caches
        self.group_kv_cache_families: dict[int, str] = {
            group_id: self._get_group_family(self.kv_cache_group_families, group_id)
            for group_id in range(self.num_kv_cache_groups)
        }
        self.group_num_layers: dict[int, int] = {}

        logger.info(
            "Registering KV_Caches. use_mla: %s, use_sparse: %s, shape %s",
            self.use_mla,
            self.use_sparse,
            first_kv_cache.shape,
        )

        registered_regions: dict[int, tuple[int, int]] = {}
        for cache_or_caches in kv_caches.values():
            for cache in self._as_cache_tuple(cache_or_caches):
                base_addr = cache.data_ptr()
                _, _, region_len, _ = self._get_cache_block_metadata(cache)
                if not isinstance(region_len, int):
                    region_len = 0
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
                self._infer_cache_group_metadata(group_id, group_spec.layer_names)
        else:
            self._infer_cache_group_metadata(0, list(kv_caches.keys()))

        staging_ptrs, staging_lengths = self._create_c128_staging_buffers()

        # group_num_layers is computed from the actual kv_caches dict which
        # includes ALL attention layers (main + MTP), so it is the authoritative
        # layer count for this worker.
        original_num_layers = self.num_layers
        self.num_layers = sum(self.group_num_layers.values())
        if self.num_layers != original_num_layers:
            logger.info(
                "KVPoolWorker: updated num_layers %d -> %d (includes MTP/spec-decode draft layers).",
                original_num_layers,
                self.num_layers,
            )

        self.m_store.register_buffer(ptrs, lengths)
        if self.use_layerwise_range:
            self._init_layerwise_range_layout()
            self._range_read_registry = StoreReadLeaseRegistry(self.m_store)
        self.token_database.set_group_buffers(
            self.group_kv_caches_base_addr,
            self.group_block_len,
            self.group_block_stride,
            cache_role="kv",
            group_cache_families=self.group_kv_cache_families,
            group_num_layers=self.group_num_layers,
        )

        if self.use_layerwise and not self.use_layerwise_range:
            self.get_event = threading.Event()
            if self.kv_role in ["kv_producer", "kv_both"]:
                ready_event_sending = threading.Event()
                self.kv_send_thread = KVCacheStoreLayerSendingThread(
                    self.m_store,
                    self.token_database,
                    self.grouped_block_size,
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
                self.grouped_block_size,
                self.tp_rank,
                self.dcp_size,
                ready_event,
                self.get_event,
                self._invalid_block_ids,
                self._invalid_block_ids_lock,
            )
            self.kv_recv_thread.start()
            ready_event.wait()
        else:
            if self.kv_role in ["kv_producer", "kv_both"] or self.consumer_is_to_put:
                ready_event_sending = threading.Event()
                self.kv_send_thread = KVCacheStoreSendingThread(
                    self.m_store,
                    self.token_database,
                    self.grouped_block_size,
                    self.tp_rank,
                    self.dcp_size,
                    self.put_step,
                    self.kv_role,
                    ready_event_sending,
                    self.group_uses_align_state,
                    self.enable_kv_events,
                )
                self.kv_send_thread.start()
            if self.load_async:
                ready_event = threading.Event()
                self.kv_recv_thread = KVCacheStoreRecvingThread(
                    self.m_store,
                    self.token_database,
                    self.grouped_block_size,
                    self.tp_rank,
                    self.dcp_size,
                    ready_event,
                    self._invalid_block_ids,
                    self._invalid_block_ids_lock,
                    self._get_c128_staging_value,
                    self._merge_c128_staging_chunk,
                )
                self.kv_recv_thread.start()
                ready_event.wait()

    def _start_load_range_request(
        self,
        request: ReqMeta,
        token_len: int,
        *,
        require_all: bool = False,
    ) -> None:
        self._normalize_range_request_identity(request)
        request.skip_null_blocks_by_group = self.group_uses_align_state
        load_group_ids = request.kv_cache_group_ids or [0]
        records: list[_LayerwiseRangeBlock] = []
        for group_id in load_group_ids:
            records.extend(self._range_records(request, token_len, group_id))
        lease = self._get_range_lease(request)
        if records:
            unique_records = list({record.key: record for record in records}.values())
            keys = [record.key for record in unique_records]
            exists = self.m_store.exists(keys)
            if exists is None or len(exists) != len(keys):
                raise RuntimeError(
                    "Layerwise range exists returned invalid result for "
                    f"{request.req_id}"
                )
            if require_all:
                missing_keys = [
                    record.key
                    for record, code in zip(unique_records, exists, strict=True)
                    if int(code) != 1
                ]
                if missing_keys:
                    raise StoreCommitError(
                        "Layerwise chunk history is incomplete for "
                        f"{request.req_id}: missing_keys={missing_keys!r}"
                    )
            records = [
                record
                for record, code in zip(unique_records, exists, strict=True)
                if int(code) == 1
            ]
            history_keys = [record.key for record in records]
            if set(history_keys) - lease.history_keys:
                lease.refresh_history(
                    list(lease.history_keys) + history_keys
                )
        self._range_read_plans[request.req_id] = (request, lease, records)

    def start_load_kv(self, metadata: AscendConnectorMetadata):
        self.current_layer = 0
        self.layerwise_retrievers = []
        self._range_load_layers_seen.clear()
        logger.debug("KV pool worker start_load_kv requests=%d", len(metadata.requests))
        for request in metadata.requests:
            load_spec = request.load_spec
            history_token_len = request.history_token_len
            if self.use_layerwise_range and history_token_len > 0:
                if history_token_len > request.token_len_chunk:
                    raise ValueError(
                        "Layerwise chunk history exceeds the complete Store "
                        f"prefix for {request.req_id}: history={history_token_len}, "
                        f"stored={request.token_len_chunk}"
                    )
                logger.debug(
                    "KV pool worker prepare chunk history get req=%s "
                    "history_token_len=%d token_len_chunk=%d",
                    request.req_id,
                    history_token_len,
                    request.token_len_chunk,
                )
                try:
                    self._start_load_range_request(
                        request,
                        history_token_len,
                        require_all=True,
                    )
                except BaseException:
                    self._abort_range_read_plans()
                    raise
                continue
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
            if self.use_layerwise_range:
                self._start_load_range_request(request, token_len)
                continue
            if self.use_layerwise:
                layerwise_retriever = self.retrieve_layer(request)
                next(layerwise_retriever)  # first layer load
                self.layerwise_retrievers.append(layerwise_retriever)
            elif self.load_async:
                self.kv_recv_thread.add_request(  # type: ignore[union-attr]
                    request,
                )
            else:
                (
                    key_list,
                    addr_list,
                    size_list,
                    block_id_list,
                    c128_load_plan,
                    c128_page_count,
                ) = self._collect_sync_load_chunks(request, token_len, load_group_ids)
                if not key_list and not c128_load_plan:
                    continue
                if key_list:
                    self._load_direct_sync_chunks(
                        request,
                        token_len,
                        load_group_ids,
                        key_list,
                        addr_list,
                        size_list,
                        block_id_list,
                    )
                if c128_load_plan:
                    self._load_c128_sync_chunks(request, c128_load_plan)
                logger.debug(
                    "KV pool worker backend get returned request=%s token_len=%d groups=%s keys=%d",
                    request.req_id,
                    token_len,
                    load_group_ids,
                    len(key_list) + c128_page_count,
                )

    def _wait_for_layer_load_ranges(
        self,
        layer_names: Sequence[str],
    ) -> None:
        layer_names = tuple(layer_names)
        if not layer_names:
            return
        for req_id, (request, lease, records) in list(self._range_read_plans.items()):
            keys, ptrs, sizes, offsets = self._range_destination_for_layers(
                layer_names,
                records,
            )
            if not keys:
                continue
            rank_offset = self.tp_rank % len(keys)
            keys = keys[rank_offset:] + keys[:rank_offset]
            ptrs = ptrs[rank_offset:] + ptrs[:rank_offset]
            sizes = sizes[rank_offset:] + sizes[:rank_offset]
            offsets = offsets[rank_offset:] + offsets[:rank_offset]
            try:
                logger.debug(
                    "KV pool worker range get req=%s layers=%s keys=%d",
                    req_id,
                    layer_names,
                    len(keys),
                )
                result = lease.load_layer(keys, ptrs, sizes, offsets)
                expected_bytes = tuple(sum(row) for row in sizes)
                if len(result) != len(keys) or any(
                    int(code) != expected
                    for code, expected in zip(result, expected_bytes, strict=False)
                ):
                    raise StoreCommitError(
                        f"Store range get failed for {req_id}: "
                        f"result={result!r}, expected={expected_bytes!r}"
                    )
            except BaseException as error:
                logger.error(
                    "Layerwise range load failed req=%s layers=%s error=%s",
                    req_id,
                    layer_names,
                    error,
                )
                with self._invalid_block_ids_lock:
                    self._invalid_block_ids.update(
                        record.block_id
                        for record in records
                        if record.key in set(keys)
                    )
                # A range miss is not a prefix-cache miss: the workspace may be
                # partially written. Abort before Attention can consume it.
                self._abort_range_read_plans()
                raise StoreCommitError(
                    f"Layerwise Store load failed for request {req_id}, "
                    f"layers {layer_names}"
                ) from error
        self._range_load_layers_seen.update(layer_names)
        all_layers = {name for group_id, name in self._range_components}
        if all_layers and all_layers.issubset(self._range_load_layers_seen):
            for req_id in list(self._range_read_plans):
                # The read plan is scoped to this forward/chunk, but the
                # request-level lease is not. A non-final prefill chunk must
                # retain its history lease so the following chunk can reuse
                # the same get session. The lease is released by
                # wait_for_save() on the final chunk, or by the cancellation
                # and failure paths after all range operations quiesce.
                self._range_read_plans.pop(req_id, None)

    def _wait_for_layer_load_range(self, layer_name: str | None) -> None:
        if layer_name is None:
            return
        self._wait_for_layer_load_ranges((layer_name,))

    def wait_for_layer_load(self, layer_name: str | None = None) -> None:
        if self.use_layerwise_range:
            self._wait_for_layer_load_range(layer_name)
            return
        del layer_name
        for layerwise_retriever in self.layerwise_retrievers:
            ret_token_mask = next(layerwise_retriever)
            if self.current_layer == self.num_layers - 1:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.debug("Retrieved %s tokens", num_retrieved_tokens)

    def wait_for_layer_loads(self, layer_names: Sequence[str]) -> None:
        if self.use_layerwise_range:
            self._wait_for_layer_load_ranges(layer_names)
            return
        for layer_name in layer_names:
            self.wait_for_layer_load(layer_name)

    def get_block_ids_with_load_errors(self) -> set[int]:
        with self._invalid_block_ids_lock:
            invalid_blocks = self._invalid_block_ids.copy()
            self._invalid_block_ids.clear()
        return invalid_blocks

    def _completed_future(self) -> Future[None]:
        future: Future[None] = Future()
        future.set_result(None)
        return future

    def _start_range_save(self, connector_metadata: AscendConnectorMetadata) -> None:
        if self._range_save_started:
            return

        assert self._range_put_registry is not None
        execution_id = self._range_execution_id + 1
        sessions: dict[str, _LayerwiseRangeSessionState] = {}
        execution_keys: set[str] = set()
        try:
            for request in connector_metadata.requests:
                self._normalize_range_request_identity(request)
                request.skip_null_blocks_by_group = self.group_uses_align_state
                records: list[_LayerwiseRangeBlock] = []
                for group_id in request.kv_cache_group_ids or [0]:
                    group_records = self._range_records(
                        request, request.token_len_chunk, group_id
                    )
                    records.extend(
                        self._shard_range_save_records(
                            request,
                            group_id,
                            group_records,
                        )
                    )
                lease = self._get_range_lease(request)
                missing: list[_LayerwiseRangeBlock] = []
                if records:
                    unique = list(
                        {record.key: record for record in records}.values()
                    )
                    known = set(lease.history_keys)
                    candidates = [
                        record for record in unique if record.key not in known
                    ]
                    if candidates:
                        exists = self.m_store.exists(
                            [record.key for record in candidates]
                        )
                        if exists is None or len(exists) != len(candidates):
                            raise RuntimeError(
                                "Layerwise range exists returned invalid result "
                                f"for {request.req_id}"
                            )
                        complete_keys: list[str] = []
                        for record, code in zip(
                            candidates, exists, strict=True
                        ):
                            if int(code) == 1:
                                complete_keys.append(record.key)
                            else:
                                missing.append(record)
                        if complete_keys:
                            lease.refresh_history(
                                list(lease.history_keys) + complete_keys
                            )
                # A final partial chunk may have no new full blocks. It still
                # needs a session so its existing read lease can be released and
                # get_finished() can observe completion.
                bindings = (
                    [
                        PutBinding(
                            record.key,
                            self._range_object_sizes[record.group_id],
                        )
                        for record in missing
                    ]
                    if request.can_save
                    else []
                )
                claim = self._range_put_registry.claim_execution(
                    execution_id, bindings
                )
                execution_keys.update(claim.commit_futures)
                if claim.yielded_keys:
                    # Cross-instance writer already owns these canonical keys
                    # (put_start hit OBJECT_ALREADY_EXISTS). The local chunk
                    # skips their range put; later chunks pick the objects up
                    # through the stateless exists() triage above.
                    logger.info(
                        "Layerwise Store yielded %d put key(s) to a "
                        "cross-instance owner for request %s",
                        len(claim.yielded_keys),
                        request.req_id,
                    )
                session = ChunkStoreSession(
                    RequestChunkKey(
                        request.req_id,
                        request.request_generation,
                        request.chunk_id,
                    ),
                    self.m_store,
                    lease,
                    self._range_put_registry,
                )
                try:
                    session.start(sorted(lease.history_keys), claim)
                except BaseException:
                    try:
                        session.revoke_uncommitted()
                    except BaseException:
                        logger.exception(
                            "Failed to revoke partially initialized Store "
                            "session %s",
                            request.req_id,
                        )
                    raise
                sessions[request.req_id] = _LayerwiseRangeSessionState(
                    request, session, missing
                )
        except BaseException:
            for state in sessions.values():
                try:
                    state.session.revoke_uncommitted()
                except BaseException:
                    logger.exception(
                        "Failed to revoke Store session during initialization "
                        "rollback: %s",
                        state.request.req_id,
                    )
            # A synchronous save-initialization failure aborts the whole model
            # execution. Drop every read plan and request lease as one unit so a
            # retry cannot retain a plan that points at a closed lease.
            self._abort_range_read_plans()
            self._range_put_registry.forget(sorted(execution_keys))
            self._range_sessions_by_request.clear()
            self._range_layer_futures.clear()
            self._range_save_started = False
            raise

        self._range_execution_id = execution_id
        self._range_sessions_by_request = sessions
        self._range_layer_futures.clear()
        self._range_save_started = True

    def _save_range_layer(self, layer_name: str, connector_metadata: AscendConnectorMetadata, kv_ready_event=None) -> Future[None]:
        self._start_range_save(connector_metadata)
        if self._is_mtp_layer_name(layer_name):
            return self._completed_future()
        assert self._range_executor is not None
        tasks: list[Future[list[int]]] = []
        for state in self._range_sessions_by_request.values():
            if not state.session.put_keys:
                continue
            records = [record for record in state.records if record.key in set(state.session.put_keys)]
            keys, ptrs, sizes, offsets = self._range_destination(layer_name, records)
            if not keys:
                continue
            tasks.append(self._range_executor.submit(state.session.save_layer, keys, ptrs, sizes, offsets, kv_ready_event))
        def wait_tasks() -> None:
            for task in tasks:
                task.result()
        aggregate = self._range_executor.submit(wait_tasks)
        self._range_layer_futures.append(aggregate)
        return aggregate

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer,
        connector_metadata: AscendConnectorMetadata,
        *,
        kv_ready_event=None,
    ) -> Future[None] | None:
        if self.use_layerwise_range:
            return self._save_range_layer(layer_name, connector_metadata, kv_ready_event)
        del layer_name, kv_layer
        # MTP speculative decoding re-runs the base model's attention layers
        # during draft execution (_run_merged_draft), causing extra
        # save_kv_layer calls beyond num_layers. These extra calls would
        # exhaust the store_layer generators and raise StopIteration.
        if self.current_layer >= self.num_layers:
            return
        if self.current_layer == 0:
            self.layerwise_storers = []
            current_event = kv_ready_event
            for request in connector_metadata.requests:
                can_save = request.can_save
                if can_save is None or not can_save:
                    continue

                request.current_event = current_event
                self.kv_send_thread.add_stored_request(  # type: ignore[union-attr]
                    request.req_id
                )
                layerwise_storer = self.store_layer(request, current_event)
                self.layerwise_storers.append(layerwise_storer)
        for layerwise_storer in self.layerwise_storers:
            try:
                if self.current_layer == 0:
                    next(layerwise_storer)
                else:
                    layerwise_storer.send(kv_ready_event)
            except Exception:
                raise
        self.current_layer = self.current_layer + 1

    def wait_for_save(self, connector_metadata: AscendConnectorMetadata):
        if self.use_layerwise_range:
            states = list(self._range_sessions_by_request.items())
            execution_put_keys: set[str] = set()
            for _, state in states:
                execution_put_keys.update(state.session.put_keys)
            try:
                errors: list[BaseException] = []
                # Source-range futures must all settle before any put_end.
                for future in self._range_layer_futures:
                    try:
                        future.result()
                    except BaseException as error:
                        errors.append(error)

                # Stop committing new owners after a source-range failure. The
                # cleanup below revokes every still-open owner and resolves its
                # followers with the same execution failure.
                for _, state in states:
                    if errors:
                        break
                    try:
                        state.session.commit_owned()
                    except BaseException as error:
                        errors.append(error)

                if errors:
                    raise errors[0]

                for req_id, state in states:
                    state.session.finalize_followed(
                        has_more_prefill_chunks=state.request.is_last_chunk is not True
                    )
                    if state.request.is_last_chunk is True:
                        owner = (state.request.req_id, state.request.request_generation)
                        lease = self._range_request_leases.pop(owner, None)
                        if lease is not None:
                            lease.release()
                    self._range_finished_requests.add(req_id)
            except BaseException:
                # Wake followers for every owner that did not reach put_end,
                # then release the failed request's history lease.  A later
                # scheduler retry can reacquire complete keys through exists().
                for _, state in states:
                    try:
                        state.session.revoke_uncommitted()
                    except BaseException:
                        logger.exception("Failed to revoke Store session %s", state.request.req_id)
                    owner = (state.request.req_id, state.request.request_generation)
                    lease = self._range_request_leases.pop(owner, None)
                    if lease is not None:
                        try:
                            lease.release()
                        except BaseException:
                            logger.exception("Failed to release Store lease %s", owner)
                raise
            finally:
                # Put registry entries are scoped to one scheduler execution.
                # Complete objects remain discoverable through exists; do not retain stale futures.
                if self._range_put_registry is not None:
                    self._range_put_registry.forget(sorted(execution_put_keys))
                self._range_sessions_by_request.clear()
                self._range_layer_futures.clear()
                self._range_save_started = False
            return
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
                        logger.info(
                            "Layerwise get failed. Timeout waiting for get_event. Check receiver thread status."
                        )
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
        group_id = 0
        group_block_size = self.grouped_block_size[group_id]
        group_block_hashes = get_block_hashes(request.block_hashes, group_block_size, self.hash_block_size)
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
                    token_ids=request.token_ids,
                    original_block_size=request.original_block_size,
                    block_hashes=group_block_hashes,
                    kv_cache_group_id=group_id,
                )
                self.kv_send_thread.add_request(  # type: ignore[union-attr, call-arg]
                    req_meta
                )  # type: ignore[union-attr, call-arg, arg-type]
                next_event = yield
                if next_event is not None:
                    current_event = next_event
        else:
            for layer_id in range(self.num_layers):
                yield

    def get_finished(self, finished_req_ids: set[str], meta: AscendConnectorMetadata) -> tuple[set[str], set[str]]:
        if self.use_layerwise_range:
            # A preempted request may not reach another save hook. Quiescing is
            # guaranteed by the scheduler before this callback, so release its
            # Store read lease instead of leaking the native session.
            for req_id in meta.preempted_req_ids:
                for owner in [owner for owner in self._range_request_leases if owner[0] == req_id]:
                    lease = self._range_request_leases.pop(owner)
                    lease.release()
                self._range_read_plans.pop(req_id, None)
                self._drop_range_request_identity(req_id)
            done_sending = self._range_finished_requests.intersection(finished_req_ids)
            self._range_finished_requests.difference_update(done_sending)
            for req_id in done_sending:
                for owner in [owner for owner in self._range_request_leases if owner[0] == req_id]:
                    lease = self._range_request_leases.pop(owner)
                    lease.release()
                self._range_read_plans.pop(req_id, None)
                self._drop_range_request_identity(req_id)
            return done_sending, set()
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
                for chunk in self.token_database.process_transfer_chunks(
                    token_len,
                    block_hashes,
                    kv_cache_group_id=group_id,
                ):
                    start, end, key = chunk.raw_start, chunk.raw_end, chunk.key
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

    @staticmethod
    def _chunk_hash_to_bytes(chunk_hash: str) -> bytes:
        if len(chunk_hash) == 64:
            try:
                return bytes.fromhex(chunk_hash)
            except ValueError:
                pass
        return chunk_hash.encode("utf-8")

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

    def _lookup_with_coordinator(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int],
        use_layerwise: bool,
        include_all_ranks: bool,
    ) -> int | None:
        if self.cache_coordinator is None or use_layerwise:
            return None
        if sorted(kv_cache_group_ids) != list(range(self.num_kv_cache_groups)):
            return None

        exists: set[tuple[int, bytes]] = set()
        for group_id in kv_cache_group_ids:
            keys: list[str] = []
            chunk_hashes: list[str] = []
            variant_counts: list[int] = []
            for chunk in self.token_database.process_transfer_chunks(
                token_len,
                block_hashes,
                kv_cache_group_id=group_id,
            ):
                key = chunk.key
                variants = self._expand_lookup_key_variants(key.to_string(), group_id, include_all_ranks)
                keys.extend(variants)
                chunk_hashes.append(key.chunk_hash)
                variant_counts.append(len(variants))

            if not keys:
                continue
            res = self.m_store.exists(keys)  # type: ignore[assignment]
            offset = 0
            for chunk_hash, count in zip(chunk_hashes, variant_counts, strict=True):
                values = res[offset : offset + count]  # type: ignore[index]
                if values and all(value == 1 for value in values):
                    exists.add((group_id, self._chunk_hash_to_bytes(chunk_hash)))
                offset += count

            logger.debug(
                "KV pool coordinator lookup group=%d token_len=%d keys=%d exists_chunks=%d/%d sample_keys=%s",
                group_id,
                token_len,
                len(keys),
                sum(1 for group, _ in exists if group == group_id),
                len(chunk_hashes),
                keys[:3],
            )

        _, hit_length = self.cache_coordinator.find_longest_cache_hit(
            block_hashes,
            token_len,
            ExternalCachedBlockPool(exists),
            apply_eagle=False,
        )
        logger.info(
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
        use_layerwise_range: bool = False,
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
            coordinator_hit = self._lookup_with_coordinator(
                token_len,
                block_hashes,
                kv_cache_group_ids,
                use_layerwise or use_layerwise_range,
                include_all_ranks=True,
            )
            if coordinator_hit is not None:
                return coordinator_hit
            for group_id in kv_cache_group_ids:
                keys = []
                starts = []
                ends = []
                for chunk in self.token_database.process_transfer_chunks(
                    token_len,
                    block_hashes,
                    kv_cache_group_id=group_id,
                ):
                    # Unpack the transfer chunk before constructing the
                    # canonical lookup key.  In range mode the key is rebuilt
                    # from the chunk metadata; without this assignment the
                    # NameError is swallowed below and every lookup appears
                    # as a zero-token hit.
                    start, end, key = chunk.raw_start, chunk.raw_end, chunk.key
                    if use_layerwise_range:
                        # Range mode stores one object per block containing
                        # every layer/component. Query the exact canonical key
                        # used by _range_records() instead of the legacy
                        # per-layer keys generated by split_layers().
                        keys.append(self._range_key(key, group_id))
                    elif use_layerwise:
                        keys_multi_layer = key.split_layers(self.num_layers)
                        for item in keys_multi_layer:
                            keys.append(item.to_string())
                    else:
                        keys.append(key.to_string())
                    starts.append(start)
                    ends.append(end)

                if not keys:
                    return 0

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
                if use_layerwise and not use_layerwise_range:
                    res = self.check_all_layers_exists(res, self.num_layers)
                    num_block = len(keys) // self.num_layers
                multi_tp_values = [
                    res[i * num_block : (i + 1) * num_block]  # type: ignore[index]
                    for i in range(group_tp_size * self.pp_size)
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
        if self.use_mamba and isinstance(self.kv_send_thread, KVCacheStoreSendingThread):
            if ce := self.kv_send_thread.get_completed_events():
                return AscendStoreKVConnectorWorkerMetadata(ce)
        return None
