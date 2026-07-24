from __future__ import annotations

import importlib
import math
import threading
from collections.abc import Generator
from typing import Any

import numpy as np
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
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    backend_map,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerBlockRange,
    LayerLoadTask,
    LayerMultiBlockReqMeta,
    LayerTransferTask,
    ReqMeta,
    block_hash_to_str,
    get_block_hashes,
    get_cache_family_granularity,
    infer_cache_family_ratio,
    infer_group_cache_families,
    infer_tp_mismatch_info,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import (
    AscendStoreCoordinator,
    ExternalCachedBlockPool,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreKeyLayerRecvingThread,
    KVCacheStoreKeyLayerSendingThread,
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
    LayerBatchBuilder,
    _circular_shift,
    record_failed_blocks,
)
from vllm_ascend.distributed.utils import (
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
)
from vllm_ascend.memcache_comm_fence import (
    get_attention_compute_start_gate,
    reset_attention_compute_start_gate,
)
from vllm_ascend.utils import vllm_version_is

# Read lease TTL (ms) for the layerwise load path. batch_add_lease acquires a
# read lease before batch_copy(G2L); the lease must cover the asynchronous
# multi-layer load time.
LAYERWISE_READ_LEASE_TTL_MS = 5 * 60 * 1000


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
        self._init_layerwise_config()

    def _init_parallelism_info(self, model_config, parallel_config) -> None:
        self.local_rank = envs.LOCAL_RANK

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
        self.model_name = model_config.model.split("/")[-1]

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
        self.use_mamba = self._uses_mamba_kv_cache(self.use_hybrid, kv_cache_config)
        self.original_block_size = self._infer_group_block_sizes(vllm_config, kv_cache_config)
        cp_scale = self.pcp_size * self.dcp_size
        self.grouped_block_size = [block_size * cp_scale for block_size in self.original_block_size]
        requested_hash_block_size = (
            vllm_config.cache_config.hash_block_size
            if vllm_version_is("0.25.1")
            else vllm_config.cache_config.prefix_match_unit
        )
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
        self.h2d_stagger_us = int(extra_config.get("h2d_stagger_us", 0))
        self.layerwise_max_transfer_blocks = int(extra_config.get("layerwise_max_transfer_blocks", 0))
        self.layerwise_max_transfer_bytes = int(extra_config.get("layerwise_max_transfer_bytes", 0))

        logger.info(
            "use_hybrid: %s, use_mamba: %s, num_kv_cache_groups: %s, hash_block_size: %s, lcm_block_size: %s",
            self.use_hybrid,
            self.use_mamba,
            self.num_kv_cache_groups,
            self.hash_block_size,
            self.lcm_block_size,
        )

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
            if self.use_layerwise:
                raise ValueError(
                    f"TP mismatch (local_tp={self.tp_size}, peer_tp={self.peer_tp_size}) "
                    "is not supported with layerwise KV transfer (use_layerwise=True). "
                    "The layerwise threads do not implement TP-mismatch handling."
                )
            if self.use_hybrid:
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
            self.metadata, self.grouped_block_size, partitions, self.use_hybrid, self.hash_block_size
        )
        self.cache_coordinator = self._build_cache_coordinator(vllm_config)
        self.token_database.set_cache_coordinator(self.cache_coordinator)

    def _init_backend(self, parallel_config, extra_config) -> None:
        backend = backend_map.get(self.backend.lower())
        assert backend is not None
        backend_path = backend.get("path")
        backend_name = backend.get("name")
        assert backend_path is not None and backend_name is not None
        backend_module = importlib.import_module(backend_path)
        real_backend = getattr(backend_module, backend_name)

        if self.backend.lower() == "memcache":
            self.m_store = real_backend(  # type: ignore[misc]
                parallel_config,
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
        self._transfer_threads_started = False
        # Per-rank GVA cache: maps per-rank store key to its allocated GVA.
        # batch_alloc is non-idempotent (returns MMC_DUPLICATED_OBJECT for an
        # existing key without registering the blob), so the worker must track
        # which keys it has already allocated and reuse those GVAs instead of
        # re-allocating them on every save step.
        self._allocated_gvas: dict[str, int] = {}

    def _init_layerwise_config(self) -> None:
        # Build mapping: physical_layer -> [(group_id, layer_idx_in_group), ...]
        # layer_idx_in_group is the index of the physical layer within the
        # group (not the index in layer_names). Multiple layer_names at the
        # same physical layer (e.g. indexer.k_cache + attn) are treated as
        # multiple cache tensors of ONE layer (caches_per_layer > 1).
        self.physical_layer_to_group_layers: dict[int, list[tuple[int, int]]] = {}

        if self.kv_cache_config is not None and self.num_kv_cache_groups > 1:
            for group_id, group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
                # Map each unique physical layer to a sequential layer_idx_in_group
                phys_to_layer_idx: dict[int, int] = {}
                for layer_name in group_spec.layer_names:
                    physical_layer = self._extract_physical_layer_index(layer_name)
                    if physical_layer >= getattr(self.hf_config, "num_hidden_layers", self.num_layers):
                        continue
                    if physical_layer not in phys_to_layer_idx:
                        phys_to_layer_idx[physical_layer] = len(phys_to_layer_idx)

                # Add one entry per unique physical layer (no duplicates)
                for physical_layer, layer_idx_in_group in phys_to_layer_idx.items():
                    existing = self.physical_layer_to_group_layers.setdefault(physical_layer, [])
                    entry = (group_id, layer_idx_in_group)
                    if entry not in existing:
                        existing.append(entry)

                logger.info(
                    "layerwise group %d: %d layer_names, %d unique physical layers, caches_per_layer=%d",
                    group_id,
                    len(group_spec.layer_names),
                    len(phys_to_layer_idx),
                    len(group_spec.layer_names) // max(1, len(phys_to_layer_idx)),
                )

        self.layer_load_tasks: list[list[LayerTransferTask]] = [[] for _ in range(self.num_layers)]
        self.layer_save_tasks: list[list[LayerTransferTask]] = [[] for _ in range(self.num_layers)]
        self.layer_load_finished_events: list[threading.Event] | None = None
        self.layer_save_finished_events: list[threading.Event] | None = None

        self.next_layer_to_submit = 0
        self.num_prefetch_layers = int(self._extra_config.get("layerwise_prefetch_layers", 1))
        self.sync_save_events: list[torch.npu.Event] | None = None

        logger.info(
            "layerwise config: num_layers=%d num_groups=%d physical_layer_to_group_layers_sample=%s",
            self.num_layers,
            self.num_kv_cache_groups,
            {k: v for k, v in list(self.physical_layer_to_group_layers.items())[:3]},
        )

    def _build_group_layer_builders(self) -> list[LayerBatchBuilder]:
        builders = []
        for group_id in range(self.num_kv_cache_groups):
            group_num_layers = self.group_num_layers.get(group_id, self.num_layers)
            group_block_len = self.group_block_len.get(group_id, self.group_block_len.get(0, []))
            if group_block_len and group_num_layers > 0:
                group_page_size = sum(group_block_len) // group_num_layers
            else:
                group_page_size = self.page_size_bytes
            builders.append(
                LayerBatchBuilder(
                    self.token_database,
                    self.my_key_index,
                    self.num_ranks_per_layer,
                    group_page_size,
                    group_num_layers,
                    group_id=group_id,
                )
            )
        return builders

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
                    self.layerwise_max_transfer_blocks,
                    self.layerwise_max_transfer_bytes,
                    group_builders=self._build_group_layer_builders(),
                )
                self.kv_send_thread.start()
                ready_event_sending.wait()
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
                    self.layerwise_max_transfer_blocks,
                    self.layerwise_max_transfer_bytes,
                    group_builders=self._build_group_layer_builders(),
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
                    self.group_uses_align_state,
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

    def _get_effective_group_block_size(self, group_id: int) -> int:
        cache_family = self._get_group_family(self.kv_cache_group_families, group_id)
        return self._get_group_block_size(group_id) * max(infer_cache_family_ratio(cache_family), 1)

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

    def _extract_physical_layer_index(self, layer_name: str) -> int:
        import regex as re

        m = re.search(r"layers\.(\d+)", layer_name)
        if m:
            return int(m.group(1))
        # MTP layers have names like "mtp.0.self_attn.xxx" without "layers."
        # prefix. Map them after the main model layers.
        if ".mtp." in f".{layer_name}.":
            m = re.search(r"mtp\.(\d+)", layer_name)
            if m:
                num_hidden_layers = getattr(self.hf_config, "num_hidden_layers", self.num_layers)
                return num_hidden_layers + int(m.group(1))
        m = re.search(r"(\d+)", layer_name)
        return int(m.group(1)) if m else 0

    def _infer_cache_group_metadata(self, group_id: int, layer_names: list[str]):
        group_addrs: list[int] = []
        group_block_lens: list[int] = []
        group_block_strides: list[int] = []
        physical_layers = set()
        for layer_name in layer_names:
            phys = self._extract_physical_layer_index(layer_name)
            if phys >= getattr(self.hf_config, "num_hidden_layers", self.num_layers):
                continue
            physical_layers.add(phys)
            cache_or_caches = self.kv_caches[layer_name]
            for cache in self._as_cache_tuple(cache_or_caches):
                base_addr = cache.data_ptr()
                block_len, block_stride, _, _ = self._get_cache_block_metadata(cache)
                group_addrs.append(base_addr)
                group_block_lens.append(block_len)
                group_block_strides.append(block_stride)
        self.group_kv_caches_base_addr[group_id] = group_addrs
        self.group_block_len[group_id] = group_block_lens
        self.group_block_stride[group_id] = group_block_strides
        self.group_num_layers[group_id] = len(physical_layers)

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

        self.kv_caches_base_addr = []

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

        self._align_kv_ptrs(registered_regions)
        ptrs = [start for start, _ in registered_regions.values()]
        lengths = [end - start for start, end in registered_regions.values()]

        if self.kv_cache_config is not None and self.use_hybrid:
            for group_id, group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
                self._infer_cache_group_metadata(group_id, group_spec.layer_names)
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
        if self.num_kv_cache_groups == 1 and new_num_layers != original_num_layers:
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

        # Initialize store, register buffers, and start transfer threads
        # directly here (like main) — no separate init_backend handshake.
        self.m_store.register_buffer(ptrs, lengths)
        self._start_kv_transfer_threads()

    def start_load_kv(self, metadata: AscendConnectorMetadata):
        self.current_layer = 0
        self.layerwise_retrievers: list[Any] = []
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
                    addr, size, block_id = self.token_database.prepare_value(
                        start,
                        end,
                        block_ids,
                        kv_cache_group_id=group_id,
                        block_id=block_id,
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
            logger.debug(
                "KV pool worker calls backend get request=%s token_len=%d groups=%s keys=%d sample_keys=%s",
                request.req_id,
                token_len,
                load_group_ids,
                len(key_list_c),
                key_list_c[:3],
            )
            ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
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

    def _process_save_for_layer_batch(
        self,
        requests: list[ReqMeta],
        layer_id: int,
        group_id: int = 0,
        layer_idx_in_group: int = 0,
    ) -> None:
        # Only the first rank in each put_step group saves to the
        # pool.  Other ranks in the same group share the same KV cache
        # (e.g. MLA latent), so they skip save to avoid redundant writes.
        if self.tp_rank % self.put_step != 0:
            return
        block_size = self._get_effective_group_block_size(group_id)
        request_block_ranges = []
        for request in requests:
            if request.can_save is None or not request.can_save:
                continue
            save_start_block = request.save_start_token // block_size
            save_end_block = request.save_end_token // block_size
            # Skip blocks that are hit in the KV pool — their KV is already
            # in the pool (loaded via load_prepare), so re-saving would write
            # to a READABLE blob and fail with MMC_UNMATCHED_KEY.
            if request.load_spec is not None and request.load_spec.can_load:
                hit_full_blocks = request.load_spec.kvpool_cached_tokens // block_size
                save_start_block = max(save_start_block, hit_full_blocks)
            # Skip blocks already saved in previous requests (GVA cached
            # locally). The overall hit_tokens is min(hits_per_group), so
            # individual groups may have more saved blocks than the overall
            # hit suggests. Without this skip, batch_copy would re-write to
            # already-written blobs, causing -3107 which corrupts the blob
            # state and cascades to load failures (-3101/-3102).
            if self.use_gva_layerwise and save_start_block < save_end_block:
                group_block_hashes = get_block_hashes(request.block_hashes, block_size, self.hash_block_size)
                while save_start_block < save_end_block and save_start_block < len(group_block_hashes):
                    key = self._make_layerwise_gva_key(
                        group_id, block_hash_to_str(group_block_hashes[save_start_block])
                    )
                    if key in self._allocated_gvas:
                        save_start_block += 1
                    else:
                        break
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
                    group_id=group_id,
                    layer_idx_in_group=layer_idx_in_group,
                )
            )

    def _process_load_for_layer_batch(
        self,
        requests: list[ReqMeta],
        layer_id: int,
        group_id: int = 0,
        layer_idx_in_group: int = 0,
    ) -> None:
        block_size = self._get_effective_group_block_size(group_id)
        request_block_ranges = []
        for request in requests:
            if request.load_spec is None or not request.load_spec.can_load:
                continue
            cached_tokens = request.load_spec.kvpool_cached_tokens
            load_start_block = request.load_spec.vllm_cached_tokens // block_size
            cached_full_blocks = cached_tokens // block_size
            full_blocks = min(cached_full_blocks, len(request.block_hashes))
            needs_last_block_at_boundary = (
                cached_tokens > 0 and cached_tokens % block_size == 0 and full_blocks < cached_full_blocks
            )
            if request.last_block_gva is not None and (cached_tokens % block_size != 0 or needs_last_block_at_boundary):
                partial_block_index = cached_full_blocks if cached_tokens % block_size != 0 else cached_full_blocks - 1
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
                    group_id=group_id,
                    layer_idx_in_group=layer_idx_in_group,
                )
            )

    def _make_layerwise_gva_key(self, group_id: int, block_hash_hex: str) -> str:
        """Generate GVA key for layerwise transfer.

        Single-group models use the PR #11585 format (model@hash@rank) for
        backward compatibility. Multi-group models include group_id
        (model@group_id@hash@rank) to distinguish groups.
        """
        if self.num_kv_cache_groups > 1:
            return f"{self.model_name}@{group_id}@{block_hash_hex}@{self.head_or_tp_rank}"
        else:
            return f"{self.model_name}@{block_hash_hex}@{self.head_or_tp_rank}"

    def _alloc_gvas_for_save(self, requests: list[ReqMeta]) -> None:
        """Allocate per-group GVA on the worker side right before batch_copy.

        For multi-group models, iterates all KV cache groups and allocates
        per-group GVAs. Key format: model@group_id@hash@head_or_tp_rank
        (multi-group) or model@hash@head_or_tp_rank (single-group, backward
        compat with PR #11585).
        """
        if not self.use_gva_layerwise:
            return
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_put:
            return
        if self.tp_rank % self.put_step != 0:
            return
        for request in requests:
            if request.can_save is None or not request.can_save:
                continue
            block_hashes = request.block_hashes
            if not block_hashes:
                continue

            all_group_gvas: list[np.ndarray] = []
            all_group_block_ids: list[np.ndarray] = []
            for group_id in range(self.num_kv_cache_groups):
                group_block_size = self.grouped_block_size[group_id]
                cache_family = self._get_group_family(self.kv_cache_group_families, group_id)
                ratio = max(infer_cache_family_ratio(cache_family), 1)
                effective_block_size = group_block_size * ratio
                group_num_layers = self.group_num_layers.get(group_id, self.num_layers)
                group_block_len = self.group_block_len.get(group_id, self.group_block_len.get(0, []))
                if group_block_len and group_num_layers > 0:
                    group_page_size = sum(group_block_len) // group_num_layers
                else:
                    group_page_size = self.page_size_bytes
                alloc_size = group_page_size * group_num_layers

                group_block_hashes = get_block_hashes(block_hashes, effective_block_size, self.hash_block_size)
                block_ids_by_group = (
                    request.block_ids_by_group_np[group_id]
                    if (request.block_ids_by_group_np is not None and group_id < len(request.block_ids_by_group_np))
                    else request.block_ids_np
                )
                if block_ids_by_group is None:
                    raise RuntimeError(f"Block IDs are not initialized for request {request.req_id}")

                save_start_block = request.save_start_token // effective_block_size
                save_end_block = request.save_end_token // effective_block_size
                if request.load_spec is not None and request.load_spec.can_load:
                    hit_full_blocks = request.load_spec.kvpool_cached_tokens // effective_block_size
                    save_start_block = max(save_start_block, hit_full_blocks)
                # Skip blocks already saved (GVA cached locally from previous
                # request). See _process_save_for_layer_batch for rationale.
                while save_start_block < save_end_block and save_start_block < len(group_block_hashes):
                    key = self._make_layerwise_gva_key(
                        group_id, block_hash_to_str(group_block_hashes[save_start_block])
                    )
                    if key in self._allocated_gvas:
                        save_start_block += 1
                    else:
                        break

                block_gvas: list[int] = []
                new_keys: list[str] = []
                new_positions: list[int] = []
                for blk_idx in range(save_start_block, min(save_end_block, len(group_block_hashes))):
                    key = self._make_layerwise_gva_key(group_id, block_hash_to_str(group_block_hashes[blk_idx]))
                    cached = self._allocated_gvas.get(key)
                    if cached is not None:
                        block_gvas.append(cached)
                    else:
                        new_keys.append(key)
                        new_positions.append(len(block_gvas))
                        block_gvas.append(0)

                if new_keys:
                    new_gvas = self.m_store.batch_alloc(new_keys, [alloc_size] * len(new_keys))
                    if any(gva <= 0 for gva in new_gvas):
                        logger.error(
                            "alloc_gvas FAIL: req=%s group=%d alloc_size=%d new_keys=%d gvas_sample=%s zero_count=%d",
                            request.req_id,
                            group_id,
                            alloc_size,
                            len(new_keys),
                            new_gvas[:5],
                            sum(1 for g in new_gvas if g <= 0),
                        )
                    for pos, key, gva in zip(new_positions, new_keys, new_gvas):
                        if gva > 0:
                            block_gvas[pos] = gva
                            self._allocated_gvas[key] = gva

                logger.info(
                    "alloc_gvas: req=%s group=%d eff_bs=%d save_blocks=[%d,%d) "
                    "new_keys=%d cached_keys=%d alloc_size=%d",
                    request.req_id,
                    group_id,
                    effective_block_size,
                    save_start_block,
                    save_end_block,
                    len(new_keys),
                    len(block_gvas) - len(new_keys),
                    alloc_size,
                )

                # Pad block_gvas to match block_ids length (fill 0 for blocks before save_start)
                full_gvas = [0] * len(block_ids_by_group)
                for i, gva in enumerate(block_gvas):
                    if save_start_block + i < len(full_gvas):
                        full_gvas[save_start_block + i] = gva

                all_group_gvas.append(np.asarray(full_gvas, dtype=np.int64))
                all_group_block_ids.append(np.asarray(block_ids_by_group, dtype=np.int64))

            if all_group_gvas:
                request.block_gvas_by_group_np = all_group_gvas
                request.block_ids_by_group_np = all_group_block_ids
                request.block_gvas_np = all_group_gvas[0]
                request.gva_block_offset = 0

    def _prepare_load_gvas(self, requests: list[ReqMeta]) -> None:
        """Fetch per-rank GVA and acquire read lease for the load path.

        memcache requires batch_copy (read) to find the blob in the per-process
        gvaBlobTracker with a valid lease. The scheduler only checks existence
        (batch_is_exist) to decide the load range; before batch_copy(G2L) the
        worker must, for its own per-rank keys:
          1. batch_get_key_info to fetch the GVA (fills block_gvas_np)
          2. batch_add_lease to register the blob locally + acquire a read lease
        """
        if not self.use_gva_layerwise:
            return
        for request in requests:
            if request.load_spec is None or not request.load_spec.can_load:
                continue
            cached_tokens = request.load_spec.kvpool_cached_tokens
            block_hashes = request.block_hashes

            all_group_load_gvas: list[np.ndarray] = []
            all_group_load_keys: list[str] = []
            for group_id in range(self.num_kv_cache_groups):
                group_block_size = self.grouped_block_size[group_id]
                cache_family = self._get_group_family(self.kv_cache_group_families, group_id)
                ratio = max(infer_cache_family_ratio(cache_family), 1)
                effective_block_size = group_block_size * ratio

                group_block_hashes = get_block_hashes(block_hashes, effective_block_size, self.hash_block_size)
                load_start_block = request.load_spec.vllm_cached_tokens // effective_block_size
                cached_full_blocks = cached_tokens // effective_block_size
                full_blocks = min(cached_full_blocks, len(group_block_hashes))

                block_ids_by_group = (
                    request.block_ids_by_group_np[group_id]
                    if (request.block_ids_by_group_np is not None and group_id < len(request.block_ids_by_group_np))
                    else request.block_ids_np
                )
                if block_ids_by_group is None:
                    all_group_load_gvas.append(np.zeros(0, dtype=np.int64))
                    continue
                full_len = len(block_ids_by_group)

                if load_start_block >= full_blocks:
                    all_group_load_gvas.append(np.zeros(full_len, dtype=np.int64))
                    continue

                keys = [
                    self._make_layerwise_gva_key(group_id, block_hash_to_str(group_block_hashes[i]))
                    for i in range(load_start_block, full_blocks)
                ]
                if not keys:
                    all_group_load_gvas.append(np.zeros(full_len, dtype=np.int64))
                    continue

                key_infos = self.m_store.batch_get_key_info(keys)
                gvas = []
                valid_keys_for_lease = []
                valid_block_ids = []
                invalid_block_ids: list[int] = []
                for idx, (ki, key) in enumerate(zip(key_infos, keys)):
                    sizes = ki.size()
                    gva = ki.gva_list()[0] if sizes and sizes > 0 else 0
                    gvas.append(gva)
                    if gva > 0:
                        valid_keys_for_lease.append(key)
                        block_idx = load_start_block + idx
                        if block_idx < len(block_ids_by_group):
                            valid_block_ids.append(int(block_ids_by_group[block_idx]))
                    else:
                        block_idx = load_start_block + idx
                        if block_idx < len(block_ids_by_group):
                            invalid_block_ids.append(int(block_ids_by_group[block_idx]))
                        logger.warning(
                            "load_gvas: req=%s group=%d got invalid gva=%d (size=%d), block_id=%s load failed",
                            request.req_id,
                            group_id,
                            gva,
                            sizes if sizes else 0,
                            int(block_ids_by_group[block_idx]) if block_idx < len(block_ids_by_group) else "N/A",
                        )

                # Only call batch_add_lease for keys with valid size
                if valid_keys_for_lease:
                    lease_results = self.m_store.batch_add_lease(valid_keys_for_lease, LAYERWISE_READ_LEASE_TTL_MS)
                    # Report lease failures as invalid blocks
                    for i, lease_res in enumerate(lease_results):
                        if lease_res != 0 and i < len(valid_block_ids):
                            invalid_block_ids.append(valid_block_ids[i])
                            logger.warning(
                                "load_gvas: req=%s group=%d lease failed result=%d, block_id=%d load failed",
                                request.req_id,
                                group_id,
                                lease_res,
                                valid_block_ids[i],
                            )
                else:
                    lease_results = []

                # Report invalid blocks to scheduler for recompute.
                # Single-group models can safely report individual block IDs.
                # Multi-group (hybrid) models must not report partial group
                # failures, as the scheduler cannot handle inconsistent KV
                # cache state across groups (see PR #9701 for rationale).
                if invalid_block_ids:
                    if len(request.block_ids_by_group) == 1:
                        with self._invalid_block_ids_lock:
                            self._invalid_block_ids.update(invalid_block_ids)
                    else:
                        logger.error(
                            "KV load failed for hybrid request %s. "
                            "Skip invalid-block fallback to avoid scheduler crash. "
                            "failed_blocks=%s",
                            request.req_id,
                            invalid_block_ids,
                        )
                all_group_load_keys.extend(valid_keys_for_lease)

                logger.info(
                    "load_gvas: req=%s group=%d eff_bs=%d load_blocks=[%d,%d) keys=%d valid_gvas=%d lease_fail=%d",
                    request.req_id,
                    group_id,
                    effective_block_size,
                    load_start_block,
                    full_blocks,
                    len(keys),
                    sum(1 for g in gvas if g > 0),
                    sum(1 for r in lease_results if r != 0),
                )

                # Pad to match block_ids_by_group length, with 0s before load_start_block
                full_gvas = [0] * full_len
                for i, gva in enumerate(gvas):
                    if load_start_block + i < len(full_gvas):
                        full_gvas[load_start_block + i] = gva
                all_group_load_gvas.append(np.asarray(full_gvas, dtype=np.int64))

            if all_group_load_gvas:
                request.load_keys = all_group_load_keys
                request.load_block_gvas_by_group_np = all_group_load_gvas
                request.load_block_gvas_np = all_group_load_gvas[0]
                request.load_gva_block_offset = 0

    def _build_shared_save_data(self) -> None:
        """Build shared block data once and attach to all layer save tasks.

        For GVA path (KVCacheStoreLayerSendingThread): pre-computes
        SharedBlockData via LayerBatchBuilder.build_shared().

        For Key path (KVCacheStoreKeyLayerSendingThread): pre-computes
        cached process_tokens via build_cached_process_tokens().

        In multi-group mode, shared data is built per-group because each
        group has different block_ranges (different effective_block_size).
        """
        if isinstance(self.kv_send_thread, KVCacheStoreLayerSendingThread):
            for group_id in range(self.num_kv_cache_groups):
                first_task = None
                for layer_id in range(self.num_layers):
                    for task in self.layer_save_tasks[layer_id]:
                        if task.group_id == group_id:
                            first_task = task
                            break
                    if first_task:
                        break
                if first_task is None:
                    continue
                shared = self.kv_send_thread.build_shared_data(first_task)
                if shared is not None:
                    for layer_id in range(self.num_layers):
                        for task in self.layer_save_tasks[layer_id]:
                            if task.group_id == group_id:
                                task.shared_block_data = shared
        elif isinstance(self.kv_send_thread, KVCacheStoreKeyLayerSendingThread):
            first_task = None
            for layer_id in range(self.num_layers):
                if self.layer_save_tasks[layer_id]:
                    first_task = self.layer_save_tasks[layer_id][0]
                    break
            if first_task is None:
                return
            cached = self.kv_send_thread.build_cached_process_tokens(first_task)
            if cached is not None:
                for layer_id in range(self.num_layers):
                    for task in self.layer_save_tasks[layer_id]:
                        task.cached_process_tokens = cached

    def _build_shared_load_data(self) -> None:
        """Build shared block data once and attach to all layer load tasks.

        In multi-group mode, shared data is built per-group because each
        group has different block_ranges (different effective_block_size).
        """
        if not isinstance(self.kv_recv_thread, KVCacheStoreLayerRecvingThread):
            return
        for group_id in range(self.num_kv_cache_groups):
            first_task = None
            for layer_id in range(self.num_layers):
                for task in self.layer_load_tasks[layer_id]:
                    if task.group_id == group_id:
                        first_task = task
                        break
                if first_task:
                    break
            if first_task is None:
                continue
            shared = self.kv_recv_thread.build_shared_data(first_task)
            if shared is not None:
                for layer_id in range(self.num_layers):
                    for task in self.layer_load_tasks[layer_id]:
                        if task.group_id == group_id:
                            task.shared_block_data = shared

    def process_layer_data(self, requests: list[ReqMeta]) -> None:
        if not requests:
            return
        for physical_layer in range(self.num_layers):
            group_layers = self.physical_layer_to_group_layers.get(physical_layer, [(0, physical_layer)])
            for group_id, layer_idx_in_group in group_layers:
                self._process_save_for_layer_batch(requests, physical_layer, group_id, layer_idx_in_group)
        self._alloc_gvas_for_save(requests)
        self._build_shared_save_data()
        self._prepare_load_gvas(requests)
        for physical_layer in range(self.num_layers):
            group_layers = self.physical_layer_to_group_layers.get(physical_layer, [(0, physical_layer)])
            for group_id, layer_idx_in_group in group_layers:
                self._process_load_for_layer_batch(requests, physical_layer, group_id, layer_idx_in_group)
        self._build_shared_load_data()

    def _submit_ready_layer_loads(self) -> None:
        assert self.kv_recv_thread is not None
        recv_thread = self.kv_recv_thread

        def submit_layer_load(layer_id: int) -> bool:
            if not self.layer_load_tasks[layer_id]:
                return False
            wait_for_save_layer = None
            attention_start_gate = None
            if layer_id != self.current_layer:
                attention_start_gate = get_attention_compute_start_gate()
            recv_thread.add_request(
                LayerLoadTask(  # type: ignore[arg-type]
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
        if self.current_layer >= self.num_layers:
            return
        assert self.layer_load_finished_events is not None
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
        if self.current_layer >= self.num_layers:
            return
        assert self.sync_save_events is not None
        assert self.layer_save_finished_events is not None
        assert self.kv_send_thread is not None
        send_thread = self.kv_send_thread
        self.sync_save_events[self.current_layer].record()
        if self.layer_save_tasks[self.current_layer]:
            for task in self.layer_save_tasks[self.current_layer]:
                for block_range in task.block_ranges:
                    send_thread.add_stored_request(block_range.request.req_id)
            send_thread.add_request(self.layer_save_tasks[self.current_layer])  # type: ignore[arg-type]
        else:
            self.layer_save_finished_events[self.current_layer].set()
        if self.current_layer == self.num_layers - 1:
            is_finish = self.layer_save_finished_events[self.num_layers - 1].wait(timeout=10)
            if not is_finish:
                logger.info("Layerwise %d save wait timed out", self.current_layer)
            for layer_id in range(self.num_layers):
                if self.layer_save_finished_events[layer_id].is_set():
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
                yield
        else:
            for layer_id in range(self.num_layers):
                yield

    def _make_sub_key_str(self, base_key, effective_rank: int) -> str:
        """Rewrite ``@head_or_tp_rank:<local>`` in base_key.to_string() to ``<effective_rank>``.

        Under TP mismatch, both sides address the pool at the effective_tp_size
        namespace rather than the local TP rank.
        """
        return self._replace_key_field(base_key.to_string(), "head_or_tp_rank", effective_rank)

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
        for start, end, base_key, block_id in self.token_database.process_tokens_with_block_ids(
            token_len,
            block_hashes,
            block_ids,
            mask_num,
        ):
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
    ) -> None:
        keys, addrs, sizes, key_block_ids = self._build_tp_mismatch_keys_and_addrs(
            block_hashes, block_ids, token_len, mask_num
        )
        if not keys:
            return
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
        ret = self.m_store.get(keys_c, addrs_c, sizes_c)
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

    def _store_kv_tp_mismatch(self, req_meta: ReqMeta) -> None:
        send_thread = self.kv_send_thread
        if send_thread is None:
            return
        req_id = req_meta.req_id
        if not send_thread.is_stored_request(req_id):  # type: ignore[attr-defined]
            return
        try:
            token_len = req_meta.token_len_chunk
            block_ids = req_meta.block_ids_by_group[0]
            keys, addrs, sizes, _ = self._build_tp_mismatch_keys_and_addrs(
                req_meta.block_hashes, block_ids, token_len, mask_num=0
            )
            if not keys:
                return
            exists_states = send_thread.lookup(keys)  # type: ignore[attr-defined]
            missing_indices = [i for i, exists in enumerate(exists_states) if not exists]
            if not missing_indices:
                return
            keys = [keys[i] for i in missing_indices]
            addrs = [addrs[i] for i in missing_indices]
            sizes = [sizes[i] for i in missing_indices]
            if req_meta.current_event is not None:
                req_meta.current_event.synchronize()
            logger.debug(
                "KV pool worker tp_mismatch put req=%s keys=%d sample_keys=%s",
                req_id,
                len(keys),
                keys[:3],
            )
            self.m_store.put(keys, addrs, sizes)

            if self.enable_kv_events:
                event_block_size = (
                    req_meta.original_block_size[0]
                    if isinstance(req_meta.original_block_size, list)
                    else req_meta.original_block_size
                )
                stored_events: list[BlockStored] = []
                prev_key = None
                for idx, (start, end, _base_key) in enumerate(
                    self.token_database.process_tokens(token_len, req_meta.block_hashes)
                ):
                    if idx >= len(req_meta.block_hashes):
                        break
                    block_hash = maybe_convert_block_hash(req_meta.block_hashes[idx])
                    token_ids = req_meta.token_ids[start:end] if req_meta.token_ids is not None else None
                    stored_events.append(
                        BlockStored(
                            block_hashes=[block_hash],
                            parent_block_hash=prev_key,
                            token_ids=token_ids,
                            block_size=event_block_size,
                            lora_id=None,
                            medium="cpu",
                            lora_name=None,
                        )
                    )
                    prev_key = block_hash
                if stored_events:
                    send_thread.update_kv_event(stored_events)  # type: ignore[attr-defined]
        finally:
            send_thread.dec_stored_request(req_id)  # type: ignore[attr-defined]

    def get_finished(self, finished_req_ids: set[str], meta: AscendConnectorMetadata) -> tuple[set[str], set[str]]:
        if self.kv_send_thread is not None:
            send_thread = self.kv_send_thread
            for req_id in meta.preempted_req_ids:
                if isinstance(send_thread, (KVCacheStoreSendingThread, KVCacheStoreLayerSendingThread)):
                    send_thread.delete_finished_stored_request(req_id)
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

    def ensure_store_initialized(self) -> None:
        ensure_initialized = getattr(self.m_store, "ensure_initialized", None)
        if ensure_initialized is not None:
            ensure_initialized()

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
                keys: list[str] = []
                starts = []
                ends = []
                for start, end, key in self.token_database.process_tokens(
                    token_len,
                    block_hashes,
                    kv_cache_group_id=group_id,
                ):
                    if use_layerwise:
                        keys_multi_layer = key.split_layers(self.num_layers)
                        for layer_key in keys_multi_layer:
                            keys.append(layer_key.to_string())
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
        expanded: list[str] = []
        for pp_rank in range(self.pp_size):
            for tp_rank in range(self.get_group_tp_size(group_id)):
                for key in keys:
                    tp_key = self._replace_key_field(key, "head_or_tp_rank", tp_rank)
                    expanded.append(self._replace_key_field(tp_key, "pp_rank", pp_rank))
        return expanded

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
        return self._expand_lookup_keys_by_rank([key], group_id)

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
            for _, _, key in self.token_database.process_tokens(
                token_len,
                block_hashes,
                kv_cache_group_id=group_id,
            ):
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
            ExternalCachedBlockPool(self.hash_block_size, exists),
            apply_eagle=False,
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
                use_layerwise,
                include_all_ranks=True,
            )
            if coordinator_hit is not None:
                return coordinator_hit
            for group_id in kv_cache_group_ids:
                keys: list[str] = []
                starts = []
                ends = []
                for start, end, key in self.token_database.process_tokens(
                    token_len,
                    block_hashes,
                    kv_cache_group_id=group_id,
                ):
                    if use_layerwise:
                        keys_multi_layer = key.split_layers(self.num_layers)
                        for layer_key in keys_multi_layer:
                            keys.append(layer_key.to_string())
                    else:
                        keys.append(key.to_string())
                    starts.append(start)
                    ends.append(end)

                if not keys:
                    return 0

                multi_tp_keys = self._expand_lookup_keys_by_rank(keys, group_id)
                num_ranks = len(multi_tp_keys) // len(keys)
                res = self.m_store.exists(multi_tp_keys)  # type: ignore[assignment]
                num_block = len(keys)
                if use_layerwise:
                    res = self.check_all_layers_exists(res, self.num_layers)
                    num_block = len(keys) // self.num_layers
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
        if self.use_mamba and isinstance(self.kv_send_thread, KVCacheStoreSendingThread):
            if ce := self.kv_send_thread.get_completed_events():
                return AscendStoreKVConnectorWorkerMetadata(ce)
        return None
