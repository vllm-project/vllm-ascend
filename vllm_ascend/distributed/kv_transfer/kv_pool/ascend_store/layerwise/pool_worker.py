from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import Any, cast

import torch
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tp_group,
)
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import (
    get_attention_compute_start_gate,
    reset_attention_compute_start_gate,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    get_backend_class,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import (
    require_aligned_batch_results,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.session_tracker import (
    LayerwiseSessionTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_cache import (
    DSparkPrefixCache,
    DSparkPrefixKeys,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVTransferThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreLayerSendingThreadBase,
    LayerBatchBuilder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    LayerwiseCacheConfig,
    get_layerwise_kv_cache_specs,
    get_layerwise_reuse_layout,
    has_shared_kv_sources,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import (
    LayerBlockRange,
    LayerLoadTask,
    LayerTransferTask,
    SharedBlockData,
    get_partial_block_index,
    masked_block_runs,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.planes import make_data_plane
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    ReqMeta,
    block_hash_to_str,
    get_block_hashes,
    get_group_block_size,
    infer_dcp_mismatch_info,
    is_kv_save_role,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import BaseKVPoolWorker

# PERF-TUNE switch: defer the last layer's save drain out of the forward
# critical path; the drain runs at the start of the next step's start_load_kv.
# Opt-in (default off): the synchronous inline drain is the long-standing
# contract that layerwise tests (e.g. test_mooncake_pipeline) and downstream
# readers rely on, and deferring widens the window in which a subsequent
# request can observe a partially published prefix.
_LW_DEFER_LAST_SAVE = os.environ.get("VLLM_ASCEND_LW_DEFER_LAST_SAVE", "0") == "1"


class LayerwiseKVPoolWorker(BaseKVPoolWorker):
    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwise: bool,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        super().__init__(vllm_config, use_layerwise, kv_cache_config)
        self.external_slot_release_waiter: Callable[[int], None] | None = None
        self._put_started_keys: set[str] = set()
        self._put_started_keys_lock = threading.Lock()
        self._load_session_lock = threading.Lock()
        self._layer_load_aborted = threading.Event()
        self._layerwise_session_tracker = LayerwiseSessionTracker()
        self._current_layerwise_request_ids: set[str] = set()
        self._current_layerwise_last_chunk_req_ids: set[str] = set()
        # (tag, req_id, group_id, start_block, end_block) -> runs, cleared per
        # step in process_layer_data(). masked_block_runs inputs are
        # layer-invariant, so only the first layer pays the O(blocks) scan.
        self._masked_runs_cache: dict[tuple[str, str, int, int, int], list[tuple[int, int]]] = {}
        self._dspark_prefix_keys: DSparkPrefixKeys | None = None
        self.dspark_prefix_cache: DSparkPrefixCache | None = None
        self._init_layerwise_config()

    def _init_kv_transfer_config(self, vllm_config, extra_config, use_layerwise, kv_cache_config) -> None:
        backend_name = extra_config.get("backend", "mooncake").lower()
        self.layerwise_backend = get_backend_class(backend_name)
        self.layerwise_data_plane = (
            self.layerwise_backend.layerwise_data_plane if self.layerwise_backend is not None else None
        )
        self.use_block_key_layerwise = self.layerwise_data_plane == "block_key"
        self.use_layerwise_transfer = self.layerwise_data_plane == "gva"
        self.has_shared_kv = self.use_layerwise_transfer and has_shared_kv_sources(vllm_config)
        self._step_load_keys: set[str] = set()
        # Request ID -> cache group ID -> request block index -> private pool key.
        self._snapshot_keys: dict[str, dict[int, dict[int, str]]] = {}
        super()._init_kv_transfer_config(vllm_config, extra_config, use_layerwise, kv_cache_config)
        self.block_key_hybrid = self.use_block_key_layerwise and self.use_hybrid
        self._dspark_draft_layer_names = (
            set(getattr(kv_cache_config, "dspark_draft_layer_names", ())) if self.use_layerwise_transfer else set()
        )
        self.dflash_draft_layer_names = (
            frozenset(getattr(kv_cache_config, "dflash_draft_layer_names", ()))
            if self.use_layerwise_transfer
            else frozenset()
        )
        self._dflash_save_pending = False
        if (
            self.backend_name != "memcache"
            and not self.use_block_key_layerwise
            and len(self.cacheable_group_ids) != len(self.original_block_size)
        ):
            raise ValueError("AscendStore private KV state requires Mooncake block-key or non-layerwise transfer")
        self.layerwise_keys = (
            self.layerwise_backend.bind_layerwise_keys(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                model_name=self.model_name,
                use_hybrid=self.use_hybrid,
                grouped_block_size=self.grouped_block_size,
            )
            if self.layerwise_backend is not None and self.layerwise_data_plane is not None
            else None
        )
        self.layerwise_max_transfer_blocks = int(extra_config.get("layerwise_max_transfer_blocks", 0))
        self.layerwise_max_transfer_bytes = int(extra_config.get("layerwise_max_transfer_bytes", 0))
        # The plane routes the plane-specific parts of each transfer step; it
        # resolves worker helpers at call time, so post-init flag adjustments
        # in tests keep affecting it.
        self._data_plane = make_data_plane(self)

    def _compute_use_hybrid(self, vllm_config, kv_cache_groups) -> bool:
        return super()._compute_use_hybrid(vllm_config, kv_cache_groups) or (
            self.use_block_key_layerwise and kv_cache_groups is not None and len(kv_cache_groups) > 1
        )

    def _init_key_head_config(self, model_config) -> None:
        super()._init_key_head_config(model_config)
        if self.tp_mismatch:
            raise ValueError(
                f"TP mismatch (local_tp={self.tp_size}, peer_tp={self.peer_tp_size}) "
                "is not supported with layerwise KV transfer (use_layerwise=True). "
                "The layerwise threads do not implement TP-mismatch handling."
            )
        if self.dcp_size > 1 and self.pcp_size == 1:
            expected_dcp_rank = self.tp_rank % self.dcp_size
            if expected_dcp_rank != self.dcp_rank:
                raise ValueError(
                    f"Layerwise DCP rank mapping mismatch: tp_rank={self.tp_rank}, "
                    f"dcp_size={self.dcp_size} maps to dcp_rank={expected_dcp_rank}, "
                    f"but the DCP group reports {self.dcp_rank}."
                )
        # Both PD sides must derive the same GVA shard layout.
        if self.kv_role in ("kv_producer", "kv_consumer") and infer_dcp_mismatch_info(
            self.kv_role, self._extra_config, self.dcp_size, self.pcp_size
        ):
            peer_role = "prefill" if self.kv_role == "kv_consumer" else "decode"
            raise ValueError(
                f"Decode-context-parallel mismatch in PD-disaggregation "
                f"(local dcp_size={self.dcp_size}, local pcp_size={self.pcp_size}, "
                f"peer role={peer_role}) is not supported with layerwise KV "
                f"transfer. Both the producer and consumer must use the same "
                f"dcp_size/pcp_size so the layerwise GVA shard layout is consistent."
            )

    def _init_layerwise_config(self) -> None:
        self._layerwise_reuse_layout = None

        # Build mapping: physical_layer -> [(group_id, layer_idx_in_group), ...]
        # layer_idx_in_group is the index of the physical layer within the
        # group (not the index in layer_names). Multiple cache names at the
        # same physical layer are treated as entries of one layer.
        self.physical_layer_to_group_layers: dict[int, list[tuple[int, int]]] = {}
        self._global_to_local_layer: dict[int, int] = {}
        self._reused_layers_by_group: dict[int, set[int]] = {}
        self._recurrent_layers: set[int] = set()
        stage_num_layers = self.num_layers
        dflash_draft_names: frozenset[str] = getattr(self, "dflash_draft_layer_names", frozenset())
        self._dflash_draft_start_layer: int | None = None
        # Defaults for partial initialization (unit tests construct the worker
        # without the full _init_parallelism_info path).
        if not hasattr(self, "layerwise_key_layers"):
            self.layerwise_key_layers = 0
            self.layerwise_key_layer_offset = 0

        # Layerwise pool-key layer addressing. Without pipeline parallel the
        # key space is the local layer range [0, num_layers) with offset 0
        # (identical to previous releases). With PP>1 each stage emits global
        # layer ids derived from its actual stage start, so producers and
        # consumers share one global layer key space regardless of which
        # stage computed the layer.
        if getattr(self, "pp_size", 1) > 1 and getattr(self, "total_layers", None) is not None:
            self.layerwise_key_layers = self.num_layers
            self.layerwise_key_layer_offset = self.pp_layer_offset
            logger.info(
                "Layerwise PP key space: stage %d/%d writes global layers [%d, %d) (total %d).",
                self.pp_rank,
                self.pp_size,
                self.layerwise_key_layer_offset,
                self.layerwise_key_layer_offset + self.num_layers,
                self.total_layers,
            )
        else:
            self.layerwise_key_layers = self.num_layers
            self.layerwise_key_layer_offset = 0

        if self.kv_cache_config is not None:
            draft_names = getattr(self, "_dspark_draft_layer_names", ())
            layer_specs = get_layerwise_kv_cache_specs(self.kv_cache_config)
            target_specs = {name: spec for name, spec in layer_specs.items() if name not in draft_names}
            if self.use_layerwise_transfer:
                self._layerwise_reuse_layout = get_layerwise_reuse_layout(self.kv_cache_config)
                if (
                    draft_names
                    and self._layerwise_reuse_layout is None
                    and getattr(self, "_dspark_prefix_keys", None) is None
                ):
                    # Without scratch reuse, retain the original target+draft
                    # pool for prefix storage and load/save addressing.
                    self._dspark_draft_layer_names = draft_names = set()
                    target_specs = layer_specs
            physical_layers = {
                self._extract_physical_layer_index(layer_name)
                for group_spec in self.kv_cache_config.kv_cache_groups
                for layer_name in group_spec.layer_names
                if layer_name not in draft_names
            }
            if getattr(self, "has_shared_kv", False):
                # Execution layers include holes; group ordinals stay dense.
                physical_layers.update(range(getattr(self.hf_config, "num_hidden_layers", self.num_layers)))
                self.num_layers = max(physical_layers) + 1
            if physical_layers:
                self._global_to_local_layer = {
                    global_layer: local_layer for local_layer, global_layer in enumerate(sorted(physical_layers))
                }
                if getattr(self, "use_layerwise", False) or self.use_layerwise_transfer:
                    self._recurrent_layers = {
                        self._global_to_local_layer[self._extract_physical_layer_index(name)]
                        for name, spec in target_specs.items()
                        if isinstance(spec, MambaSpec)
                    }
                effective_num_layers = max(self.num_layers, len(physical_layers))
                if effective_num_layers != self.num_layers:
                    logger.info(
                        "KVPoolWorker: updated num_layers %d -> %d from cache group layout.",
                        self.num_layers,
                        effective_num_layers,
                    )
                    self.num_layers = effective_num_layers
                self._init_dflash_draft_layers(target_specs.keys() & dflash_draft_names)
        if self.kv_cache_config is not None and self.num_kv_cache_groups > 1:
            for group_id, group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
                if group_id not in self.cacheable_group_ids:
                    continue
                physical_layers = set()
                for layer_name in group_spec.layer_names:
                    if layer_name in draft_names:
                        continue
                    physical_layer = self._extract_physical_layer_index(layer_name)
                    physical_layers.add(self._global_to_local_layer[physical_layer])
                phys_to_layer_idx = {
                    physical_layer: layer_index for layer_index, physical_layer in enumerate(sorted(physical_layers))
                }

                # Add one entry per unique physical layer (no duplicates)
                for physical_layer, layer_idx_in_group in phys_to_layer_idx.items():
                    existing = self.physical_layer_to_group_layers.setdefault(physical_layer, [])
                    entry = (group_id, layer_idx_in_group)
                    if entry not in existing:
                        existing.append(entry)

                if getattr(self, "use_layerwise", False) or self.use_layerwise_transfer:
                    logger.info(
                        "layerwise group %d: %d layer_names, %d unique physical layers",
                        group_id,
                        len(group_spec.layer_names),
                        len(phys_to_layer_idx),
                    )

        self.layer_load_tasks: list[list[LayerTransferTask]] = [[] for _ in range(self.num_layers)]
        self.layer_save_tasks: list[list[LayerTransferTask]] = [[] for _ in range(self.num_layers)]
        self.layer_load_finished_events: list[threading.Event] | None = None
        self.layer_save_finished_events: list[threading.Event] | None = None

        self.next_layer_to_submit = 0
        self.layerwise_offload = False
        self.prefetch_layer_map: dict[int, tuple[int, ...]] = {}
        if self.use_layerwise_transfer:
            if self._layerwise_reuse_layout is None:
                stage_start = self.layerwise_key_layer_offset
                stage_layers = range(stage_start, stage_start + stage_num_layers)
                total_layers = getattr(self, "total_layers", None)
                base_layers = getattr(self.hf_config, "num_hidden_layers", stage_num_layers)
                layerwise_settings = LayerwiseCacheConfig.parse(
                    total_layers if isinstance(total_layers, int) else base_layers,
                    self._extra_config,
                    stage_layers=stage_layers,
                )
                self.num_prefetch_layers = layerwise_settings.num_prefetch_layers
            else:
                self.prefetch_layer_map = self._remap_layout_to_stage_local(
                    self._layerwise_reuse_layout.prefetch_layer_map, self._global_to_local_layer
                )
                self.layerwise_offload = self._layerwise_reuse_layout.has_layer_reuse
                self.num_prefetch_layers = self._layerwise_reuse_layout.num_prefetch_layers
                for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups):
                    layers = {
                        self._global_to_local_layer[self._layerwise_reuse_layout.cache_layer_indices[name]]
                        for name in group.layer_names
                        if name in self._layerwise_reuse_layout.reused_cache_names
                    }
                    if layers:
                        self._reused_layers_by_group[group_id] = layers
        else:
            self.num_prefetch_layers = 2
            if self.use_layerwise:
                prefetch_layers = self._extra_config.get("layerwise_prefetch_layers", 2)
                if isinstance(prefetch_layers, bool):
                    raise ValueError("layerwise_prefetch_layers must be a positive integer")
                try:
                    self.num_prefetch_layers = int(prefetch_layers)
                except (TypeError, ValueError) as exc:
                    raise ValueError("layerwise_prefetch_layers must be a positive integer") from exc
                if self.num_prefetch_layers <= 0:
                    raise ValueError("layerwise_prefetch_layers must be a positive integer")
        self.sync_save_events: list[torch.npu.Event] | None = None

        logger.info(
            "layerwise config: num_layers=%d num_groups=%d physical_layer_to_group_layers_sample=%s",
            self.num_layers,
            self.num_kv_cache_groups,
            {k: v for k, v in list(self.physical_layer_to_group_layers.items())[:3]},
        )

    def _init_dflash_draft_layers(self, draft_names: set[str]) -> None:
        draft_layers = {self._global_to_local_layer[self._extract_physical_layer_index(name)] for name in draft_names}
        if not draft_layers:
            return
        first_draft_layer = min(draft_layers)
        if draft_layers != set(range(first_draft_layer, self.num_layers)):
            raise ValueError("DFlash draft layers must follow target layers in the pool")
        self._dflash_draft_start_layer = first_draft_layer

    @staticmethod
    def _remap_layout_to_stage_local(
        prefetch_layer_map: dict[int, tuple[int, ...]],
        global_to_local: dict[int, int],
    ) -> dict[int, tuple[int, ...]]:
        """Translate a global-indexed reuse layout to stage-local indices."""
        return {
            global_to_local[layer]: tuple(global_to_local[source] for source in sources)
            for layer, sources in prefetch_layer_map.items()
        }

    def _build_group_layer_builders(self) -> list[LayerBatchBuilder]:
        builders = []
        # Sync layerwise_key_layers with the (possibly updated) num_layers.
        # When num_kv_cache_groups == 1 and extra layers (e.g. MTP or
        # spec-decode draft layers) are present, num_layers is updated in
        # register_kv_caches to include them; the key layer count must follow
        # or those extra layers are skipped during the save/load lifecycle. A
        # block-key store owns a stage-local object, so its count stays local
        # under PP. A GVA store retains the global PP layout and offset below.
        if (
            self.use_block_key_layerwise
            or (self.num_kv_cache_groups == 1 and getattr(self, "pp_size", 1) == 1)
            or (self.use_layerwise_transfer and getattr(self, "pp_size", 1) > 1)
        ):
            self.layerwise_key_layers = self.num_layers
        for group_id in range(self.num_kv_cache_groups):
            group_num_layers = self.group_num_layers.get(group_id, self.num_layers)
            group_page_size = (
                sum(self.group_block_len[group_id])
                if self.use_block_key_layerwise
                else self._global_group_alloc_size(group_id)
            )
            # Global byte offsets for the shared GVA region:
            #   PP: each stage writes its local layers at
            #       (pp_layer_offset + local_layer) * per_layer_bytes.
            #   DCP (shard-major): ranks sharing head_or_tp_rank (put_step>1)
            #       hold different context shards of the same logical block
            #       and share ONE region; shard d occupies the byte range
            #       [d * shard_stride, (d+1) * shard_stride) where the stride
            #       is the group byte total aligned up to the GVA hugepage
            #       size (unaligned strides yield misaligned GVA addresses).
            #       With put_step == 1 every rank owns a distinct region key
            #       and no shard separation is needed.
            layer_byte_offset = self._group_pp_byte_offset(group_id)
            gbl = self.group_block_len.get(group_id) or []
            if not self.use_block_key_layerwise and gbl and group_num_layers > 0:
                if self.dcp_size > 1 and self.put_step > 1:
                    # Use the GLOBAL region size (PP-aware) as the basis for
                    # the per-shard stride; under PP>1 the local sum(gbl) only
                    # covers this stage's layers and would under-allocate.
                    shard_stride = self._global_group_alloc_size(group_id) // self.dcp_size
                    shard_idx = self.dcp_rank
                    layer_byte_offset += shard_idx * shard_stride
            builders.append(
                LayerBatchBuilder(
                    self.token_database,
                    group_page_size,
                    group_num_layers,
                    group_id=group_id,
                    layer_byte_offset=layer_byte_offset,
                )
            )
        return builders

    def configure_dspark_prefix_cache(self, keys: DSparkPrefixKeys) -> None:
        self._dspark_prefix_keys = keys
        self._dspark_draft_layer_names = set(getattr(self.kv_cache_config, "dspark_draft_layer_names", ()))
        assert self.kv_cache_config is not None
        # Companion objects own draft-only groups; retain original group IDs
        # and the existing cacheability policy for the ordinary target pool.
        self.cacheable_group_ids = [
            group_id
            for group_id in self.cacheable_group_ids
            if any(
                name not in self._dspark_draft_layer_names
                for name in self.kv_cache_config.kv_cache_groups[group_id].layer_names
            )
        ]
        # A non-reuse initial plan may have included draft physical layers.
        # Reset the target hook extent before rebuilding its separate view.
        self.num_layers = len(
            {
                self._extract_physical_layer_index(name)
                for group in self.kv_cache_config.kv_cache_groups
                for name in group.layer_names
                if name not in self._dspark_draft_layer_names
            }
        )
        self._init_layerwise_config()

    def shutdown(self) -> None:
        if self._closed:
            return
        threads = [thread for thread in (self.kv_send_thread, self.kv_recv_thread) if thread is not None]
        for thread in threads:
            thread.stop()
        for thread in threads:
            if thread.is_alive():
                thread.join()
        try:
            if self._step_load_keys:
                result = self.m_store.batch_remove_lease(list(self._step_load_keys))
                if result != 0:
                    logger.error("AscendStore shutdown lease cleanup returned %s", result)
                self._step_load_keys.clear()
        finally:
            self.m_store.close()
        self._closed = True

    def set_external_slot_release_waiter(self, waiter: Callable[[int], None]) -> bool:
        """Accept the MultiConnector composite slot-release waiter.

        Only the layerwise transfer mode has receive threads waiting on an
        externally supplied slot-release callback, so the gate lives here
        (at the data-plane consumer) instead of at the connector proxy.
        """
        if not self.use_layerwise_transfer:
            return False
        self.external_slot_release_waiter = waiter
        # The receive thread snapshots the waiter at construction; if it is
        # already running, hand the waiter over directly so a late
        # registration still takes effect.
        if isinstance(self.kv_recv_thread, KVCacheStoreLayerRecvingThread):
            self.kv_recv_thread.external_slot_release_waiter = waiter
        return True

    def _global_group_alloc_size(self, group_id: int) -> int:
        if getattr(self, "has_shared_kv", False) and self.group_block_len.get(group_id):
            return sum(self.group_block_len[group_id])
        return super()._global_group_alloc_size(group_id)

    def _start_kv_transfer_threads(self) -> None:
        if self.use_layerwise_transfer:
            self.m_store.ensure_initialized()
        elif self.use_block_key_layerwise:
            self.m_store.validate_layerwise_support()
        if len(self.layer_load_tasks) != self.num_layers:
            self.layer_load_tasks = [[] for _ in range(self.num_layers)]
            self.layer_save_tasks = [[] for _ in range(self.num_layers)]
        if self._transfer_threads_started:
            return
        self.layer_load_finished_events = [threading.Event() for i in range(self.num_layers)]
        self.layer_save_finished_events = [threading.Event() for i in range(self.num_layers)]
        self.sync_save_events = [torch.npu.Event() for i in range(self.num_layers)]
        can_save = is_kv_save_role(self.kv_role, self.save_decode_cache)
        if (self.use_block_key_layerwise or self.use_layerwise_transfer) and can_save:
            ready_event_sending = threading.Event()
            self.kv_send_thread = KVCacheStoreLayerSendingThread(
                self.m_store,
                self.token_database,
                self.block_size,
                self.tp_rank,
                self.tp_size,
                self.dcp_size,
                self.page_size_bytes,
                ready_event_sending,
                self.num_layers,
                self.layer_save_finished_events,
                self.sync_save_events,
                self.layerwise_max_transfer_blocks,
                self.layerwise_max_transfer_bytes,
                group_builders=self._build_group_layer_builders(),
                put_started_keys=self._put_started_keys,
                put_started_keys_lock=self._put_started_keys_lock,
                session_tracker=self._layerwise_session_tracker if self.use_block_key_layerwise else None,
                raise_on_transfer_failure=self.layerwise_offload,
                invalid_block_ids=self._invalid_block_ids,
                invalid_block_ids_lock=self._invalid_block_ids_lock,
            )
            self.kv_send_thread.start()
            ready_event_sending.wait()
        ready_event = threading.Event()
        self.kv_recv_thread = KVCacheStoreLayerRecvingThread(
            self.m_store,
            self.token_database,
            self.block_size,
            self.tp_rank,
            self.tp_size,
            self.dcp_size,
            self.page_size_bytes,
            ready_event,
            self.layer_load_finished_events,
            self.layer_save_finished_events,
            self.sync_save_events,
            self.num_layers,
            self.h2d_stagger_us,
            self.layerwise_max_transfer_blocks,
            self.layerwise_max_transfer_bytes,
            group_builders=self._build_group_layer_builders(),
            external_slot_release_waiter=self.external_slot_release_waiter,
            save_failure_checker=(self.kv_send_thread.raise_if_failed if self.kv_send_thread is not None else None),
            invalid_block_ids=self._invalid_block_ids,
            invalid_block_ids_lock=self._invalid_block_ids_lock,
            load_abort_event=self._layer_load_aborted,
        )
        self.kv_recv_thread.save_disabled = self._save_disabled
        if self.kv_send_thread is not None:
            self.kv_send_thread.save_disabled = self._save_disabled
        self.kv_recv_thread.start()
        ready_event.wait()
        self._transfer_threads_started = True

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self.use_recoverssm:
            kv_caches = self._recoverssm_store_caches(kv_caches)
        draft_names = set(getattr(self, "_dspark_draft_layer_names", ()))
        draft_caches = {name: cache for name, cache in kv_caches.items() if name in draft_names}
        target_caches = (
            {name: cache for name, cache in kv_caches.items() if name not in draft_names} if draft_names else kv_caches
        )
        self._set_num_blocks(target_caches)
        self._register_dspark_prefix_cache(draft_caches)
        auxiliary_caches = draft_caches if getattr(self, "dspark_prefix_cache", None) is not None else None
        self._register_kv_caches(target_caches, auxiliary_caches, draft_names)

    def _register_dspark_prefix_cache(self, draft_caches: dict[str, torch.Tensor]) -> None:
        prefix_keys: DSparkPrefixKeys | None = getattr(self, "_dspark_prefix_keys", None)
        if prefix_keys is not None and self.pp_rank == self.pp_size - 1:
            self.dspark_prefix_cache = DSparkPrefixCache(
                prefix_keys,
                self.m_store,
                self.tp_rank,
                self.block_size,
                self.hash_block_size,
                self._copy_dspark_prefix,
            )
            assert self.kv_cache_config is not None
            layer_group_ids = {
                name: group_id
                for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups)
                for name in group.layer_names
                if name in self._dspark_draft_layer_names
            }
            self.dspark_prefix_cache.register(draft_caches, layer_group_ids, self.original_block_size, self.num_blocks)

    def _copy_dspark_prefix(self, gvas: Any, addrs: Any, sizes: Any, direction: int) -> int:
        assert isinstance(self.kv_recv_thread, KVTransferThread)
        return self.kv_recv_thread._batch_copy_with_limits(
            gvas,
            addrs,
            sizes,
            direction,
            self.layerwise_max_transfer_blocks,
            self.layerwise_max_transfer_bytes,
        )

    def start_load_kv(self, metadata: AscendConnectorMetadata) -> None:
        # wait_for_layer_load submits loads from tasks prepared at bind.
        # A deferred call must not reset state between target and MTP.
        return

    def get_finished(self, finished_req_ids: set[str], meta: AscendConnectorMetadata) -> tuple[set[str], set[str]]:
        if self._snapshot_keys:
            for req_id in finished_req_ids | meta.preempted_req_ids:
                self._snapshot_keys.pop(req_id, None)
        if self._step_load_keys:
            if self.kv_recv_thread is not None:
                self._drain_attention_transfers(full=True)
            keys, self._step_load_keys = list(self._step_load_keys), set()
            result = self.m_store.batch_remove_lease(keys)
            if result != 0:
                logger.error("AscendStore lease cleanup returned %s; backend TTL will reclaim leases", result)
        if self.use_block_key_layerwise:
            self._release_layerwise_requests_terminal(finished_req_ids | meta.preempted_req_ids)
        return super().get_finished(finished_req_ids, meta)

    def _build_lookup_keys(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        group_id: int,
    ) -> tuple[list[str], list[int], list[int]]:
        keys: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        for start, end, pool_key in self.token_database.process_tokens(
            token_len, block_hashes, kv_cache_group_id=group_id
        ):
            keys.extend(
                item.to_string()
                for item in pool_key.split_layers(
                    getattr(self, "layerwise_key_layers", 0) or self.num_layers,
                    getattr(self, "layerwise_key_layer_offset", 0),
                )
            )
            starts.append(start)
            ends.append(end)
        return keys, starts, ends

    def prepare_layerwise_step(self, metadata: AscendConnectorMetadata) -> None:
        """Prepare per-step hook state when the runner binds metadata."""
        assert self.use_layerwise
        self._drain_deferred_last_save()
        # The final-layer fence drains this rank only. Readers of replicated
        # KV must also wait for the writer before querying the previous step.
        # Use configuration, never local requests/tasks, to gate the collective.
        if self.use_layerwise_transfer and self.layerwise_offload and self.tp_size > 1:
            get_tp_group().barrier()
        self.current_layer = 0
        self._dflash_save_pending = False
        self.layerwise_retrievers: list[Any] = []
        self.next_layer_to_submit = 0
        reset_attention_compute_start_gate()
        if self.use_kvpp:
            assert self.layer_load_finished_events is not None
            for event in self.layer_load_finished_events:
                event.clear()
        self._attention_saved_layers = set()
        self._save_disabled.clear()
        if self.kv_recv_thread is not None:
            self.kv_recv_thread.load_disabled.clear()
        self.process_layer_data(metadata.requests)

    def _is_layerwise_save_leader(self) -> bool:
        """Exactly one rank per (pcp, dcp, head_or_tp) group saves/allocates.

        Without DCP the plain ``tp_rank % put_step == 0`` dedup is correct
        because all TP ranks hold identical KV. With DCP the context is
        sharded: ranks sharing (dcp_rank, head_or_tp_rank) hold the same
        shard, while the put_step groups span DIFFERENT shards -- the plain
        dedup would drop every non-zero DCP shard from the pool. The leader
        is the smallest tp_rank inside the rank's own (dcp, head) group.
        """
        if self.dcp_size <= 1:
            return self.tp_rank % self.put_step == 0
        head = self.tp_rank // self.put_step
        peers = [
            t for t in range(head * self.put_step, (head + 1) * self.put_step) if t % self.dcp_size == self.dcp_rank
        ]
        if not peers:
            return True
        return self.tp_rank == min(peers)

    def _is_layerwise_save_owner(self, group_id: int = 0) -> bool:
        if not is_kv_save_role(self.kv_role, self.save_decode_cache):
            return False

        # Each rank must save its own Mamba state shard.
        if group_id in self.sharded_mamba_group_ids:
            return True

        # Deduplicate PCP replicas only when DCP does not shard their KV.
        if self.dcp_size == 1 and self.pcp_rank != 0:
            return False

        return self._is_layerwise_save_leader() or self.group_uses_align_state[group_id]

    def _groups_for_layerwise_transfer(self, local_layer: int) -> list[tuple[int, int]]:
        if self.use_block_key_layerwise:
            # The map and buffer builders both use stage-local physical layer
            # indices. A group's dense layer index is a third, distinct axis.
            if self.num_kv_cache_groups == 1 and not getattr(self, "use_kvpp", False):
                return [(0, local_layer)]
            groups = self.physical_layer_to_group_layers.get(local_layer)
            if groups is None:
                if getattr(self, "use_kvpp", False):
                    return []
                raise RuntimeError(f"Mooncake layerwise: no KV cache group for local layer {local_layer}")
            return [(group_id, layer_idx) for group_id, layer_idx in groups if group_id in self.cacheable_group_ids]
        # GVA groups use stage-local indices too; PP offsets apply to pool keys and remote addresses.
        return self.physical_layer_to_group_layers.get(
            local_layer, [] if getattr(self, "has_shared_kv", False) else [(0, local_layer)]
        )

    def _masked_block_runs_cached(
        self,
        tag: str,
        request: ReqMeta,
        group_id: int,
        start_block: int,
        end_block: int,
        mask: Sequence[bool] | None,
    ) -> list[tuple[int, int]]:
        # masked_block_runs is O(end_block - start_block); the layer loop
        # calls it with identical (mask, range) for every layer, so memoize
        # per step. start_block/end_block stay in the key: the load path
        # varies them across independent layers.
        key = (tag, request.req_id, group_id, start_block, end_block)
        runs = self._masked_runs_cache.get(key)
        if runs is None:
            runs = masked_block_runs(mask, start_block, end_block)
            self._masked_runs_cache[key] = runs
        return runs

    def _make_layerwise_full_key(self, group_id: int, block_hash_hex: str) -> str:
        """Use the backend-bound layout shared with scheduler hit checks."""
        assert self.layerwise_keys is not None
        return self.layerwise_keys.make_full_key(
            group_id,
            block_hash_hex,
            self.metadata[group_id].head_or_tp_rank,
            self.pp_rank,
            self.pcp_rank,
            self.dcp_rank,
        )

    def _make_layerwise_partial_key(
        self,
        request: ReqMeta,
        group_id: int,
        block_index: int,
        end_token: int,
    ) -> str:
        """Use the backend-bound layout shared with full-key construction."""
        assert self.layerwise_keys is not None
        return self.layerwise_keys.make_partial_key(
            self.model_name,
            request.req_id,
            group_id,
            block_index,
            end_token,
            self.metadata[group_id].head_or_tp_rank,
            self.pp_rank,
        )

    def _get_layerwise_load_snapshot(
        self,
        request: ReqMeta,
        group_id: int,
        num_computed_tokens: int,
        block_hashes: Sequence[BlockHash],
    ) -> tuple[int, Callable[[int], tuple[str, bool] | None]] | None:
        if group_id not in self._reused_layers_by_group:
            # Resident pages remain available for the running request.
            return None

        full_blocks = num_computed_tokens // self.grouped_block_size[group_id]
        snapshot = self._snapshot_keys.get(request.req_id, {}).get(group_id, {})

        def get_saved_key(block_idx: int) -> tuple[str, bool] | None:
            """Return the pool key and whether it is public."""
            if block_idx < full_blocks and block_idx < len(block_hashes):
                block_hash = block_hash_to_str(block_hashes[block_idx])
                return self._make_layerwise_full_key(group_id, block_hash), True
            private_key = snapshot.get(block_idx)
            if private_key is not None:
                return private_key, False
            logger.error("No saved KV page for request %s group %d block %d", request.req_id, group_id, block_idx)
            return None

        return full_blocks, get_saved_key

    def _layerwise_key_batches(self, keys: list[str]) -> list[list[str]]:
        batch_size = self.layerwise_max_transfer_blocks if self.layerwise_max_transfer_blocks > 0 else max(1, len(keys))
        return [keys[start : start + batch_size] for start in range(0, len(keys), batch_size)]

    def _probe_pool_missing_keys(self, keys: list[str], exists, error_context: str) -> list[str]:
        """Batched existence probe returning the keys not yet in the pool.

        Shared by both layerwise data planes. Failing open (treat as missing)
        is intentional: an unreadable pool costs hits or transfer bandwidth,
        while a bogus skip would cost the blocks themselves. Keys are returned
        in input order, duplicates preserved; callers deduplicate when the
        probe pattern, not the key list, must stay minimal.
        """
        missing: list[str] = []
        for key_batch in self._layerwise_key_batches(keys):
            try:
                states = exists(key_batch)
                if len(states) != len(key_batch):
                    raise RuntimeError(
                        f"{error_context} existence check returned unexpected number of states: "
                        f"expected={len(key_batch)}, actual={len(states)}"
                    )
                missing.extend(key for key, state in zip(key_batch, states, strict=True) if int(state) != 1)
            except Exception as exc:
                logger.error(
                    "%s existence check failed keys=%d error=%s; treating all as missing",
                    error_context,
                    len(key_batch),
                    exc,
                )
                missing.extend(key_batch)
        return missing

    def _filter_pool_existing_keys(self, keys: list[str]) -> list[str]:
        """Drop keys whose block is already readable in the pool.

        A block-key backend may reject overwriting a committed object: the put
        session start fails for that key and a following range write can make
        the readable object unusable. The layerwise save path must therefore
        never open a session over a key that is already complete.

        This mirrors ``KVCacheStoreSendingThread._get_missing_indices``, which
        is what keeps the non-layerwise path immune to both repeat requests
        sharing a prefix and concurrent prefill ranks writing the same keys.

        Returns the subset of ``keys`` that is safe to start a session for.
        Failing open (treat as missing) is intentional: an unreadable pool costs
        hits, while a bogus skip would cost the blocks themselves.
        """
        if not keys:
            return []
        if not getattr(self.m_store, "requires_exists_before_put", True):
            return keys
        exists = getattr(self.m_store, "exists", None)
        if not callable(exists):
            exists = getattr(self.m_store, "batch_is_exist", None)
        if not callable(exists):
            return keys
        return self._probe_pool_missing_keys(keys, exists, "Block-key pool")

    def _start_layerwise_put_keys(self, keys: list[str], object_size: int) -> list[int]:
        results: list[int] = []
        for key_batch in self._layerwise_key_batches(keys):
            results.extend(
                require_aligned_batch_results(
                    "batch_put_start",
                    key_batch,
                    self.m_store.batch_put_start(key_batch, [object_size] * len(key_batch)),
                )
            )
        return results

    def _start_layerwise_get_keys(self, keys: list[str]) -> list[int]:
        results: list[int] = []
        for key_batch in self._layerwise_key_batches(keys):
            results.extend(
                require_aligned_batch_results(
                    "batch_get_start",
                    key_batch,
                    self.m_store.batch_get_start(key_batch),
                )
            )
        return results

    def _layerwise_object_size_bytes(self) -> int:
        group_block_len = getattr(self, "group_block_len", {}).get(0, [])
        object_size = sum(group_block_len) if group_block_len else self.page_size_bytes
        if object_size <= 0:
            raise RuntimeError(f"Layerwise object size must be positive, got {object_size}")
        return object_size

    def _queue_layerwise_revoke_keys(self, keys: list[str]) -> None:
        if not keys:
            return
        try:
            assert isinstance(self.kv_send_thread, KVCacheStoreLayerSendingThread)
            self.kv_send_thread.add_revoke_request(keys)
        except Exception as exc:
            logger.error("Failed to queue layerwise revoke keys=%s error=%s", keys, exc)
            with self._put_started_keys_lock:
                self._put_started_keys.difference_update(keys)

    def _end_layerwise_load_keys(self, keys: list[str]) -> None:
        for key_batch in self._layerwise_key_batches(list(dict.fromkeys(keys))):
            try:
                result = self.m_store.batch_get_end(key_batch)
                if result != 0:
                    logger.error("Layerwise batch_get_end failed keys=%s result=%s", key_batch, result)
            except Exception as exc:
                logger.error("Layerwise batch_get_end raised keys=%s error=%s", key_batch, exc)

    def _release_layerwise_requests_for_retry(self, req_ids: set[str]) -> None:
        with self._load_session_lock:
            self._end_layerwise_load_keys(self._layerwise_session_tracker.release_for_retry(req_ids))

    def _release_layerwise_requests_terminal(self, req_ids: set[str]) -> None:
        with self._load_session_lock:
            self._end_layerwise_load_keys(self._layerwise_session_tracker.release_terminal(req_ids))

    def _release_failed_layerwise_get_attempts(self, request_ids_by_key: dict[str, set[str]]) -> None:
        with self._load_session_lock:
            keys = self._layerwise_session_tracker.release_failed_get_attempts(request_ids_by_key)
            self._end_layerwise_load_keys(keys)

    def _open_layerwise_get_sessions(self, request_key_slots: list[tuple[ReqMeta, str, int, int | None]]) -> None:
        if not request_key_slots:
            return
        keys = list(dict.fromkeys(key for _, key, _, _ in request_key_slots))
        request_ids_by_key: dict[str, set[str]] = {}
        for request, key, _, _ in request_key_slots:
            request_ids_by_key.setdefault(key, set()).add(request.req_id)
        try:
            results = self._start_layerwise_get_keys(keys)
        except Exception as exc:
            logger.error("Layerwise batch_get_start failed keys=%s error=%s", keys, exc)
            self._release_failed_layerwise_get_attempts(request_ids_by_key)
            self._record_invalid_blocks_ids([block_id for _, _, block_id, _ in request_key_slots])
            for request, _, _, slot in request_key_slots:
                if slot is None:
                    request.load_last_block_key = None
                else:
                    request.load_block_keys[slot] = None
            return

        results_by_key = dict(zip(keys, results, strict=True))
        for key, result in results_by_key.items():
            self._layerwise_session_tracker.record_get_result(key, request_ids_by_key[key], succeeded=result == 0)
        request_load_keys: dict[int, list[str]] = {}
        request_seen_keys: dict[int, set[str]] = {}
        requests_by_id: dict[int, ReqMeta] = {}
        for request, key, block_id, slot in request_key_slots:
            request_identity = id(request)
            requests_by_id[request_identity] = request
            request_load_keys.setdefault(request_identity, [])
            request_seen_keys.setdefault(request_identity, set())
            if results_by_key[key] == 0:
                if key not in request_seen_keys[request_identity]:
                    request_load_keys[request_identity].append(key)
                    request_seen_keys[request_identity].add(key)
                continue
            self._record_invalid_blocks_ids([block_id])
            if slot is None:
                request.load_last_block_key = None
            else:
                request.load_block_keys[slot] = None
        for request_identity, request in requests_by_id.items():
            request.load_keys = request_load_keys[request_identity]

    def _process_save_for_layer_batch(
        self,
        requests: list[ReqMeta],
        layer_id: int,
        group_id: int = 0,
        layer_idx_in_group: int = 0,
    ) -> None:
        # MLA replicas share a writer; recurrent state is saved by every TP rank.
        if not self._is_layerwise_save_owner(group_id):
            return
        block_size = get_group_block_size(self.grouped_block_size, group_id)
        request_block_ranges = []
        for request in requests:
            if request.can_save is None or not request.can_save:
                continue
            save_start_block = request.save_start_token // block_size
            group_block_hashes = get_block_hashes(
                request.block_hashes,
                block_size,
                self.hash_block_size,
            )
            partial_block_index = get_partial_block_index(
                request.target_token_len,
                block_size,
                len(group_block_hashes),
                self.layerwise_offload,
            )
            save_end_block = self._layerwise_save_full_end(request, block_size, partial_block_index)
            # External hits already have full keys. A previous-step hit may
            # instead use snapshots, which still need promotion to full keys.
            if request.load_spec is not None and not request.load_spec.layerwise_previous_step:
                save_start_block = request.load_spec.clamp_save_start(save_start_block, block_size)
            if partial_block_index is None:
                partial_block_index = request.partial_block_index
            num_allocated_blocks = (
                len(request.block_ids_by_group_np[group_id])
                if request.block_ids_by_group_np is not None and group_id < len(request.block_ids_by_group_np)
                else len(request.block_ids_by_group[group_id])
            )
            save_end_block = min(save_end_block, num_allocated_blocks)
            save_start_block = min(save_start_block, save_end_block)
            if partial_block_index is not None and partial_block_index >= num_allocated_blocks:
                partial_block_index = None
            group_store_mask = (
                request.store_masks[group_id]
                if request.store_masks is not None and group_id < len(request.store_masks)
                else None
            )
            block_runs = self._masked_block_runs_cached(
                "save", request, group_id, save_start_block, save_end_block, group_store_mask
            )
            if not block_runs and partial_block_index is None:
                continue
            for run_idx, (run_start_block, run_end_block) in enumerate(block_runs):
                # The trailing partial block rides on the last run (or on its
                # own zero-length range) so it is transferred exactly once.
                is_last_run = run_idx == len(block_runs) - 1
                request_block_ranges.append(
                    LayerBlockRange(
                        request=request,
                        start_block=run_start_block,
                        end_block=run_end_block,
                        partial_block_index=partial_block_index if is_last_run else None,
                    )
                )
            if not block_runs and partial_block_index is not None:
                request_block_ranges.append(
                    LayerBlockRange(
                        request=request,
                        start_block=partial_block_index,
                        end_block=partial_block_index,
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
                    use_key_major_ranges=self.use_block_key_layerwise,
                    final_group_layer=layer_idx_in_group
                    == getattr(self, "group_num_layers", {}).get(group_id, self.num_layers) - 1,
                )
            )

    def _layerwise_save_full_end(self, request: ReqMeta, block_size: int, partial_block_index: int | None) -> int:
        """Return the exclusive end of the group's full-block save range."""
        end = request.save_end_token // block_size
        if self.use_layerwise_transfer and self.layerwise_offload:
            end = request.target_token_len // block_size
            if partial_block_index is not None:
                end = min(end, partial_block_index)
        return end

    def _pool_keys_missing(self, keys: list[str]) -> set[str]:
        """Return the subset of ``keys`` not yet committed in the pool.

        The check is medium-agnostic on purpose: a block committed on any
        medium (DRAM or SSD) does not need re-saving, and the load path
        rewarms SSD blocks on read. Fails open (treats check errors as
        missing) so an unreadable pool costs transfer bandwidth, never the
        blocks themselves.
        """
        if not keys:
            return set()
        unique_keys = list(dict.fromkeys(keys))
        return set(self._probe_pool_missing_keys(unique_keys, self.m_store.batch_is_exist, "Pool"))

    def _build_shared_save_data(self) -> None:
        """Build shared block data once and attach to all layer save tasks.

        For the layerwise send thread (KVCacheStoreLayerSendingThread):
        pre-computes SharedBlockData via LayerBatchBuilder.build_shared().

        In multi-group mode, shared data is built per-group because each
        group has different block_ranges (different effective_block_size).
        """
        for tasks in self.layer_save_tasks:
            for task in tasks:
                task.block_ranges = [br for br in task.block_ranges if br.request.can_save is not False]
        if isinstance(self.kv_send_thread, KVCacheStoreLayerSendingThread):
            all_save_keys: list[str] = []
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
                    all_save_keys.extend(shared.save_keys)
                    for layer_id in range(self.num_layers):
                        for task in self.layer_save_tasks[layer_id]:
                            if task.group_id == group_id:
                                task.shared_block_data = shared
            last_task = None
            for tasks in self.layer_save_tasks:
                for task in tasks:
                    task.write_finish_keys.clear()
                    if task.shared_block_data is not None:
                        last_task = task
            if last_task is not None:
                last_task.write_finish_keys.extend(dict.fromkeys(all_save_keys))

    def _compute_reachable_store_masks(
        self,
        request: ReqMeta,
    ) -> tuple[Sequence[bool] | None, ...] | None:
        """Per-group reachable-store masks for the layerwise save path.

        Mirrors the non-layerwise store path (KVCacheStoreSendingThread):
        for reachability-limited groups (sliding window / compressor state)
        only the blocks the KV cache managers consider reachable are
        persisted, instead of the full sequence. Falls back to None (store
        everything) when the coordinator is unavailable or the save extent
        is not aligned.
        """
        if getattr(self, "block_key_hybrid", False) and request.can_save and request.save_end_token > 0:
            if self.cache_coordinator is None or request.save_end_token % self.cache_transfer_granularity:
                raise ValueError("Block-key hybrid saves require a coordinator-aligned full-block boundary")
            return self.token_database.store_mask(request.save_end_token, request.num_prompt_tokens)
        masks = None
        if (
            self.cache_coordinator is not None
            and request.save_end_token > 0
            and request.save_end_token % self.cache_transfer_granularity == 0
        ):
            try:
                masks = self.token_database.store_mask(request.save_end_token, request.num_prompt_tokens)
            except AssertionError:
                masks = None
        align_groups = getattr(self, "group_uses_align_state", [])
        if not any(align_groups):
            return masks
        # Mamba align slots can be null even when their token range is
        # reachable. They do not contain a state snapshot. Do not allocate
        # readable pool keys or copy block zero for these holes; the
        # non-layerwise path applies the same skip_null_blocks rule.
        result: list[list[bool] | None] = list(masks) if masks is not None else [None] * self.num_kv_cache_groups
        for group_id, uses_align in enumerate(align_groups):
            if not uses_align:
                continue
            block_ids = request.block_ids_by_group[group_id]
            original_mask = result[group_id]
            num_blocks = request.save_end_token // self.grouped_block_size[group_id]
            if self.layerwise_offload:
                num_blocks = self._layerwise_save_full_end(request, self.grouped_block_size[group_id], None)
            # An aligned prefill chunk ends with a real recurrent checkpoint,
            # even when it is not the final prompt replay boundary. Keep that
            # checkpoint before the next chunk moves/reuses its state slots.
            chunk_checkpoint = (
                request.num_prompt_tokens is not None
                and request.target_token_len == num_blocks * self.grouped_block_size[group_id]
                and (self.layerwise_offload or request.save_end_token % self.cache_transfer_granularity == 0)
                and 0 < request.target_token_len < request.num_prompt_tokens
            )
            result[group_id] = [
                index < len(block_ids)
                and block_ids[index] != 0
                and (
                    original_mask is None
                    or (index < len(original_mask) and original_mask[index])
                    or (chunk_checkpoint and index == num_blocks - 1)
                )
                for index in range(num_blocks)
            ]
        return tuple(result)

    def _process_load_for_layer_batch(
        self,
        requests: list[ReqMeta],
        layer_id: int,
        group_id: int = 0,
        layer_idx_in_group: int = 0,
    ) -> None:
        block_size = get_group_block_size(self.grouped_block_size, group_id)
        reuses_buffer = self.layerwise_offload and layer_id in self._reused_layers_by_group.get(group_id, ())
        request_block_ranges = []
        for request in requests:
            if self.use_block_key_layerwise:
                # Block-key sessions retain committed block ownership across
                # chunked-prefill steps.  A later chunk can therefore have no
                # new load_spec while still needing to restore the previously
                # committed prefix into its newly allocated local blocks.
                populated_slots = [index for index, key in enumerate(request.load_block_keys) if key is not None]
                if not populated_slots:
                    continue
                request_block_ranges.append(
                    LayerBlockRange(
                        request=request,
                        start_block=min(populated_slots),
                        end_block=max(populated_slots) + 1,
                    )
                )
                continue
            if request.load_spec is None or not request.load_spec.can_load:
                continue
            cached_tokens = request.load_spec.kvpool_cached_tokens
            if not getattr(self, "use_eagle", False):
                cached_tokens = request.load_spec.pool_hit_tokens
            if (
                getattr(self, "use_eagle", False)
                and request.load_spec.kvpool_cached_tokens == request.target_token_len - 1
            ):
                # Full-hit path: the trailing block is recomputed and will be
                # re-stored by the normal save path, so never skip it here.
                logger.debug(
                    "Reqid: %s full-hit tail recompute path, tail block will be re-stored",
                    request.req_id,
                )
            group_block_hashes = get_block_hashes(
                request.block_hashes,
                block_size,
                self.hash_block_size,
            )
            load_start_block = request.load_spec.vllm_cached_tokens // block_size if not reuses_buffer else 0
            cached_full_blocks = cached_tokens // block_size
            full_blocks = min(cached_full_blocks, len(group_block_hashes))
            partial_block_index = None
            if self.layerwise_offload and request.load_spec.layerwise_previous_step:
                if not reuses_buffer:
                    continue
                full_blocks = cached_full_blocks
                partial_block_index = self._resolve_layerwise_load_partial(cached_tokens, block_size)
            partial_gva = (
                request.partial_load_gva_per_group[group_id]
                if group_id < len(request.partial_load_gva_per_group)
                else request.last_block_gva
            )
            if partial_gva is None or partial_gva <= 0:
                partial_block_index = None
            if partial_block_index is not None and partial_block_index < load_start_block:
                partial_block_index = None
            group_load_mask = (
                request.load_masks[group_id]
                if request.load_masks is not None and group_id < len(request.load_masks)
                else None
            )
            block_runs = self._masked_block_runs_cached(
                "load", request, group_id, load_start_block, full_blocks, group_load_mask
            )
            # A preserved partial snapshot may now sit on a hash boundary
            # after speculative rollback.  It must be transferred exactly
            # once through the partial GVA, not also as a full-key block.
            if partial_block_index is not None and partial_block_index < full_blocks:
                split_runs = []
                for run_start, run_end in block_runs:
                    if partial_block_index < run_start or partial_block_index >= run_end:
                        split_runs.append((run_start, run_end))
                        continue
                    if run_start < partial_block_index:
                        split_runs.append((run_start, partial_block_index))
                    if partial_block_index + 1 < run_end:
                        split_runs.append((partial_block_index + 1, run_end))
                block_runs = split_runs
            if not block_runs and partial_block_index is None:
                continue
            for run_idx, (run_start_block, run_end_block) in enumerate(block_runs):
                # The trailing partial block rides on the last run (or on its
                # own zero-length range) so it is transferred exactly once.
                is_last_run = run_idx == len(block_runs) - 1
                request_block_ranges.append(
                    LayerBlockRange(
                        request=request,
                        start_block=run_start_block,
                        end_block=run_end_block,
                        partial_block_index=partial_block_index if is_last_run else None,
                    )
                )
            if not block_runs and partial_block_index is not None:
                request_block_ranges.append(
                    LayerBlockRange(
                        request=request,
                        start_block=partial_block_index,
                        end_block=partial_block_index,
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
                    use_key_major_ranges=self.use_block_key_layerwise,
                    final_group_layer=layer_idx_in_group
                    == getattr(self, "group_num_layers", {}).get(group_id, self.num_layers) - 1,
                )
            )

    @staticmethod
    def _resolve_layerwise_load_partial(cached_tokens: int, block_size: int) -> int | None:
        """Return the incomplete block at the end of the previous step."""
        if cached_tokens % block_size:
            return cached_tokens // block_size
        return None

    def _check_hybrid_load_errors(self) -> None:
        # Memcache reports invalid blocks through the existing scheduler policy.
        if self.use_layerwise_transfer and not self.layerwise_offload:
            return
        if self.layerwise_offload:
            # Completion events also fire on failure. Reused buffers no longer
            # have a resident copy to fall back to after a failed PUT or GET.
            if self._save_disabled.is_set() or (self.kv_recv_thread and self.kv_recv_thread.load_disabled.is_set()):
                self._layer_load_aborted.set()
                raise RuntimeError("Layerwise transfer failed; refusing incomplete KV/state")
        # A hybrid block ID can belong to SWA, compressed KV or compressor
        # state. Do not feed partially restored state to the attention kernel.
        with self._invalid_block_ids_lock:
            if self._invalid_block_ids or (self.layerwise_offload and self._failed_requests):
                self._layer_load_aborted.set()
                raise RuntimeError("Layerwise transfer failed; refusing incomplete KV/state")

    def _build_shared_load_data(self) -> None:
        """Share load metadata, preserving layer-specific selections for reuse."""
        if not isinstance(self.kv_recv_thread, KVCacheStoreLayerRecvingThread):
            return
        if not self.layerwise_offload:
            # Ordinary layerwise storage has the same load range per group.
            # Preserve its single-build path without selection snapshots.
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
            return
        for group_id in range(self.num_kv_cache_groups):
            shared_by_ranges: dict[
                tuple[bool, tuple[tuple[int, int, int, int | None], ...]], SharedBlockData | None
            ] = {}
            for layer_id in range(self.num_layers):
                for task in self.layer_load_tasks[layer_id]:
                    if task.group_id != group_id:
                        continue
                    # Independent layers retain local prefix KV, while reused
                    # layers must restore it, including any partial block.
                    # Request identity also matters: different requests can
                    # select the same logical range but different blocks/GVAs.
                    ranges = tuple(
                        (id(r.request), r.start_block, r.end_block, r.partial_block_index) for r in task.block_ranges
                    )
                    key = (task.use_key_major_ranges, ranges)
                    if key not in shared_by_ranges:
                        shared = self.kv_recv_thread.build_shared_data(task)
                        if shared is not None:
                            # The builder reuses scratch arrays on the next
                            # call; retain each selection's own snapshot.
                            shared.block_ids_arr = shared.block_ids_arr.copy()
                            if shared.block_gvas_arr is not None:
                                shared.block_gvas_arr = shared.block_gvas_arr.copy()
                        shared_by_ranges[key] = shared
                    task.shared_block_data = shared_by_ranges[key]

    def _compute_reachable_load_masks(
        self,
        request: ReqMeta,
        cached_tokens: int,
    ) -> tuple[Sequence[bool] | None, ...] | None:
        """Per-group load masks for the layerwise load path.

        Mirrors the non-layerwise load path: only the blocks the KV cache
        managers need for a hit of ``cached_tokens`` are fetched, which is a
        subset of the blocks persisted by the reachable store masks.
        """
        if cached_tokens <= 0:
            return None
        if getattr(self, "block_key_hybrid", False):
            if self.cache_coordinator is None:
                raise ValueError("Block-key hybrid loads require a reachability coordinator")
            return self.token_database.load_mask(request.block_hashes, cached_tokens)
        if self.cache_coordinator is None:
            masks = None
        else:
            try:
                masks = self.token_database.load_mask(request.block_hashes, cached_tokens)
            except AssertionError:
                masks = None
        if not self.layerwise_offload or request.load_spec is None or not request.load_spec.layerwise_previous_step:
            return masks
        reused_state_groups = [
            group_id
            for group_id, uses_align in enumerate(self.group_uses_align_state)
            if uses_align and group_id in self._reused_layers_by_group
        ]
        if not reused_state_groups:
            return masks
        # Preserve other groups' reachability masks. An aligned recurrent page
        # represents one endpoint; unaligned state uses the partial GVA path.
        result: list[Sequence[bool] | None] = list(masks) if masks is not None else [None] * self.num_kv_cache_groups
        for group_id in reused_state_groups:
            block_ids = request.block_ids_by_group[group_id]
            mask = [False] * len(block_ids)
            full_blocks, tail_tokens = divmod(cached_tokens, self.grouped_block_size[group_id])
            last_full_block = full_blocks - 1
            if tail_tokens == 0 and 0 <= last_full_block < len(block_ids) and block_ids[last_full_block] != 0:
                mask[last_full_block] = True
            result[group_id] = mask
        return tuple(result)

    def wait_for_previous_save(self) -> None:
        self._drain_deferred_last_save()

    def _drain_deferred_last_save(self) -> None:
        """PERF-TUNE(2): deferred tail of the previous step's last-layer save."""
        if not getattr(self, "_pending_last_save_drain", False):
            return
        self._pending_last_save_drain = False
        assert self.kv_send_thread is not None
        # PP-aware: only the per-stage LOCAL layers are forwarded; mirror
        # save_kv_layer's num_local computation for the drain lifecycle.
        if getattr(self, "pp_size", 1) > 1:
            num_local = getattr(self, "layerwise_key_layers", 0) or self.num_layers
        else:
            num_local = self.num_layers
        self._wait_for_final_layer_save(num_local)

    def process_layer_data(self, requests: list[ReqMeta]) -> None:
        self._masked_runs_cache.clear()
        # Keep this method safe for direct callers as well as metadata binding.
        # Worker threads may still own the lists from the preceding step.
        self.layer_save_tasks = [[] for _ in range(self.num_layers)]
        self.layer_load_tasks = [[] for _ in range(self.num_layers)]
        if not requests:
            return
        # Mooncake uses the projected stage-local cache layout, including any
        # draft layers. The GVA plane retains its existing PP key count.
        if not self.use_block_key_layerwise and getattr(self, "pp_size", 1) > 1:
            num_local = getattr(self, "layerwise_key_layers", 0) or self.num_layers
        else:
            num_local = self.num_layers
        for request in requests:
            request.store_masks = self._compute_reachable_store_masks(request)
        group_requests = self._data_plane.prepare_sessions(requests)
        for local_layer in range(num_local):
            for group_id, layer_idx_in_group in self._groups_for_layerwise_transfer(local_layer):
                self._process_save_for_layer_batch(
                    group_requests.get(group_id, requests), local_layer, group_id, layer_idx_in_group
                )
        self._build_shared_save_data()
        for local_layer in range(num_local):
            for group_id, layer_idx_in_group in self._groups_for_layerwise_transfer(local_layer):
                self._process_load_for_layer_batch(
                    group_requests.get(group_id, requests), local_layer, group_id, layer_idx_in_group
                )
        self._build_shared_load_data()
        if self.layerwise_offload:
            self._check_hybrid_load_errors()

    def _submit_attention_save(self, layer_id: int) -> None:
        """Start current-layer put only once KV is ready at attention entry."""
        if not is_kv_save_role(self.kv_role, self.save_decode_cache):
            return
        assert self.sync_save_events is not None
        assert self.layer_save_finished_events is not None
        assert self.kv_send_thread is not None
        sync_save_events = self.sync_save_events
        save_finished_events = self.layer_save_finished_events
        send_thread = cast(KVCacheStoreLayerSendingThreadBase, self.kv_send_thread)
        self._attention_saved_layers.add(layer_id)
        sync_save_events[layer_id].record()
        tasks = self.layer_save_tasks[layer_id]
        if not tasks:
            save_finished_events[layer_id].set()
            return
        for task in tasks:
            for block_range in task.block_ranges:
                send_thread.add_stored_request(block_range.request.req_id)
        send_thread.add_request(tasks)

    def _finish_attention_window(self) -> None:
        try:
            self._drain_attention_transfers()
        except Exception:
            self._layer_load_aborted.set()
            self._data_plane.finish_sessions()
            raise

    def _drain_attention_transfers(self, drain_recv: bool | None = None, full: bool = False) -> None:
        """Apply the configured transfer policy at an attention boundary.

        The LOAD queue is deliberately left running. Every load this rank needs
        is already guaranteed complete by ``wait_for_layer_load`` before the
        layer consumes it, and the rest of that queue is exactly the prefetch
        issued for layers L+1..L+P to overlap with the compute currently in
        flight. Draining it here therefore waited, on every single layer, for
        transfers that were scheduled early on purpose, which is what pinned the
        fence at ~900 ms per layer and made layerwise slower than the
        non-layerwise bulk path. This matches the MemCache layerwise path, which
        gates the *start* of transfers to the attention boundary and never
        quiesces the load queue.

        The SAVE queue is bounded rather than drained. A committed block-key
        object is unreadable until the put session ends, so a put that is
        still open while the next collective launches cannot expose a partial
        block to any reader; the cost of waiting for it here is purely that the
        transfer stops overlapping compute. Letting the send thread run a bounded
        number of layers behind the compute stream keeps the ranged puts off the
        critical path, and ordering is preserved because that thread still
        consumes the queue in submission order. The step fence drains all saves,
        including earlier PUTs when the final layer has no save task.

        ``full`` drains the selected queues to zero, defaulting to both queues
        for teardown. Step completion drains writers first, then reuse readers
        before resetting save completion events.
        """
        assert self.layerwise_backend is not None
        if full and drain_recv is None:
            drain_recv = True
        elif drain_recv is None:
            drain_recv = self.layerwise_backend.fence_drains_recv()
        channels = [("send", self.kv_send_thread)]
        if drain_recv:
            channels.insert(0, ("recv", self.kv_recv_thread))
        for channel, thread in channels:
            if thread is None:
                continue
            backlog_limit = 0 if (full or channel != "send") else self.layerwise_backend.send_fence_backlog()
            queue = thread.request_queue
            with queue.all_tasks_done:
                while queue.unfinished_tasks > backlog_limit:
                    thread.raise_if_failed()
                    queue.all_tasks_done.wait(timeout=0.1)
            thread.raise_if_failed()
        if drain_recv and self.kv_recv_thread is not None:
            # Errors are still surfaced per layer by wait_for_layer_load.
            self._check_hybrid_load_errors()

    def _submit_ready_layer_loads(self) -> None:
        assert self.kv_recv_thread is not None
        recv_thread = self.kv_recv_thread

        def submit_layer_load(layer_id: int) -> bool:
            reuse_sources = self.prefetch_layer_map.get(layer_id, ())
            if not self.layer_load_tasks[layer_id] and not reuse_sources:
                return False
            attention_start_gate = None
            if (
                self.layer_load_tasks[layer_id]
                and layer_id != self.current_layer
                and not getattr(self, "has_shared_kv", False)
            ):
                attention_start_gate = get_attention_compute_start_gate()
            recv_thread.add_request(
                LayerLoadTask(  # type: ignore[arg-type]
                    wait_for_save_layers=reuse_sources,
                    transfer_tasks=self.layer_load_tasks[layer_id],
                    layer_id=layer_id,
                    attention_start_gate=attention_start_gate,
                )
            )
            return True

        if getattr(self, "has_shared_kv", False):
            assert self.hf_config is not None
            source_layers = self.hf_config.kv_source_layer_ids
            window_end = self.current_layer
            if self.current_layer >= source_layers[-1]:
                # Keep tail SWA and draft loads one layer ahead.
                window_end += 1
            elif self.current_layer == 0 or self.current_layer in source_layers:
                # Submit the intervening SWA before the next shared source.
                window_end = next(layer for layer in source_layers if layer > self.current_layer)
        else:
            window_end = self.current_layer + self.num_prefetch_layers
        window_end = min(window_end, self.num_layers - 1)
        draft_start = getattr(self, "_dflash_draft_start_layer", None)
        if draft_start is not None:
            # Draft context load runs after target attention has stopped creating gates.
            if self.current_layer < draft_start:
                window_end = min(window_end, draft_start - 1)
            else:
                window_end = self.current_layer
        while self.next_layer_to_submit <= window_end:
            layer_id = self.next_layer_to_submit
            self.next_layer_to_submit += 1
            submit_layer_load(layer_id)

    def wait_for_layer_load_ready(self, layer_name: str) -> None:
        physical_layer = self._extract_physical_layer_index(layer_name)
        layer_id = self._global_to_local_layer.get(physical_layer)
        if layer_id is None:
            if self._global_to_local_layer:
                return
            layer_id = physical_layer
        if not 0 <= layer_id < len(self.layer_load_tasks):
            return
        if self.kv_recv_thread is None:
            raise RuntimeError("kv_recv_thread is not initialized")
        self.kv_recv_thread.raise_if_failed()
        if not self.layer_load_tasks[layer_id] and layer_id not in self.prefetch_layer_map:
            return
        if self.layer_load_finished_events is None:
            raise RuntimeError("layer_load_finished_events is not initialized")
        while not self.layer_load_finished_events[layer_id].wait(timeout=1):
            self.kv_recv_thread.raise_if_failed()
            if self._layer_load_aborted.is_set():
                raise RuntimeError("Layerwise load aborted before KVPP broadcast")
        self.kv_recv_thread.raise_if_failed()

    def load_dflash_draft_kv(self) -> None:
        # Finish draft reads before post_forward releases the step's pool leases.
        first_draft_layer = self._dflash_draft_start_layer
        if first_draft_layer is None:
            return
        # no_forward also calls finish_forward, without running any layer hooks.
        if self.current_layer == 0 and self.next_layer_to_submit == 0:
            return
        if self.current_layer != first_draft_layer:
            raise RuntimeError("Target KV hooks did not reach the DFlash draft layer boundary")
        for layer_id in range(first_draft_layer, self.num_layers):
            self.current_layer = layer_id
            self.wait_for_layer_load()
        self.current_layer = first_draft_layer
        self._dflash_save_pending = self.can_put

    def save_dflash_draft_kv(self, metadata: AscendConnectorMetadata) -> None:
        if not self._dflash_save_pending:
            return
        # This runs after the speculator has written its context KV.
        self._dflash_save_pending = False
        while self.current_layer < self.num_layers:
            self.save_kv_layer(metadata)
        self._drain_deferred_last_save()

    def wait_for_layer_load(self) -> None:
        if self.current_layer >= self.num_layers:
            return
        assert self.layer_load_finished_events is not None
        assert self.kv_recv_thread is not None
        gate = reset_attention_compute_start_gate()
        try:
            self.kv_recv_thread.raise_if_failed()
            if getattr(self, "block_key_hybrid", False):
                layer_id = self.current_layer
                # Conv/recurrent state is updated inside attention, after this
                # entry event. Preserve its post-compute save hook.
                if layer_id not in self._recurrent_layers:
                    gate.on_start = lambda: self._submit_attention_save(layer_id)
                gate.on_finish = self._finish_attention_window
            if getattr(self, "block_key_hybrid", False) and self.next_layer_to_submit <= self.current_layer:
                # An unprefetched demand load must not race earlier collectives.
                boundary = torch.npu.Event()
                boundary.record()
                boundary.synchronize()
            self._submit_ready_layer_loads()
            should_wait = (
                bool(self.layer_load_tasks[self.current_layer]) or self.current_layer in self.prefetch_layer_map
            )
            if should_wait:
                while not self.layer_load_finished_events[self.current_layer].wait(timeout=10):
                    self.kv_recv_thread.raise_if_failed()
                    logger.info("Layerwise %d load not done, keep waiting", self.current_layer)
                self.kv_recv_thread.raise_if_failed()
            elif self.external_slot_release_waiter is not None:
                self.external_slot_release_waiter(self.current_layer)
            if getattr(self, "layerwise_offload", False) or getattr(self, "block_key_hybrid", False):
                self._check_hybrid_load_errors()
        except Exception:
            if hasattr(self, "_layer_load_aborted"):
                self._layer_load_aborted.set()
            if getattr(self, "block_key_hybrid", False):
                gate.cancel()
                # Range calls must finish before releasing their get sessions.
                with suppress(Exception):
                    self._drain_attention_transfers(full=True)
            if getattr(self, "use_block_key_layerwise", False):
                self._data_plane.finish_sessions()
            raise
        # KVPP can observe completion after attention; reset at the next bind.
        if not self.use_kvpp:
            self.layer_load_finished_events[self.current_layer].clear()
        if (
            getattr(self, "block_key_hybrid", False) or (self.use_kvpp and self.use_block_key_layerwise)
        ) and self.current_layer == self.num_layers - 1:
            # The final model layer can have no reachable load rows. Completion
            # belongs to the whole request, not whichever group happens to end here.
            for req_id in self._current_layerwise_last_chunk_req_ids:
                self.kv_recv_thread.set_finished_request(req_id)
        if getattr(self, "use_block_key_layerwise", False) and (
            self._layer_load_aborted.is_set() or self.current_layer == self.num_layers - 1
        ):
            self._data_plane.finish_sessions()
        if hasattr(self, "kv_role") and not is_kv_save_role(self.kv_role, getattr(self, "save_decode_cache", False)):
            self.current_layer += 1

    def save_kv_layer(self, connector_metadata: AscendConnectorMetadata) -> None:
        # Match process_layer_data(), including stage-local draft layers.
        if not self.use_block_key_layerwise and getattr(self, "pp_size", 1) > 1:
            num_local = getattr(self, "layerwise_key_layers", 0) or self.num_layers
        else:
            num_local = self.num_layers
        if self.current_layer >= num_local:
            return
        assert self.sync_save_events is not None
        assert self.layer_save_finished_events is not None
        assert self.kv_send_thread is not None
        send_thread = cast(KVCacheStoreLayerSendingThreadBase, self.kv_send_thread)
        send_thread.raise_if_failed()
        if self.current_layer not in self._attention_saved_layers:
            self.sync_save_events[self.current_layer].record()
            if self.layer_save_tasks[self.current_layer]:
                for task in self.layer_save_tasks[self.current_layer]:
                    for block_range in task.block_ranges:
                        send_thread.add_stored_request(block_range.request.req_id)
                send_thread.add_request(self.layer_save_tasks[self.current_layer])  # type: ignore[arg-type]
            else:
                self.layer_save_finished_events[self.current_layer].set()
        if self.block_key_hybrid:
            # Recurrent and record-only paths do not exit an attention window.
            self._finish_attention_window()
        if self.current_layer == num_local - 1:
            if _LW_DEFER_LAST_SAVE:
                # The next step drains pending saves before it can reuse cache pages.
                self._pending_last_save_drain = True
            else:
                self._wait_for_final_layer_save(num_local)
        self.current_layer += 1

    def _wait_for_final_layer_save(self, num_local: int) -> None:
        """Drain step saves before rearming completion events for the next step."""
        assert self.layer_save_finished_events is not None
        save_finished_events = self.layer_save_finished_events
        self._drain_attention_transfers(drain_recv=False, full=True)
        if getattr(self, "layerwise_offload", False):
            self._check_hybrid_load_errors()
        for event in save_finished_events[:num_local]:
            event.clear()
