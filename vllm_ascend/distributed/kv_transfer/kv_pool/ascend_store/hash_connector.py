# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in Ascend Store adapter for Core's hash-boundary handoff protocol.

Use this module as kv_connector_module_path with AscendStoreConnector. The
separate entrypoint keeps older Core installations and block_only deployments
independent of the newer upstream Store APIs, including their metrics schema.
"""

import hashlib
from contextlib import nullcontext
from copy import copy
from dataclasses import replace
from pathlib import Path

import torch
from filelock import FileLock
from vllm import envs
from vllm.config import get_current_vllm_config_or_none

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import MooncakeStoreConnector
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import ChunkedTokenDatabase
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import MooncakeStoreScheduler
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import MooncakeStoreWorker
except ImportError as exc:
    raise ImportError(
        "Ascend Store hash recovery requires vLLM Core's Mooncake Store boundary-state APIs; "
        "use a compatible Core version, not a private Core patch or block-only fallback"
    ) from exc

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorRole
from vllm.v1.kv_cache_interface import MambaSpec, group_kernel_blocks

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hash_layout import AscendHashStoreLayout


def _engine_transfer_config(vllm_config, kv_cache_config):
    # MultiConnector replaces only the child's transfer config. The scheduler
    # planner retains the engine config; worker RPCs run in Core's explicit
    # current-config context (the dynamic planner attribute may not cross IPC).
    resolved = getattr(kv_cache_config, "kv_transfer_config", None)
    if resolved is not None:
        return resolved
    current = get_current_vllm_config_or_none()
    return (current if current is not None else vllm_config).kv_transfer_config


def _align_store_eagle_policy(coord, kv_transfer_config):
    """Use the existing Ascend producer checkpoint policy in Store too.

    Core Store derives its policy from speculative config, whereas Ascend's
    scheduler/manager suppress the drop on a pure prefill producer. Otherwise
    Store publishes attention at a different hash boundary from Mamba. Only
    this connector's coordinator is changed; MTP and engine config are intact.
    """
    # Do not import the platform patch package here: that installs unrelated
    # scheduler patches even for an isolated Core Store adapter import.
    pure_producer = (
        kv_transfer_config is not None
        and getattr(kv_transfer_config, "is_kv_producer", False)
        and not getattr(kv_transfer_config, "is_kv_consumer", False)
    )
    if not pure_producer:
        return
    coord.use_eagle = False
    coord.attention_groups = [group._replace(use_eagle=False) for group in coord.attention_groups]
    coord.eagle_group_ids = set()
    coord.eagle_proof_margin_by_group = {}


class _AlignedStore:
    """Translate registration only; logical payload addresses stay untouched."""

    def __init__(self, store, views, alignment=2 * 1024 * 1024):
        self._store = store
        self._regions = {}
        self._registered = set()
        for view in views:
            if not view.numel():
                continue
            storage = view.untyped_storage()
            raw, size = storage.data_ptr(), storage.nbytes()
            aligned = ((raw + alignment - 1) // alignment) * alignment
            # Register strictly inside this allocation. Check the true strided
            # extent, not numel() or the first layer's minimum data pointer.
            span = (1 + sum((n - 1) * s for n, s in zip(view.shape, view.stride()))) * view.element_size()
            if any(s <= 0 for s in view.stride()) or not aligned <= view.data_ptr() < raw + size:
                raise ValueError("KV view starts outside the allocation's aligned registration region")
            if view.data_ptr() + span > raw + size:
                raise ValueError("KV view exceeds its storage allocation")
            self._regions[raw] = (size, aligned, raw + size - aligned)

    def register_buffer(self, address, size):
        expected, aligned, length = self._regions[address]
        if size != expected:
            raise ValueError("Unexpected Store registration span")
        if address in self._registered:
            return 0
        result = self._store.register_buffer(aligned, length)
        if result != 0:
            # Upstream's rank-local registration logs failures and continues.
            # On NPU this must fail before any payload can be published.
            raise RuntimeError(f"NPU Store registration failed: {result}")
        self._registered.add(address)
        return result

    def unregister_buffer(self, address):
        result = self._store.unregister_buffer(self._regions[address][1])
        if result == 0:
            self._registered.discard(address)
        return result

    def __getattr__(self, name):
        return getattr(self._store, name)


class AscendHashStoreWorker(MooncakeStoreWorker):
    def __init__(self, vllm_config, kv_cache_config):
        # Some FabricMem runtimes fail concurrent VA reservations. This opt-in
        # host-local bootstrap lock does not serialize requests or transfers.
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        lock_path = extra.get("npu_store_setup_lock")
        if lock_path is not None and not Path(lock_path).is_absolute():
            raise ValueError("npu_store_setup_lock must be an absolute host-local path")
        lock = FileLock(lock_path, timeout=60) if lock_path is not None else nullcontext()
        with lock:
            super().__init__(vllm_config, kv_cache_config)
        _align_store_eagle_policy(self.coord, _engine_transfer_config(vllm_config, kv_cache_config))

    def _select_store_layout(self, extra_config):
        # Never mix partial attention objects with legacy whole-page keys.
        schema = tuple(repr(group.kv_cache_spec) for group in self._kv_cache_groups)
        digest = hashlib.sha256(repr((schema, self.hash_block_size, envs.VLLM_KV_CACHE_LAYOUT)).encode()).hexdigest()[
            :16
        ]
        return None, f"@ascend_hash_v2:tp{self.tp_size}:pp{self.pp_size}:{digest}", None

    def _build_token_databases(self, metadata, plans):
        databases = []
        # Match AscendAttentionBackend.get_kv_cache_shape, not Core's logical
        # [B,H,N,C] descriptor: Ascend exposes the backend-shaped cache views.
        head_major = envs.VLLM_KV_CACHE_LAYOUT in ("HND", "LBHNC")
        for group_id, group in enumerate(self._kv_cache_groups):
            spec = group.kv_cache_spec
            is_state = isinstance(spec, MambaSpec)
            group_metadata = replace(
                metadata, group_id=group_id, tp_rank=self.tp_rank // self._group_tp_replication_factors[group_id]
            )
            layout = AscendHashStoreLayout(
                group_metadata, spec.block_size, self.hash_block_size, is_state=is_state, head_major=head_major
            )
            databases.append(
                ChunkedTokenDatabase(
                    group_metadata,
                    spec.block_size,
                    hash_block_size=self.hash_block_size,
                    store_layout=layout,
                )
            )
        return databases

    def register_kv_caches(self, kv_caches):
        if self._capacity_only:
            return
        # MRV2 carries the resolved capacity on KVCacheConfig rather than
        # filling the legacy cache-config field consumed by Store's worker.
        # Bridge the Core result on a worker-local copy; never re-plan it.
        self.cache_config = copy(self.cache_config)
        resolved_blocks = self._kv_cache_config.num_blocks
        if self.cache_config.num_gpu_blocks not in (None, resolved_blocks):
            raise ValueError("Store capacity disagrees with Core-resolved KVCacheConfig")
        self.cache_config.num_gpu_blocks = resolved_blocks
        if self._kv_cache_config.hisparse_host_num_blocks is not None:
            raise ValueError("Ascend hash Store does not support host-resident cache groups")
        names = {name for group in self._kv_cache_groups for name in group.layer_names}
        if names - kv_caches.keys():
            raise ValueError("Missing transferable Ascend Store layers")
        caches = {
            name: list(cache) if isinstance(cache, tuple) else cache
            for name, cache in kv_caches.items()
            if name in names
        }
        views = [view for cache in caches.values() for view in (cache if isinstance(cache, list) else [cache])]
        self.store = _AlignedStore(self.store, views)
        # Register every component's allocation, including separately allocated
        # recurrent state. Upstream may register only a representative tensor;
        # the wrapper deduplicates those later registration calls.
        for view in views:
            if view.numel():
                storage = view.untyped_storage()
                self.store.register_buffer(storage.data_ptr(), storage.nbytes())
        for group, database in zip(self._kv_cache_groups, self.token_dbs, strict=True):
            group_views = []
            for name in group.layer_names:
                value = caches[name]
                components = value if isinstance(value, list) else [value]
                group_views.extend(group_kernel_blocks(view, resolved_blocks) for view in components)
            database.store_layout.bind_group_views(group_views)
        super().register_kv_caches(caches)

    def wait_for_save(self, metadata):
        if self._capacity_only or not self.can_put:
            return
        pending = [request for request in metadata.requests if request.can_save]
        if not pending:
            return
        event = torch.npu.Event()
        event.record()
        for request in pending:
            request.current_event = event
            assert self.kv_send_thread is not None
            self.kv_send_thread.add_request(request)


class AscendStoreConnector(MooncakeStoreConnector):
    """Reuse upstream lookup, boundary pins, acknowledgements and recovery."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        if extra.get("pool_prefix_mode") != "hash":
            raise ValueError("This entrypoint requires pool_prefix_mode='hash'; use the original module for block_only")
        unsupported = (
            extra.get("backend", "mooncake").lower() != "mooncake"
            or extra.get("use_layerwise", False)
            or not extra.get("load_async", True)
            or extra.get("consumer_is_to_put", False)
            or extra.get("custom_mem_pool")
            or extra.get("store_tp_size") is not None
            or extra.get("enable_store_tp_lcm", False)
            or extra.get("store_chunk_size") is not None
            or vllm_config.parallel_config.prefill_context_parallel_size != 1
            or vllm_config.parallel_config.decode_context_parallel_size != 1
            or vllm_config.model_config.enable_sleep_mode
        )
        if unsupported:
            raise ValueError(
                "Hash recovery requires rank-local Mooncake, async bulk loads, no custom memory pool or CP"
            )
        if kv_cache_config is None or not hasattr(kv_cache_config, "prefix_cacheable_groups"):
            raise ValueError("Hash recovery requires Core prefix-cacheable groups and boundary-state handoff")
        if not hasattr(MooncakeStoreScheduler, "register_finished_partial_tail"):
            raise ValueError("Core Store scheduler lacks partial-tail handoff support")
        if kv_cache_config.hisparse_host_num_blocks is not None:
            raise ValueError("Hash recovery does not support HiSparse host caches")
        self._validate_kv_cache_config(vllm_config, kv_cache_config)
        # Upstream's initializer constructs its CUDA worker directly. Reuse all
        # protocol hooks unchanged, substituting only the NPU worker here.
        KVConnectorBase_V1.__init__(self, vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self._capacity_only = (
            self.kv_role == "kv_consumer"
            and not extra.get("enable_lookup", True)
            and not extra.get("save_decode_cache", False)
        )
        self._kv_cache_config = kv_cache_config
        self._kv_cache_events = None
        self.connector_scheduler = None
        self.connector_worker = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakeStoreScheduler(vllm_config, kv_cache_config)
            _align_store_eagle_policy(
                self.connector_scheduler._store_coord, _engine_transfer_config(vllm_config, kv_cache_config)
            )
        else:
            self.connector_worker = AscendHashStoreWorker(vllm_config, kv_cache_config)
