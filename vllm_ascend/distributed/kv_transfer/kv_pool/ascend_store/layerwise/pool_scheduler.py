import time
from collections.abc import Sequence
from itertools import count
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    resolve_backend,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_cache import DSparkPrefixKeys
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    get_layerwise_reuse_backing_bytes,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.metadata import (
    LAYERWISE_READ_LEASE_TTL_MS,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
    LoadSpec,
    RequestTracker,
    SchedulerGvaReadLease,
    block_hash_to_str,
    get_block_hashes,
    get_sharded_mamba_group_ids,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import BaseKVPoolScheduler


class LayerwiseKVPoolScheduler(BaseKVPoolScheduler):
    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwise: bool,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        backend_name, self.layerwise_backend = resolve_backend(extra_config)
        self.layerwise_data_plane = (
            self.layerwise_backend.layerwise_data_plane if self.layerwise_backend is not None else None
        )
        if use_layerwise and self.layerwise_data_plane not in ("block_key", "gva"):
            raise ValueError(
                f"Backend {backend_name!r} does not support layerwise KV pooling: "
                f"unsupported layerwise_data_plane={self.layerwise_data_plane!r}"
            )
        self.use_block_key_layerwise = use_layerwise and self.layerwise_data_plane == "block_key"
        self.use_layerwise_transfer = use_layerwise and self.layerwise_data_plane == "gva"
        self.dspark_prefix_keys: DSparkPrefixKeys | None = None
        self.dspark_draft_only_group_ids = frozenset(getattr(kv_cache_config, "dspark_draft_only_group_ids", ()))
        super().__init__(vllm_config, use_layerwise, kv_cache_config)
        self.sharded_mamba_group_ids = get_sharded_mamba_group_ids(kv_cache_config)
        self.load_async = False
        self.layerwise_offload = (
            self.use_layerwise_transfer and get_layerwise_reuse_backing_bytes(kv_cache_config) is not None
        )
        self.block_key_hybrid = self.use_block_key_layerwise and self.use_hybrid
        self.layerwise_keys = (
            self.layerwise_backend.bind_layerwise_keys(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                model_name=vllm_config.model_config.model.split("/")[-1],
                use_hybrid=self.use_hybrid,
                grouped_block_size=self.grouped_block_size,
            )
            if self.layerwise_backend is not None and self.layerwise_data_plane is not None
            else None
        )
        if self.use_block_key_layerwise and self.tp_mismatch:
            raise ValueError("Mooncake layerwise does not yet support prefill/decode TP mismatch")
        self.layerwise_max_transfer_blocks = int(extra_config.get("layerwise_max_transfer_blocks", 0))
        parallel_config = vllm_config.parallel_config
        self.use_scheduler_lookup = (
            extra_config.get("use_scheduler_lookup") is True
            and self.use_layerwise_transfer
            and backend_name == "memcache"
            # Only local ranks share the scheduler's monotonic clock.
            and parallel_config.nnodes == 1
            and parallel_config.distributed_executor_backend != "ray"
            and parallel_config.data_parallel_backend != "ray"
            and getattr(self.store_scheduler.store, "supports_scheduler_read_lease", False) is True
        )
        self._read_lease_ids = count()
        self._read_lease_worker_count = parallel_config.world_size
        self._scheduler_read_leases: dict[int, tuple[list[str], int, float]] = {}
        if extra_config.get("use_scheduler_lookup") is True:
            logger.info(
                "AscendStore layerwise read leases are acquired by the %s.",
                "scheduler when possible" if self.use_scheduler_lookup else "worker",
            )
        cacheable_block_sizes = [self.original_block_size[i] for i in self.cacheable_group_ids]
        if (
            self.use_layerwise
            and self.backend_name != "memcache"
            and not self.use_block_key_layerwise
            and len(cacheable_block_sizes) != len(self.original_block_size)
        ):
            raise ValueError("AscendStore private KV state requires Mooncake block-key or non-layerwise transfer")

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> AscendConnectorMetadata:
        meta = super().build_connector_meta(scheduler_output)
        assert isinstance(meta, AscendConnectorMetadata)
        if self.use_scheduler_lookup:
            meta.scheduler_gva_read_lease = self._prepare_scheduler_read_lease(meta)
        return meta

    def _scheduler_read_keys(self, meta: AscendConnectorMetadata) -> list[str]:
        keys: dict[str, None] = {}
        seen_hashes = {group_id: set() for group_id in self.cacheable_group_ids}
        for request in meta.requests:
            load = request.load_spec
            if load is None or not load.can_load:
                continue
            cached_tokens = load.kvpool_cached_tokens if self.use_eagle else load.pool_hit_tokens
            if cached_tokens <= 0:
                continue
            masks = (
                self.cache_coordinator.load_mask(request.block_hashes, cached_tokens)
                if self.cache_coordinator is not None
                else None
            )
            for group_id in self.cacheable_group_ids:
                block_size = self.grouped_block_size[group_id]
                first = 0 if self.layerwise_offload else load.vllm_cached_tokens // block_size
                hashes = get_block_hashes(request.block_hashes, block_size, self.hash_block_size)
                end = min(cached_tokens // block_size, len(hashes))
                mask = masks[group_id] if masks is not None else None
                for index in range(first, end):
                    if mask is not None and index < len(mask) and not mask[index]:
                        continue
                    block_hash = hashes[index]
                    if block_hash in seen_hashes[group_id]:
                        continue
                    seen_hashes[group_id].add(block_hash)
                    for key in self._make_layerwise_hit_check_keys(group_id, block_hash_to_str(block_hash)):
                        keys[key] = None
        return list(keys)

    def _prepare_scheduler_read_lease(self, meta: AscendConnectorMetadata) -> SchedulerGvaReadLease | None:
        if self.layerwise_offload and any(
            (load := request.load_spec) is not None and load.can_load and load.layerwise_previous_step
            for request in meta.requests
        ):
            # Only the worker can resolve private snapshots for resumed prefill.
            return None

        now = time.monotonic()
        held_keys = set()
        for event_id, (keys, _, expires_after) in list(self._scheduler_read_leases.items()):
            if now >= expires_after:
                # An undelivered completion cannot hold back later steps after the lease expires.
                self._scheduler_read_leases.pop(event_id)
            else:
                held_keys.update(keys)

        leased: list[str] = []
        gvas: dict[str, int] = {}
        try:
            keys = self._scheduler_read_keys(meta)
            # A partial plan would send workers back to local lookup.
            if not keys or not held_keys.isdisjoint(keys):
                return None
            leased = keys
            started = time.monotonic()
            # MemCache only grants a lease on a readable GVA, so one query
            # after acquisition is enough. SSD-only keys stay on the worker path.
            statuses = self.store_scheduler.batch_add_lease(leased, LAYERWISE_READ_LEASE_TTL_MS)
            # If the response is malformed, release every lease that this call could have acquired.
            if statuses is None or len(statuses) != len(leased):
                self._release_scheduler_read_keys(leased)
                return None
            leased = [key for key, result in zip(leased, statuses, strict=True) if result == 0]
            if not leased:
                return None
            expires_after = time.monotonic() + LAYERWISE_READ_LEASE_TTL_MS / 1000
            infos = self.store_scheduler.batch_get_key_info(leased)
            if infos is not None and len(infos) == len(leased):
                for key, info in zip(leased, infos, strict=True):
                    if info is not None and info.size() > 0:
                        address = next((int(gva) for gva in info.gva_list() if 0 < gva < 1 << 63), 0)
                        if address:
                            gvas[key] = address
        except Exception:
            logger.warning("Scheduler GVA read preparation failed; using worker lookup", exc_info=True)
            self._release_scheduler_read_keys(leased)
            return None
        if not gvas:
            self._release_scheduler_read_keys(leased)
            return None

        event_id = next(self._read_lease_ids)
        self._scheduler_read_leases[event_id] = leased, self._read_lease_worker_count, expires_after
        # Leave half the lease lifetime for copies already submitted by a worker.
        return SchedulerGvaReadLease(event_id, gvas, started + LAYERWISE_READ_LEASE_TTL_MS / 2000)

    def _release_scheduler_read_keys(self, keys: list[str]) -> None:
        if not keys:
            return
        try:
            result = self.store_scheduler.batch_remove_lease(keys)
            if result != 0:
                logger.warning("Scheduler GVA read lease cleanup returned %s", result)
        except Exception:
            logger.warning("Scheduler GVA read lease cleanup failed", exc_info=True)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        super().update_connector_output(connector_output)
        worker_meta = connector_output.kv_connector_worker_meta
        if not self.use_scheduler_lookup or not isinstance(worker_meta, AscendStoreKVConnectorWorkerMetadata):
            return
        for event_id, completed in worker_meta.completed_events.items():
            lease = self._scheduler_read_leases.get(event_id)
            if lease is None:
                continue
            keys, remaining, expires_after = lease
            if completed >= remaining:
                self._scheduler_read_leases.pop(event_id)
                self._release_scheduler_read_keys(keys)
            else:
                self._scheduler_read_leases[event_id] = keys, remaining - completed, expires_after

    def _process_hit_without_alloc(self, store_skip_tokens: int) -> bool:
        return store_skip_tokens > 0

    def get_running_target_token_len(self, request: Request, num_new_tokens: int) -> int:
        target_token_len = request.num_computed_tokens + num_new_tokens
        if self.layerwise_offload:
            return target_token_len
        return min(target_token_len, len(request.all_token_ids))

    def _running_load_spec(self, request_tracker: RequestTracker, num_current_tokens: int) -> LoadSpec | None:
        if self.layerwise_offload and num_current_tokens > 0:
            return LoadSpec(
                vllm_cached_tokens=num_current_tokens,
                kvpool_cached_tokens=num_current_tokens,
                can_load=True,
                layerwise_previous_step=True,
            )
        return None

    def _use_hybrid(self, vllm_config: VllmConfig, kv_cache_groups: list[Any] | None) -> bool:
        return super()._use_hybrid(vllm_config, kv_cache_groups) or (
            self.use_block_key_layerwise and kv_cache_groups is not None and len(kv_cache_groups) > 1
        )

    def _lookup_external_hit_tokens(self, request: Request, num_computed_tokens: int) -> int | None:
        prompt_token_len = len(request.prompt_token_ids)
        if self.use_block_key_layerwise:
            token_len = self._floor_to_cache_transfer_granularity(prompt_token_len)
            if not self.use_eagle and len(self.cacheable_group_ids) != len(self.original_block_size):
                token_len = self._floor_to_cache_transfer_granularity(max(prompt_token_len - 1, 0))
            if token_len < self.cache_transfer_granularity:
                return None
            return self._get_block_key_layerwise_hit_tokens(request, token_len, num_computed_tokens)

        if self.use_layerwise_transfer:
            return self._get_layerwise_hit_tokens(request, prompt_token_len, num_computed_tokens)

        raise RuntimeError(f"Unsupported layerwise data plane: {self.layerwise_data_plane!r}")

    def _get_layerwise_hit_tokens(
        self,
        request: "Request",
        token_len: int,
        num_computed_tokens: int,
    ) -> int:
        self._get_or_create_request_tracker(request.request_id)
        if self.cache_coordinator is not None:
            if self.use_eagle:
                final_block_start = (request.num_tokens - 1) // self.lcm_block_size * self.lcm_block_size
                token_len = min(token_len, final_block_start)
                if token_len <= 0:
                    return 0
            return self._lookup_layerwise_with_coordinator(
                request, token_len, 0 if self.layerwise_offload else num_computed_tokens
            )
        return self._lookup_layerwise_contiguous(request, token_len, num_computed_tokens)

    def _get_block_key_layerwise_hit_tokens(
        self,
        request: "Request",
        token_len: int,
        num_computed_tokens: int,
    ) -> int:
        if self.block_key_hybrid:
            return self._lookup_layerwise_with_coordinator(request, token_len)
        return self._lookup_block_key_contiguous(request, token_len, num_computed_tokens)

    def _lookup_layerwise_with_coordinator(
        self,
        request: "Request",
        token_len: int,
        num_computed_tokens: int = 0,
    ) -> int:
        """Reachability-aware hit check for hybrid models.

        Only the blocks the KV cache managers consider reachable (sliding-
        window / compressor-state tails) are queried for reachability-limited
        groups — those groups are stored sparsely by the layerwise save path —
        and the hit length is derived by the coordinator lookup shared with
        the non-layerwise path, so it matches the store-side mask semantics.
        """
        coordinator = self.cache_coordinator
        assert coordinator is not None

        def query_group_hits(
            group_id: int,
            group_block_hashes: Sequence[BlockHash | str],
            lookup_mask: Sequence[bool] | None,
        ) -> list[BlockHash]:
            keys_by_block: list[list[str]] = []
            allowed_hashes: list[BlockHash] = []
            local_blocks = num_computed_tokens // self.grouped_block_size[group_id]
            for block_idx, block_hash in enumerate(group_block_hashes):
                if block_idx < local_blocks:
                    continue
                if lookup_mask is not None and not (block_idx < len(lookup_mask) and lookup_mask[block_idx]):
                    continue
                keys_by_block.append(self._make_layerwise_hit_check_keys(group_id, block_hash_to_str(block_hash)))
                allowed_hashes.append(block_hash)
            block_hits = self._query_layerwise_block_hits(keys_by_block)
            return [block_hash for block_hash, hit in zip(allowed_hashes, block_hits) if hit]

        return coordinator.find_reachable_hit_tokens(
            request.block_hashes,
            token_len,
            query_group_hits,
            log_context=f"hit_check: req={request.request_id}",
            hbm_hit_tokens=num_computed_tokens,
        )

    def _query_contiguous_group_hits(
        self,
        request: "Request",
        token_len: int,
        group_block_sizes: Sequence[int],
        skip_tokens: int,
    ) -> list[tuple[int, int, int]]:
        """Contiguous hit scan shared by both layerwise data planes.

        For each participating group, align the request's block hashes to the
        group's physical block size, skip the blocks below ``skip_tokens``
        (the local prefix), probe backend readability and count leading hit
        blocks. Returns one ``(hit_tokens, hit_blocks, probed_blocks)`` tuple
        per participating group; groups with nothing to probe are omitted.

        Deliberately no cross-request hit cache: in a tiered (DRAM+SSD) pool a
        cached hit can outlive an eviction and over-report external hits,
        while a cached miss can hide a freshly saved block. Every lookup
        re-probes backend readability instead.
        """
        num_hash_blocks = token_len // self.hash_block_size
        block_hashes_to_check = request.block_hashes[:num_hash_blocks]
        results: list[tuple[int, int, int]] = []

        for group_id, effective_block_size in enumerate(group_block_sizes):
            group_block_hashes = get_block_hashes(block_hashes_to_check, effective_block_size, self.hash_block_size)
            query_start_block = min(skip_tokens // effective_block_size, len(group_block_hashes))
            group_block_hashes = group_block_hashes[query_start_block:]
            if not group_block_hashes:
                # Nothing left to probe: the skipped prefix stands as the hit.
                results.append((query_start_block * effective_block_size, query_start_block, 0))
                continue

            keys_by_block = [
                self._make_layerwise_hit_check_keys(group_id, block_hash_to_str(bh)) for bh in group_block_hashes
            ]
            if not keys_by_block:
                continue
            block_hits = self._query_layerwise_block_hits(keys_by_block)
            num_hit_blocks = 0
            for hit in block_hits:
                if hit:
                    num_hit_blocks += 1
                else:
                    break

            results.append(
                (
                    (query_start_block + num_hit_blocks) * effective_block_size,
                    query_start_block + num_hit_blocks,
                    len(keys_by_block),
                )
            )

        return results

    def _lookup_layerwise_contiguous(
        self,
        request: "Request",
        token_len: int,
        num_computed_tokens: int,
    ) -> int:
        # Layer reuse can evict earlier layers from HBM. Otherwise, only
        # probe blocks beyond the complete local prefix to avoid rewarming it.
        skip_tokens = 0 if self.layerwise_offload else num_computed_tokens
        hits_per_group = [
            hit_tokens
            for hit_tokens, _, _ in self._query_contiguous_group_hits(
                request, token_len, self.grouped_block_size, skip_tokens
            )
        ]

        if not hits_per_group:
            logger.debug(
                "hit_check: req=%s token_len=%d no participating groups (all skipped)",
                request.request_id,
                token_len,
            )
            return 0
        hit_tokens = min(hits_per_group)
        logger.debug(
            "hit_check: req=%s token_len=%d hits_per_group=%s hit_tokens=%d",
            request.request_id,
            token_len,
            hits_per_group,
            hit_tokens,
        )
        return hit_tokens

    def _lookup_block_key_contiguous(
        self,
        request: "Request",
        token_len: int,
        num_computed_tokens: int,
    ) -> int:
        # The block-key plane stores one hash-addressed object per block and
        # keeps no per-group local-prefix state: scan the first group only,
        # with nothing to skip.
        del num_computed_tokens
        results = self._query_contiguous_group_hits(request, token_len, [self._block_size], 0)
        if not results:
            return 0
        hit_tokens, num_hit_blocks, num_probed_blocks = results[0]
        logger.info(
            "Block-key layerwise hit check request=%s hit_blocks=%d/%d",
            request.request_id,
            num_hit_blocks,
            num_probed_blocks,
        )
        return hit_tokens

    def _make_layerwise_hit_check_keys(self, group_id: int, block_hash_hex: str) -> list[str]:
        """All-rank keys for scheduler-side hit check, built by the
        backend's protocol module.

        Single-group uses PR #11585 format; multi-group includes group_id.
        The protocol helper covers every PP stage and head, and every PCP
        rank for sharded Mamba groups. PCP replicas share one object.
        """
        assert self.layerwise_keys is not None
        draft_keys = self.dspark_prefix_keys
        if draft_keys is not None and group_id in self.dspark_draft_only_group_ids:
            # Draft-only groups have no ordinary layerwise target object.
            return draft_keys.make_hit_check_keys(block_hash_hex)
        if group_id in self.sharded_mamba_group_ids or group_id in self.num_speculative_blocks_by_group:
            head_or_tp_ranks = self.tp_size
        else:
            head_or_tp_ranks = self.tp_size // self.put_step
        keys = self.layerwise_keys.make_hit_check_keys(group_id, block_hash_hex, head_or_tp_ranks)
        if draft_keys is not None:
            keys.extend(draft_keys.make_hit_check_keys(block_hash_hex))
        return keys

    def _query_layerwise_block_hits(self, keys_by_block: list[list[str]]) -> list[bool]:
        """Probe backend readability and fold physical keys into block hits."""
        all_keys = [key for block_keys in keys_by_block for key in block_keys]
        if not all_keys:
            return []

        keys_per_block = max(len(block_keys) for block_keys in keys_by_block)
        batch_size = (
            self.layerwise_max_transfer_blocks * keys_per_block
            if self.layerwise_max_transfer_blocks > 0
            else len(all_keys)
        )
        readable: list[bool] = []
        for start in range(0, len(all_keys), batch_size):
            batch = all_keys[start : start + batch_size]
            batch_readable = self.store_scheduler.batch_is_readable(batch)
            if len(batch_readable) != len(batch) or any(type(state) is not bool for state in batch_readable):
                raise RuntimeError("Layerwise readability probe returned invalid results")
            readable.extend(batch_readable)

        block_hits = []
        offset = 0
        for block_keys in keys_by_block:
            block_hits.append(all(readable[offset : offset + len(block_keys)]))
            offset += len(block_keys)
        return block_hits
