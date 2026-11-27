import ctypes
import importlib
import os
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, cast

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    resolve_backend,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_cache import DSparkPrefixKeys
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    get_layerwise_reuse_backing_bytes,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
    LoadSpec,
    ReqMeta,
    RequestTracker,
    StoreLookupResult,
    block_hash_to_str,
    get_block_hashes,
    get_sharded_mamba_group_ids,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import BaseKVPoolScheduler

_SCHEDULER_READ_LEASE_TTL_MS = 5 * 60 * 1000
_scheduler_lease_identity: tuple[int, int] | None = None


@dataclass
class _SchedulerReadPin:
    keys: tuple[str, ...]
    in_flight_steps: int = 0
    terminal: bool = False


def _enable_metadata_lease_namespace() -> None:
    global _scheduler_lease_identity
    pid = os.getpid()
    if _scheduler_lease_identity is None or _scheduler_lease_identity[0] != pid:
        _scheduler_lease_identity = (pid, (1 << 31) + secrets.randbelow((1 << 31) - 1))
    try:
        package = importlib.import_module("memcache_hybrid")
        extension = importlib.import_module("_pymmc")
        extension_path = Path(extension.__file__).resolve()
        if extension_path.parent != Path(package.__file__).resolve().parent:
            raise RuntimeError("Memcache scheduler lease package and extension come from different wheels")
        # Resolve through the loaded extension's link scope, not a separately
        # opened .so that the extension might not actually be using.
        capability = ctypes.CDLL(str(extension_path)).mmc_metadata_read_lease_namespace_v1
    except (AttributeError, ImportError, OSError) as error:
        raise RuntimeError("Loaded Memcache Hybrid lacks the scheduler read lease namespace v1") from error
    capability.argtypes = []
    capability.restype = ctypes.c_int
    if capability() != 1:
        raise RuntimeError(f"Memcache scheduler read lease namespace version mismatch in {extension_path}")
    envs.set_memcache_metadata_lease_owner_id(_scheduler_lease_identity[1])


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
        self._pin_external_prefix = extra_config.get("layerwise_pin_external_prefix", False)
        if type(self._pin_external_prefix) is not bool:
            raise ValueError("layerwise_pin_external_prefix must be a boolean")
        if self._pin_external_prefix:
            if backend_name != "memcache" or not self.use_layerwise_transfer:
                raise ValueError("Scheduler read leases require Memcache layerwise GVA")
            if vllm_config.cache_config.enable_prefix_caching:
                raise ValueError("Scheduler read leases require local APC disabled")
            _enable_metadata_lease_namespace()
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
        cacheable_block_sizes = [self.original_block_size[i] for i in self.cacheable_group_ids]
        if (
            self.use_layerwise
            and self.backend_name != "memcache"
            and not self.use_block_key_layerwise
            and len(cacheable_block_sizes) != len(self.original_block_size)
        ):
            raise ValueError("AscendStore private KV state requires Mooncake block-key or non-layerwise transfer")

        if self._pin_external_prefix:
            if not self.layerwise_offload:
                raise ValueError("Scheduler read leases require layerwise KV offload")
            self._pending_read_pins: dict[str, _SchedulerReadPin] = {}
            self._active_read_pins: dict[str, _SchedulerReadPin] = {}
            self._read_pin_references: dict[str, int] = {}
            self._read_pin_tickets = count()
            self._read_pin_steps: dict[int, tuple[list[_SchedulerReadPin], set[int]]] = {}
            self._expected_worker_count = vllm_config.parallel_config.world_size

    def _external_read_pin_keys(self, request: Request, spec: LoadSpec) -> tuple[str, ...]:
        # Use the worker's initial GVA boundary and the existing physical shard
        # key builder, including the sparse hybrid reachability mask.
        tokens = spec.kvpool_cached_tokens if self.use_eagle else spec.pool_hit_tokens
        masks = self.cache_coordinator.load_mask(request.block_hashes, tokens) if self.cache_coordinator else None
        keys: dict[str, None] = {}
        for group_id in self.cacheable_group_ids:
            block_size = self.grouped_block_size[group_id]
            block_count, remainder = divmod(tokens, block_size)
            hashes = get_block_hashes(request.block_hashes, block_size, self.hash_block_size)
            # An initial public hit cannot promise a private partial snapshot.
            if remainder or block_count > len(hashes):
                return ()
            mask = masks[group_id] if masks is not None and group_id < len(masks) else None
            needed = (idx for idx in range(block_count) if mask is None or idx >= len(mask) or mask[idx])
            for index in needed:
                for key in self._make_layerwise_hit_check_keys(group_id, block_hash_to_str(hashes[index])):
                    keys[key] = None
        return tuple(keys)

    def _remove_native_read_leases(self, keys: list[str]) -> None:
        store = getattr(self, "store_scheduler", None)
        if not keys:
            return
        if store is None:
            logger.warning("Scheduler store unavailable during read lease cleanup for %d keys", len(keys))
            return
        try:
            status = store.batch_remove_lease(keys)
        except Exception:
            logger.exception("Scheduler read lease cleanup failed for %d keys", len(keys))
            return
        if status != 0:
            logger.error("Scheduler read lease cleanup failed for %d keys, status=%s", len(keys), status)

    def _reserve_external_read_pins(self, keys: tuple[str, ...]) -> _SchedulerReadPin | None:
        store = getattr(self, "store_scheduler", None)
        if not keys or store is None:
            return None
        new_keys = [key for key in keys if key not in self._read_pin_references]
        if new_keys:
            try:
                results = store.batch_add_lease(new_keys, _SCHEDULER_READ_LEASE_TTL_MS)
            except Exception:
                # Grants may have committed before the batch reply was lost.
                logger.exception("Scheduler read lease acquisition failed for %d keys", len(new_keys))
                self._remove_native_read_leases(new_keys)
                return None
            if not isinstance(results, (list, tuple)):
                self._remove_native_read_leases(new_keys)
                return None
            successes = [
                key for key, status in zip(new_keys, results, strict=False) if type(status) is int and status == 0
            ]
            if len(results) != len(new_keys) or len(successes) != len(new_keys):
                self._remove_native_read_leases(successes)
                return None
            # An eviction already committing may have won the race.
            try:
                readable = store.batch_is_readable(new_keys)
            except Exception:
                logger.exception("Scheduler read lease recheck failed for %d keys", len(new_keys))
                self._remove_native_read_leases(successes)
                return None
            if (
                not isinstance(readable, (list, tuple))
                or len(readable) != len(new_keys)
                or not all(status is True for status in readable)
            ):
                self._remove_native_read_leases(successes)
                return None
        for key in keys:
            self._read_pin_references[key] = self._read_pin_references.get(key, 0) + 1
        return _SchedulerReadPin(keys)

    def _release_external_read_pins(self, pin: _SchedulerReadPin) -> None:
        keys = []
        for key in pin.keys:
            owners = self._read_pin_references[key] - 1
            if owners:
                self._read_pin_references[key] = owners
            else:
                self._read_pin_references.pop(key)
                keys.append(key)
        self._remove_native_read_leases(keys)

    def _finish_external_read_pins(self, request_id: str) -> None:
        pin = self._active_read_pins.pop(request_id, None)
        if pin is not None:
            pin.terminal = True
            if pin.in_flight_steps == 0:
                self._release_external_read_pins(pin)

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int, bool]:
        if not getattr(self, "_pin_external_prefix", False):
            return super().get_num_new_matched_tokens(request, num_computed_tokens)
        # A nonzero resident prefix would prevent recomputing the whole input
        # when pinning fails. The opt-in setup requires local APC disabled.
        if num_computed_tokens:
            raise RuntimeError("Scheduler read leases require an empty resident prefix")
        prior = self._pending_read_pins.pop(request.request_id, None)
        if prior is not None:
            self._release_external_read_pins(prior)
        result = super().get_num_new_matched_tokens(request, num_computed_tokens)
        spec = self.load_specs.get(request.request_id)
        if spec is None or not spec.can_load:
            return result
        pin = self._reserve_external_read_pins(self._external_read_pin_keys(request, spec))
        if pin is None:
            self.load_specs.pop(request.request_id, None)
            logger.debug("Scheduler read lease unavailable for %s; computing the prefix", request.request_id)
            return 0, False
        self._pending_read_pins[request.request_id] = pin
        return result

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> AscendConnectorMetadata:
        if not getattr(self, "_pin_external_prefix", False):
            return cast(AscendConnectorMetadata, super().build_connector_meta(scheduler_output))
        for request_id in scheduler_output.finished_req_ids | scheduler_output.preempted_req_ids:
            self._finish_external_read_pins(request_id)
        pending = self._pending_read_pins
        self._pending_read_pins = {}
        try:
            meta = cast(AscendConnectorMetadata, super().build_connector_meta(scheduler_output))
        except Exception:
            for request_id, pin in pending.items():
                self.load_specs.pop(request_id, None)
                self._release_external_read_pins(pin)
            raise
        step_pins = []
        for request in meta.requests:
            request_id = request.req_id
            spec = request.load_spec
            if spec is None or not spec.can_load:
                continue
            if request_id in pending and not spec.layerwise_previous_step:
                self._finish_external_read_pins(request_id)
                self._active_read_pins[request_id] = pending.pop(request_id)
            pin = self._active_read_pins.get(request_id)
            if pin is not None:
                pin.in_flight_steps += 1
                step_pins.append(pin)
        for request_id, pin in pending.items():
            self.load_specs.pop(request_id, None)
            self._release_external_read_pins(pin)
        if step_pins:
            ticket = next(self._read_pin_tickets)
            meta.scheduler_read_pin_ticket = ticket
            self._read_pin_steps[ticket] = (step_pins, set())
        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        super().update_connector_output(connector_output)
        if not getattr(self, "_pin_external_prefix", False):
            return
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, AscendStoreKVConnectorWorkerMetadata):
            return
        for ticket, ranks in meta.scheduler_read_pin_acks.items():
            step = self._read_pin_steps.get(ticket)
            if step is None:
                continue
            pins, observed = step
            observed.update(ranks)
            if len(observed) < self._expected_worker_count:
                continue
            del self._read_pin_steps[ticket]
            for pin in pins:
                pin.in_flight_steps -= 1
                if pin.terminal and pin.in_flight_steps == 0:
                    self._release_external_read_pins(pin)

    def _drop_external_read_pin(self, request_id: str) -> None:
        if getattr(self, "_pin_external_prefix", False):
            pending = self._pending_read_pins.pop(request_id, None)
            if pending is not None:
                self.load_specs.pop(request_id, None)
                self._release_external_read_pins(pending)
            self._finish_external_read_pins(request_id)

    def request_finished(self, request: Request, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        self._drop_external_read_pin(request.request_id)
        return super().request_finished(request, block_ids)

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        self._drop_external_read_pin(request.request_id)
        return super().request_finished_all_groups(request, block_ids)

    def reset_store(self) -> bool:
        if getattr(self, "_pin_external_prefix", False) and self._read_pin_references:
            logger.warning("AscendStore reset deferred until scheduler read leases are released")
            return False
        return super().reset_store()

    def _process_hit_without_alloc(self, store_skip_tokens: int) -> bool:
        return store_skip_tokens > 0

    def _build_req_meta(
        self,
        request_tracker: RequestTracker,
        block_hashes: list[BlockHash],
        load_spec: LoadSpec | None,
        prompt_token_ids: list[int],
        skip_save: bool | None,
    ) -> ReqMeta | None:
        request_tuple = self._unfinished_requests.get(request_tracker.req_id)
        if self.layerwise_offload and request_tuple is not None:
            # Verifier rows have no corresponding input KV or state to snapshot.
            request_tracker.token_len = min(request_tracker.token_len, len(request_tuple[0].all_token_ids))
        return super()._build_req_meta(request_tracker, block_hashes, load_spec, prompt_token_ids, skip_save)

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

    def _lookup_external_hit_tokens(self, request: Request, num_computed_tokens: int) -> StoreLookupResult | None:
        prompt_token_len = len(request.prompt_token_ids)
        if self.use_block_key_layerwise:
            token_len = self._floor_to_cache_transfer_granularity(prompt_token_len)
            if not self.use_eagle and len(self.cacheable_group_ids) != len(self.original_block_size):
                token_len = self._floor_to_cache_transfer_granularity(max(prompt_token_len - 1, 0))
            if token_len < self.cache_transfer_granularity:
                return None
            return StoreLookupResult(self._get_block_key_layerwise_hit_tokens(request, token_len, num_computed_tokens))

        if self.use_layerwise_transfer:
            return StoreLookupResult(self._get_layerwise_hit_tokens(request, prompt_token_len, num_computed_tokens))

        raise RuntimeError(f"Unsupported layerwise data plane: {self.layerwise_data_plane!r}")

    def _get_layerwise_hit_tokens(
        self,
        request: "Request",
        token_len: int,
        num_computed_tokens: int,
    ) -> int:
        self._get_or_create_request_tracker(request.request_id)
        if self.layerwise_offload and (
            self.num_mamba_speculative_blocks_by_group or len(self.cacheable_group_ids) != len(self.original_block_size)
        ):
            # Resolve a saved checkpoint at the replayable endpoint before looking up the prefix.
            token_len = min(token_len, self._floor_to_cache_transfer_granularity(request.num_tokens - 1))
            if token_len <= 0:
                return 0
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
        if group_id in self.sharded_mamba_group_ids or group_id in self.num_mamba_speculative_blocks_by_group:
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
