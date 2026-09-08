# SPDX-License-Identifier: Apache-2.0
"""vLLM coordinator for the experimental exact-LCM Jenga prefix cache."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from math import lcm

from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheCoordinator,
)
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import (
    CrossAttentionSpec,
    KVCacheConfig,
    MambaSpec,
)
from vllm.v1.request import Request

from .jenga_prefix_runtime import JengaPrefixRuntimePool
from .single_type_kv_cache_manager import get_manager_for_kv_cache_spec
from .typed_kv_cache import TypedKVCachePlan

JENGA_STATE_CHECKPOINT_INTERVAL_TOKENS = 512


@dataclass(slots=True)
class _ManagerSnapshot:
    req_blocks_present: bool
    req_blocks: tuple[KVCacheBlock, ...]
    keyed_state: dict[str, tuple[bool, object]]
    set_membership: dict[str, bool]
    list_state: dict[str, tuple[object, ...]]


@dataclass(slots=True)
class _PendingAdmission:
    counts_by_group: dict[int, int]
    protected_by_group: dict[int, tuple[KVCacheBlock, ...]]
    timestamp: float
    prepared: bool = False
    manager_snapshots: tuple[_ManagerSnapshot, ...] | None = None


class JengaPrefixKVCacheCoordinator(HybridKVCacheCoordinator):
    """Hybrid coordinator backed by Jenga's exact-LCM two-level policy.

    The vLLM admission API exposes one scalar block count even though this
    allocator has heterogeneous page sizes. Like the address-table MVP, this
    coordinator returns a boolean 0/1 sentinel: the aggregate pool reports
    zero free scalar blocks, so 0 means the complete multi-group transaction
    fits and 1 means it does not.

    Prefix lookup stays group-aware through the upstream layer-type managers.
    Whole-page hits are reconciled at ``scheduler_block_size``; fine-grained
    partial-hit copy-on-write is deliberately disabled in the runtime pool.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        plan: TypedKVCachePlan,
        max_model_len: int,
        max_in_flight_tokens: int,
        scheduler_block_size: int,
        hash_block_size: int,
        *,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        use_eagle: bool = False,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
        state_checkpoint_interval_tokens: int = (JENGA_STATE_CHECKPOINT_INTERVAL_TOKENS),
    ) -> None:
        if plan.is_addressed or plan.is_partitioned:
            raise ValueError("Jenga prefix coordinator requires an exact-LCM plan")
        if dcp_world_size != 1 or pcp_world_size != 1:
            raise ValueError("Jenga prefix coordinator does not support DCP or PCP")
        if use_eagle:
            raise ValueError("Jenga prefix coordinator does not support speculative decoding")
        if enable_kv_cache_events:
            raise ValueError("Jenga prefix coordinator does not support cache events")
        if metrics_collector is not None:
            raise ValueError("Jenga prefix coordinator does not expose block metrics")
        if (
            isinstance(state_checkpoint_interval_tokens, bool)
            or not isinstance(state_checkpoint_interval_tokens, int)
            or state_checkpoint_interval_tokens <= 0
        ):
            raise ValueError("state checkpoint interval must be a positive integer")
        if len(kv_cache_config.kv_cache_groups) != len(plan.specs):
            raise ValueError("the Jenga plan must describe every KV cache group")
        if any(isinstance(group.kv_cache_spec, CrossAttentionSpec) for group in kv_cache_config.kv_cache_groups):
            raise ValueError("Jenga prefix coordinator does not support cross attention")
        if any(
            type(group.kv_cache_spec).__name__ == "SinkFullAttentionSpec" for group in kv_cache_config.kv_cache_groups
        ):
            raise ValueError("Jenga prefix coordinator does not support sink attention")
        if (
            isinstance(scheduler_block_size, bool)
            or not isinstance(scheduler_block_size, int)
            or scheduler_block_size <= 0
        ):
            raise ValueError("scheduler block size must be a positive integer")
        if isinstance(hash_block_size, bool) or not isinstance(hash_block_size, int) or hash_block_size <= 0:
            raise ValueError("hash block size must be a positive integer")
        if scheduler_block_size % hash_block_size:
            raise ValueError("scheduler block size must align the hash block size")

        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            typed_spec = plan.spec(group_id)
            if isinstance(group.kv_cache_spec, MambaSpec) and group.kv_cache_spec.mamba_cache_mode != "align":
                raise ValueError("Jenga recurrent-state prefix caching requires align mode")
            if typed_spec.block_size_tokens != group.kv_cache_spec.block_size:
                raise ValueError(f"group {group_id} block size differs between config and plan")
            if typed_spec.page_size_bytes != group.kv_cache_spec.page_size_bytes:
                raise ValueError(f"group {group_id} page size differs between config and plan")
            if scheduler_block_size % typed_spec.block_size_tokens:
                raise ValueError("scheduler block size must align every Jenga group")
            if typed_spec.block_size_tokens % hash_block_size:
                raise ValueError("every Jenga group must align the hash block size")

        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.max_in_flight_tokens = max_in_flight_tokens
        self.max_num_batched_tokens = max_in_flight_tokens
        self.enable_caching = True
        self.scheduler_block_size = scheduler_block_size
        self.hash_block_size = hash_block_size
        self.dcp_world_size = 1
        self.pcp_world_size = 1
        self.eagle_group_ids: set[int] = set()
        self.use_eagle = False
        self.enable_partial_hash_hits = False
        self.state_checkpoint_interval_tokens = state_checkpoint_interval_tokens
        # A cached state must land on both the paper interval and a legal
        # whole-page/common-prefix boundary. For a large runtime block this can
        # be coarser than 512; the value is exposed for evidence reporting.
        self.effective_state_checkpoint_interval_tokens = lcm(
            state_checkpoint_interval_tokens,
            scheduler_block_size,
        )
        self.retention_interval = self.effective_state_checkpoint_interval_tokens

        self.typed_pool = JengaPrefixRuntimePool(
            plan,
            hash_block_size,
            enable_kv_cache_events=False,
            metrics_collector=None,
        )
        self.block_pool = self.typed_pool
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=group.kv_cache_spec,
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
                block_pool=self.typed_pool.for_group(group_id),
                enable_caching=True,
                kv_cache_group_id=group_id,
                dcp_world_size=1,
                pcp_world_size=1,
                scheduler_block_size=scheduler_block_size,
                needs_kv_cache_zeroing=kv_cache_config.needs_kv_cache_zeroing,
            )
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        )
        self.verify_and_split_kv_cache_groups()

        self._clock = 0
        self._pending_admissions: dict[str, _PendingAdmission] = {}
        self._poisoned_error: str | None = None

    def _next_timestamp(self) -> float:
        self._clock += 1
        return float(self._clock)

    @staticmethod
    def _relevant_local_hits(
        manager,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
    ) -> tuple[KVCacheBlock, ...]:
        if request_id in manager.num_cached_block:
            if new_computed_blocks:
                raise ValueError("a running request cannot add new prefix-cache hits")
            return ()

        num_request_blocks = len(manager.req_to_blocks.get(request_id, ()))
        num_skipped_blocks = manager.get_num_skipped_tokens(total_computed_tokens) // manager.block_size
        num_skipped_new_blocks = max(0, num_skipped_blocks - num_request_blocks)
        return tuple(new_computed_blocks[num_skipped_new_blocks:])

    def _build_admission(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool,
    ) -> _PendingAdmission:
        if len(new_computed_blocks) != len(self.single_type_managers):
            raise ValueError("prefix-cache block groups do not match the Jenga plan")
        if new_computed_blocks and any(new_computed_blocks):
            if num_local_computed_tokens % self.scheduler_block_size:
                raise ValueError("Jenga prefix hits must align the common scheduler block size")

        counts: dict[int, int] = {}
        protected: dict[int, tuple[KVCacheBlock, ...]] = {}
        for group_id, manager in enumerate(self.single_type_managers):
            group_hits = new_computed_blocks[group_id]
            reported = manager.get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                group_hits,
                total_computed_tokens,
                num_local_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
            )
            # Mamba uses ``num_gpu_blocks + 1`` to defer a state that was
            # published by another request in this scheduling step.  Detect
            # that sentinel before subtracting evictable hits; otherwise it can
            # be disguised as an ordinary (very large) allocation request.
            if reported > manager.block_pool.num_gpu_blocks:
                counts[group_id] = self.typed_pool.plan.num_blocks(group_id) + 1
                return _PendingAdmission(counts, protected, self._next_timestamp())
            relevant_hits = self._relevant_local_hits(
                manager,
                request_id,
                group_hits,
                total_computed_tokens,
            )
            evictable_hits = manager._get_num_evictable_blocks(relevant_hits)
            num_new_pages = reported - evictable_hits
            if num_new_pages < 0:
                raise AssertionError("manager admission count is smaller than its evictable hits")
            counts[group_id] = num_new_pages
            protected[group_id] = tuple(block for block in relevant_hits if not block.is_null)
        return _PendingAdmission(counts, protected, self._next_timestamp())

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        if self._poisoned_error is not None:
            raise RuntimeError(
                f"Jenga prefix coordinator is unavailable after a failed rollback: {self._poisoned_error}"
            )
        if num_encoder_tokens:
            raise ValueError("Jenga prefix coordinator does not support encoder tokens")
        existing = self._pending_admissions.get(request_id)
        if existing is not None and existing.prepared:
            raise RuntimeError("cannot replace a prepared Jenga allocation admission")
        snapshots = (
            existing.manager_snapshots
            if existing is not None and existing.manager_snapshots is not None
            else tuple(self._snapshot_manager(manager, request_id) for manager in self.single_type_managers)
        )
        try:
            admission = self._build_admission(
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_local_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap,
            )
        except Exception:
            # A full-sequence request may perform more than one admission
            # check before allocation.  Never leave the earlier, unprepared
            # decision available after a later check fails.
            self._pending_admissions.pop(request_id, None)
            for manager, snapshot in zip(self.single_type_managers, snapshots):
                self._restore_manager(manager, request_id, snapshot)
            raise
        admission.manager_snapshots = snapshots
        if any(
            count > self.typed_pool.plan.num_blocks(group_id) for group_id, count in admission.counts_by_group.items()
        ):
            self._pending_admissions.pop(request_id, None)
            for manager, snapshot in zip(self.single_type_managers, snapshots):
                self._restore_manager(manager, request_id, snapshot)
            return 1
        try:
            fits = self.typed_pool.can_allocate(
                request_id,
                admission.counts_by_group,
                protected_blocks=admission.protected_by_group,
            )
        except Exception:
            self._pending_admissions.pop(request_id, None)
            for manager, snapshot in zip(self.single_type_managers, snapshots):
                self._restore_manager(manager, request_id, snapshot)
            raise
        if fits:
            self._pending_admissions[request_id] = admission
            return 0
        self._pending_admissions.pop(request_id, None)
        for manager, snapshot in zip(self.single_type_managers, snapshots):
            self._restore_manager(manager, request_id, snapshot)
        return 1

    def _require_admission(self, request_id: str) -> _PendingAdmission:
        admission = self._pending_admissions.get(request_id)
        if admission is None:
            raise RuntimeError(f"request {request_id!r} has no successful Jenga admission")
        return admission

    def _prepare(self, request_id: str, admission: _PendingAdmission) -> None:
        if admission.prepared:
            return
        if admission.manager_snapshots is None:
            admission.manager_snapshots = tuple(
                self._snapshot_manager(manager, request_id) for manager in self.single_type_managers
            )
        self.typed_pool.prepare_allocation(
            request_id,
            admission.counts_by_group,
            protected_blocks=admission.protected_by_group,
        )
        admission.prepared = True

    @staticmethod
    def _snapshot_manager(manager, request_id: str) -> _ManagerSnapshot:
        req_blocks_present = request_id in manager.req_to_blocks
        req_blocks = tuple(manager.req_to_blocks.get(request_id, ()))
        keyed_state: dict[str, tuple[bool, object]] = {}
        for attribute in (
            "num_cached_block",
            "_partial_hit_reqs",
            "last_state_block_idx",
            "_producer_partial_tail_reqs",
        ):
            mapping = getattr(manager, attribute, None)
            if mapping is not None:
                keyed_state[attribute] = (
                    request_id in mapping,
                    mapping.get(request_id),
                )
        set_membership: dict[str, bool] = {}
        for attribute in ("_allocated_block_reqs",):
            values = getattr(manager, attribute, None)
            if values is not None:
                set_membership[attribute] = request_id in values
        list_state: dict[str, tuple[object, ...]] = {}
        for attribute in (
            "new_block_ids",
            "_pending_cow_copies",
            "_pending_partial_tail_offloads",
        ):
            values = getattr(manager, attribute, None)
            if values is not None:
                list_state[attribute] = tuple(values)
        return _ManagerSnapshot(
            req_blocks_present,
            req_blocks,
            keyed_state,
            set_membership,
            list_state,
        )

    @staticmethod
    def _restore_manager(
        manager,
        request_id: str,
        snapshot: _ManagerSnapshot,
    ) -> None:
        if snapshot.req_blocks_present:
            blocks = manager.req_to_blocks.get(request_id)
            if blocks is None:
                manager.req_to_blocks[request_id] = list(snapshot.req_blocks)
            else:
                blocks[:] = snapshot.req_blocks
        else:
            manager.req_to_blocks.pop(request_id, None)
        for attribute, (present, value) in snapshot.keyed_state.items():
            mapping = getattr(manager, attribute)
            if present:
                mapping[request_id] = value
            else:
                mapping.pop(request_id, None)
        for attribute, present in snapshot.set_membership.items():
            values = getattr(manager, attribute)
            if present:
                values.add(request_id)
            else:
                values.discard(request_id)
        for attribute, values in snapshot.list_state.items():
            getattr(manager, attribute)[:] = values

    def _rollback_failed_allocation(
        self,
        request_id: str,
        admission: _PendingAdmission,
    ) -> None:
        errors: list[str] = []
        if self.typed_pool.has_prepared_allocation(request_id):
            try:
                self.typed_pool.abort_allocation(request_id)
            except Exception as error:  # pragma: no cover - fail-closed guard
                errors.append(f"runtime abort failed: {error}")
        snapshots = admission.manager_snapshots
        if snapshots is not None:
            for group_id, (manager, snapshot) in enumerate(zip(self.single_type_managers, snapshots)):
                try:
                    self._restore_manager(manager, request_id, snapshot)
                except Exception as error:  # pragma: no cover - fail-closed guard
                    errors.append(f"manager {group_id} restore failed: {error}")
        admission.prepared = False
        self._pending_admissions.pop(request_id, None)
        if errors:
            self._poisoned_error = "; ".join(errors)
            raise RuntimeError(self._poisoned_error)

    def _cancel_if_unconsumed(
        self,
        request_id: str,
        admission: _PendingAdmission,
    ) -> bool:
        if not admission.prepared:
            return True
        # Terminal request cleanup may cancel only an untouched staging
        # decision. Failure rollback uses ``abort_allocation`` instead and can
        # reverse logged block-table consumption and cache-hit touches.
        with suppress(RuntimeError):
            self.typed_pool.cancel_allocation(request_id)
            admission.prepared = False
            return True
        return False

    def _discard_admission(self, request_id: str) -> None:
        admission = self._pending_admissions.pop(request_id, None)
        if admission is None or not admission.prepared:
            return
        with self.typed_pool.request_context(
            request_id,
            timestamp=admission.timestamp,
        ):
            if not self._cancel_if_unconsumed(request_id, admission):
                raise RuntimeError("cannot discard a partially consumed Jenga allocation")

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        if num_external_computed_tokens:
            raise ValueError("Jenga prefix coordinator does not support external KV")
        admission = self._require_admission(request_id)
        with self.typed_pool.request_context(
            request_id,
            timestamp=admission.timestamp,
        ):
            try:
                self._prepare(request_id, admission)
                if any(request_id in manager.num_cached_block for manager in self.single_type_managers):
                    if any(new_computed_blocks):
                        raise ValueError("a running request cannot add new prefix-cache hits")
                    return
                for group_id, manager in enumerate(self.single_type_managers):
                    manager.add_local_computed_blocks(
                        request_id,
                        new_computed_blocks[group_id],
                        num_local_computed_tokens,
                        num_external_computed_tokens,
                    )
            except Exception:
                self._rollback_failed_allocation(request_id, admission)
                raise

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[KVCacheBlock], ...]:
        if num_encoder_tokens:
            raise ValueError("Jenga prefix coordinator does not support encoder tokens")
        admission = self._require_admission(request_id)
        with self.typed_pool.request_context(
            request_id,
            timestamp=admission.timestamp,
        ):
            try:
                self._prepare(request_id, admission)
                new_blocks = tuple(
                    manager.allocate_new_blocks(
                        request_id,
                        num_tokens,
                        num_tokens_main_model,
                    )
                    for manager in self.single_type_managers
                )
                self.typed_pool.finish_allocation(request_id)
            except Exception:
                self._rollback_failed_allocation(request_id, admission)
                raise
        del self._pending_admissions[request_id]
        return new_blocks

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        aligned_tokens = num_computed_tokens // self.scheduler_block_size * self.scheduler_block_size
        with self.typed_pool.request_context(
            request.request_id,
            timestamp=self._next_timestamp(),
        ):
            for manager in self.single_type_managers:
                if isinstance(manager.kv_cache_spec, MambaSpec):
                    self._cache_state_checkpoints(
                        manager,
                        request,
                        aligned_tokens,
                    )
                else:
                    manager.cache_blocks(
                        request,
                        aligned_tokens,
                        retention_interval=None,
                    )

    def _cache_state_checkpoints(
        self,
        manager,
        request: Request,
        aligned_tokens: int,
    ) -> None:
        """Publish recurrent state only at the configured Jenga interval.

        Upstream sparse retention also keeps the final prompt replay boundary.
        Jenga's evaluated policy uses periodic state checkpoints, so this path
        builds the mask directly and does not add an arbitrary prompt-tail
        checkpoint. Applying the same mechanism to GDN is an engineering
        extension; the paper evaluates Mamba/linear recurrent state.
        """

        num_cached_blocks = manager.num_cached_block.get(request.request_id, 0)
        num_full_blocks = aligned_tokens // manager.block_size
        if num_cached_blocks >= num_full_blocks:
            return
        interval = self.effective_state_checkpoint_interval_tokens
        block_mask = [
            ((block_index + 1) * manager.block_size) % interval == 0
            for block_index in range(num_cached_blocks, num_full_blocks)
        ]
        manager.block_pool.cache_full_blocks(
            request=request,
            blocks=manager.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=manager.block_size,
            kv_cache_group_id=manager.kv_cache_group_id,
            block_mask=block_mask,
        )
        manager.num_cached_block[request.request_id] = num_full_blocks
        cached_this_step = getattr(manager, "cached_blocks_this_step", None)
        if cached_this_step is not None:
            for block in manager.req_to_blocks[request.request_id][num_cached_blocks:num_full_blocks]:
                if not block.is_null and block.block_hash is not None:
                    cached_this_step.add(block.block_hash)

    def free(self, request_id: str) -> None:
        self._discard_admission(request_id)
        with self.typed_pool.request_context(
            request_id,
            timestamp=self._next_timestamp(),
        ):
            KVCacheCoordinator.free(self, request_id)
        self.typed_pool.clear_request_affinity(request_id)

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        self._discard_admission(request_id)
        blocks = KVCacheCoordinator.pop_blocks_for_free(self, request_id)
        self.typed_pool.release_for_deferred_free(
            request_id,
            blocks,
            timestamp=self._next_timestamp(),
        )
        self.typed_pool.clear_request_affinity(request_id)
        return blocks

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        with self.typed_pool.request_context(
            request_id,
            timestamp=self._next_timestamp(),
        ):
            KVCacheCoordinator.remove_skipped_blocks(
                self,
                request_id,
                processed_computed_tokens,
                num_prompt_tokens,
            )
