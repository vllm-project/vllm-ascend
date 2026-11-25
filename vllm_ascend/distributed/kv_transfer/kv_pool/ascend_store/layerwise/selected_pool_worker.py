# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import Counter, deque
from typing import Any, cast

from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.planes.gva import GvaDataPlane
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.selected_kv_plan import stop_transfer_threads

MAX_DEFERRED_LEASE_STEPS, THREAD_JOIN_TIMEOUT_S = 2, 5.0


class SelectedGvaDataPlane(GvaDataPlane):
    def _get_leased_key_infos(self, leased_keys: list[str]) -> Any:
        infos = super()._get_leased_key_infos(leased_keys)
        cast("SelectedKVPoolWorker", self._worker)._record_leased_infos(leased_keys, infos)
        return infos


class SelectedKVPoolWorker(LayerwiseKVPoolWorker):
    """Extends LayerwiseKVPoolWorker with cross-step deferred read-lease accounting.

    Intra-step duplicate keys inherit LayerwiseKVPoolWorker's _step_load_keys set
    folding; across overlapping async_scheduling steps, _lease_refs + _held_removes
    hold removals until every reader's ddr_done fence completes, then issue 1:1
    step-level batch_remove_lease batches.
    """

    def __init__(self, vllm_config: Any, use_layerwise: bool, kv_cache_config: Any = None) -> None:
        super().__init__(vllm_config, use_layerwise, kv_cache_config)
        if isinstance(getattr(self, "_data_plane", None), GvaDataPlane):
            self._data_plane = SelectedGvaDataPlane(self)
        self.async_scheduling = bool(getattr(vllm_config.scheduler_config, "async_scheduling", False))
        self.selected_kv_runtime: Any = None
        self._deferred_leases: deque[tuple[Any, list[str]]] = deque()
        self._lease_refs: Counter[str] = Counter()
        self._held_removes: Counter[str] = Counter()

    def _cnt(self, attr: str) -> Counter[str]:
        # Support lightweight object.__new__ worker stubs in unit/boundary tests.
        if (c := getattr(self, attr, None)) is None:
            setattr(self, attr, c := Counter())
        return c

    def _record_leased_infos(self, keys: list[str], infos: Any) -> None:
        if self.selected_kv_runtime is not None:
            self.selected_kv_runtime.record_leased_infos(keys, infos)

    def attach_selected_runtime(self, runtime: Any) -> None:
        if self.external_slot_release_waiter is not None:
            raise RuntimeError("Selected KV prefill requires sole ownership of the slot-release waiter")
        if not self.set_external_slot_release_waiter(runtime.on_slot_released):
            raise RuntimeError("Selected KV prefill requires the GVA layerwise transfer mode")
        self.selected_kv_runtime = runtime

    def _submit_ready_layer_loads(self) -> None:
        if (rt := self.selected_kv_runtime) is None or (step := rt.step) is None:
            return super()._submit_ready_layer_loads()
        cur = self.current_layer
        with step.lock:
            first = self.next_layer_to_submit
            super()._submit_ready_layer_loads()
            for lid in range(first, self.next_layer_to_submit):
                if (self.layer_load_tasks[lid] or lid >= 2) and lid != cur:
                    rt.defer_indexer(step, lid, cur)

    def reclaim_aborted_step_leases(self) -> None:
        if self._step_load_keys and not getattr(self.selected_kv_runtime, "_poisoned", False):
            if (rt := self.selected_kv_runtime) is not None:
                rt._drain_if_ready()
            self._cnt("_held_removes").update(self._step_load_keys)
            self._step_load_keys.clear()
            self._flush_ready_removes()

    def wait_for_previous_save(self) -> None:
        if self.selected_kv_runtime is not None:
            self.selected_kv_runtime.fence_aborted_step()
        self.reclaim_aborted_step_leases()
        self._reap_deferred_leases()
        super().wait_for_previous_save()

    def wait_for_layer_load(self) -> None:
        lid = self.current_layer
        super().wait_for_layer_load()
        if self.selected_kv_runtime is not None and lid < self.num_layers:
            self.selected_kv_runtime.on_native_layer_ready(lid)

    def _flush_ready_removes(self) -> None:
        refs, held = self._cnt("_lease_refs"), self._cnt("_held_removes")
        ready = Counter({k: held.pop(k) for k in list(held) if k not in refs and k not in self._step_load_keys})
        while ready:
            ready -= Counter(batch := list(ready))
            try:
                if (rc := self.m_store.batch_remove_lease(batch)) != 0:
                    logger.error("AscendStore deferred lease cleanup returned %s; TTL will reclaim leases", rc)
            except Exception:
                logger.warning("AscendStore deferred lease cleanup raised; TTL will reclaim leases", exc_info=True)

    def _reap_deferred_leases(self, *, drain_all: bool = False, cap_inflight: bool = False) -> None:
        refs, held = self._cnt("_lease_refs"), self._cnt("_held_removes")
        while self._deferred_leases:
            ev, keys = self._deferred_leases[0]
            try:
                if not ev.query():
                    if not drain_all and not (cap_inflight and len(self._deferred_leases) >= MAX_DEFERRED_LEASE_STEPS):
                        break
                    ev.synchronize()
            except BaseException:
                if self.selected_kv_runtime is not None:
                    self.selected_kv_runtime.fail_handoff()
                raise
            refs -= Counter(self._deferred_leases.popleft()[1])
            held.update(keys)
        self._flush_ready_removes()

    def get_finished(self, finished_req_ids: set[str], meta: Any) -> tuple[set[str], set[str]]:
        cap_needed, refs, held = False, self._cnt("_lease_refs"), self._cnt("_held_removes")
        if (rt := self.selected_kv_runtime) is not None:
            ok, ddr_done = rt.take_step_fence()
            if not ok:
                # Transfer threads may have been stopped with unfinished queue items;
                # skip _drain_attention_transfers and defer removal until after base cleanup.
                held.update(self._step_load_keys)
                self._step_load_keys.clear()
            elif ddr_done is not None:
                try:
                    if not ddr_done.query():
                        if not self.async_scheduling:
                            ddr_done.synchronize()
                        elif self._step_load_keys:
                            if self.kv_recv_thread is not None:
                                self._drain_attention_transfers(full=True)
                            refs.update(keys := list(self._step_load_keys))
                            self._deferred_leases.append((ddr_done, keys))
                            self._step_load_keys.clear()
                            cap_needed = True
                except BaseException:
                    rt.fail_handoff()
                    raise
        self._reap_deferred_leases(cap_inflight=cap_needed)
        if refs and self._step_load_keys and (shared := self._step_load_keys & refs.keys()):
            try:
                if self.kv_recv_thread is not None:
                    self._drain_attention_transfers(full=True)
            except BaseException:
                if rt is not None:
                    rt.fail_handoff()
                raise
            self._step_load_keys.difference_update(shared)
            held.update(shared)
        res = super().get_finished(finished_req_ids, meta)
        self._flush_ready_removes()
        return res

    def shutdown(self, *, join_timeout: float = THREAD_JOIN_TIMEOUT_S) -> None:
        if getattr(self, "_closed", False):
            return
        msg = "KV transfer thread did not exit during SelectedKVPoolWorker shutdown"
        stop_transfer_threads(self, join_timeout, msg)
        if (rt := self.selected_kv_runtime) is not None:
            rt.close()
        self._cnt("_held_removes").update(self._step_load_keys)
        self._step_load_keys.clear()
        self._reap_deferred_leases(drain_all=True)
        self.selected_kv_runtime = None
        super().shutdown()
