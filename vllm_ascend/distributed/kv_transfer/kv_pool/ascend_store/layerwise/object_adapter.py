# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Common object I/O on the ordinary pool's existing layerwise queues.

The worker retains prefetch/alias-buffer gates. This adapter never bulk-loads
different physical layers into rotating HBM slots at the same time.
"""

import hashlib
from collections import defaultdict
from dataclasses import replace
from threading import Lock
from uuid import uuid4

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import require_aligned_batch_results
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.executor import PoolBackendExecutor
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_writer import (
    LayerwiseSenderExecutor,
    LayerwiseWriter,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.prefix_lookup import logical_cache_range
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.restore_planner import RestorePlanner
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.state_snapshot import (
    MAMBA_FORMAT_PREFIX,
    plan_mamba_restore,
)


class LayerwiseObjectAdapter:
    @classmethod
    def from_pool_worker(cls, worker):
        """Reuse registered tensors, topology, key builder and the sender."""
        from uuid import uuid4

        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_metadata import PoolSourceProfile
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_layout import describe_cache_bindings
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
            get_layerwise_kv_cache_specs,
        )
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_cache_layout import PoolCacheLayout

        specs = get_layerwise_kv_cache_specs(worker.kv_cache_config)
        bindings = describe_cache_bindings(worker.kv_cache_config, worker.kv_caches, owned_layers=set(worker.kv_caches))
        bindings = {name: parts for name, parts in bindings.items() if parts[0].group_id in worker.cacheable_group_ids}
        layout = PoolCacheLayout(
            cache_config=worker.kv_cache_config,
            bindings=bindings,
            specs=specs,
            tp_rank=worker.tp_rank,
            tp_size=worker.tp_size,
            dcp_rank=worker.dcp_rank,
            dcp_size=worker.dcp_size,
            pcp_rank=worker.pcp_rank,
            pcp_size=worker.pcp_size,
            total_num_kv_heads=worker.num_kv_head,
            pp_rank=worker.pp_rank,
            base_layers=worker.vllm_config.model_config.get_total_num_hidden_layers(),
            pool_head_rank=worker.head_or_tp_rank,
            pool_layer_offset=worker.layerwise_key_layer_offset,
            data_plane=worker.layerwise_data_plane,
            worker_id=f"pp{worker.pp_rank}/pcp{worker.pcp_rank}/tp{worker.tp_rank}",
        )
        profile = PoolSourceProfile(
            uuid4().hex,
            uuid4().hex,
            worker.layerwise_keys.identity,
            group_block_sizes=tuple(worker.grouped_block_size),
        )

        def layer_id(name):
            physical = worker._extract_physical_layer_index(name)
            return worker._global_to_local_layer.get(physical, physical - worker.layerwise_key_layer_offset)

        return cls(
            worker.m_store,
            layout,
            profile,
            layer_id,
            sender=worker.kv_send_thread,
            lease_ttl_ms=worker._extra_config.get("lease_ttl_ms", 600000),
        )

    def __init__(self, backend, layout, profile, layer_id, *, sender=None, lease_ttl_ms=600000):
        self.layout, self.profile = layout, profile
        self.layer_id = layer_id
        self.writer = (
            None
            if sender is None
            else LayerwiseWriter(backend, lease_ttl_ms=lease_ttl_ms, executor=LayerwiseSenderExecutor(sender))
        )
        self.reader = PoolBackendExecutor(backend, lease_ttl_ms=lease_ttl_ms)
        self.planner = RestorePlanner()
        self.writes = {}
        self.loads = {}
        self._save_layers = defaultdict(list)
        self._load_layers = defaultdict(list)
        self._submitted = set()
        self._save_failures = {}
        self._finished_loads = set()
        # Address-free descriptors of the last successful step. Speculative
        # rejection can change both hash availability and the accepted length;
        # rebuilding the preceding partial key from the next step is unsafe.
        self.snapshots = {}
        self._pending_snapshots = {}
        self._retired_snapshot_keys = set()
        self.aborted_writes = set()
        self._passive_snapshots = set()

    def prepare_save(self, request_id, request, *, group_ids=None, transient=False, step_id=None):
        if self.writer is None:
            raise RuntimeError("Read-only pool adapter cannot prepare a save")
        if request_id in self._pending_snapshots:
            raise ValueError("Previous ordinary pool write must drain before reusing its source")
        # Source completeness is independent of an HBM prefix hit: a newly
        # hashed page can contain both old and newly computed tokens.
        objects, layers, refs = self.layout.build_source_plan(request, self.profile)
        if not objects:
            return
        if transient:
            step_id = step_id or uuid4().hex
            # Scheduled draft/scratch pages have no accepted prefix identity.
            # Save them only for this request's next step. Prefix probes never
            # generate pool-step-v1 IDs, and committed hash-backed objects keep
            # their ordinary shared identities.
            aliases = {}
            rewritten = []
            for ref in refs:
                block_size = (
                    self.profile.group_block_sizes[ref.group_id]
                    if self.profile.group_block_sizes
                    else self.layout.cache_config.kv_cache_groups[ref.group_id].kv_cache_spec.block_size
                    * self.layout.dcp_size
                )
                accepted_end = len(request.group_hashes[ref.group_id]) * block_size
                if any(
                    logical_cache_range(part.cache, self.layout.specs[part.cache.layer])[1] > accepted_end
                    for part in ref.slices
                ):
                    old = ref.resolve(self.profile.key_identity)[0].key
                    identity = hashlib.sha256(f"{step_id}:{ref.object_id}".encode()).hexdigest()
                    ref = replace(ref, object_id=f"pool-step-v1:{identity}")
                    aliases[old] = ref.resolve(self.profile.key_identity)[0].key
                rewritten.append(ref)
            refs = tuple(rewritten)
            objects = {aliases.get(key, key): size for key, size in objects.items()}
            layers = {
                name: tuple(replace(part, key=aliases.get(part.key, part.key)) for part in parts)
                for name, parts in layers.items()
            }
        # Replicas need descriptors for the leader's objects too. They retain
        # no write ownership for those objects; the normal TP step barrier
        # establishes readiness before the following offload restore.
        owned = {
            ref.resolve(self.profile.key_identity)[0].key
            for ref in refs
            if group_ids is None or ref.group_id in group_ids
        }
        objects = {key: size for key, size in objects.items() if key in owned}
        layers = {name: tuple(part for part in parts if part.key in owned) for name, parts in layers.items()}
        layers = {name: parts for name, parts in layers.items() if parts}
        private = frozenset(
            ref.resolve(self.profile.key_identity)[0].key
            for ref in refs
            if ref.object_id.startswith("pool-step-v1:") and ref.resolve(self.profile.key_identity)[0].key in owned
        )
        snapshot = (request.token_end, refs, private)
        if not objects:
            self._pending_snapshots[request_id] = snapshot
            self._passive_snapshots.add(request_id)
            return
        shared = frozenset(
            part.key
            for ref in refs
            if not ref.object_id.startswith(("pool-pd-v1:", "pool-step-v1:"))
            for part in ref.resolve(self.profile.key_identity)
            if part.key in owned
        )
        handle = self.writer.prepare(objects, layers, shared_objects=shared)
        self.writes[request_id] = handle
        self._pending_snapshots[request_id] = snapshot
        for name in layers:
            self._save_layers[self.layer_id(name)].append((handle, name))

    def submit_save_layer(self, layer_id, ready_event, done):
        """Called by the ordinary save hook with its actual recorded event."""
        if layer_id in self._submitted:
            return
        self._submitted.add(layer_id)
        futures = []
        for handle, name in self._save_layers.get(layer_id, ()):
            if handle.cancelled.is_set():
                continue
            self.writer.submit_layer(handle, name, ready_event)
            futures.append(handle.copy_futures[-1])
            if handle.submitted == handle.layers.keys():
                self.writer.finish(handle)

        # Register only after all pieces have been submitted: an already-done
        # future must not signal a shared slot while another piece is pending.
        completion_lock = Lock()
        signalled = False

        def completed(_):
            nonlocal signalled
            with completion_lock:
                if signalled or not all(future.done() for future in futures):
                    return
                signalled = True
            # Signal source-slot release exactly once, after every native
            # copy returns, including failed copies. The worker separately
            # checks the error before consuming an offloaded slot.
            done()

        for future in futures:
            future.add_done_callback(completed)
        if not futures:
            done()

    def snapshot_sources(self, request_id, token_end):
        snapshot = self.snapshots.get(request_id)
        if snapshot is None or snapshot[0] < token_end:
            raise ValueError("Ordinary pool has no committed snapshot covering the accepted prefix")
        return [part for ref in snapshot[1] for part in ref.resolve(self.profile.key_identity)]

    def release_requests(self, request_ids):
        for request_id in request_ids:
            if request_id in self.writes or self.loads.get(request_id):
                raise RuntimeError("Ordinary pool request descriptors cannot retire before I/O drains")
            self._retire_snapshot(self.snapshots.pop(request_id, None))
        self._cleanup_retired_snapshots()

    def _retire_snapshot(self, snapshot):
        if snapshot is not None:
            self._retired_snapshot_keys.update(snapshot[2])

    def _cleanup_retired_snapshots(self):
        keys = list(self._retired_snapshot_keys)
        if not keys:
            return
        self.reader.backend.set_device()
        codes = require_aligned_batch_results("ordinary snapshot revoke", keys, self.reader.backend.batch_revoke(keys))
        self._retired_snapshot_keys.difference_update(key for key, code in zip(keys, codes, strict=True) if code == 0)
        if self._retired_snapshot_keys:
            raise RuntimeError(f"Ordinary pool snapshot cleanup failed: {codes}")

    def prepare_load(self, request_id, request, *, sources=None, layer_ids=None):
        """Plan all components, but retain separate execution per layer."""
        if request_id in self.loads:
            raise ValueError("Previous ordinary pool load must drain before replacing it")
        if sources is None:
            # Same-layout ordinary pool read; cross-topology callers pass the
            # source descriptors discovered by the shared prefix query.
            _, _, refs = self.layout.build_source_plan(request, self.profile)
            sources = [part for ref in refs for part in ref.resolve(self.profile.key_identity)]
        target_layout = self.layout
        if layer_ids is not None:
            target_layout = replace(
                target_layout,
                bindings={
                    name: parts for name, parts in self.layout.bindings.items() if self.layer_id(name) in layer_ids
                },
            )
        targets = target_layout.build_target_slices(request.block_ids, request.token_start, request.token_end)
        by_layer = defaultdict(list)
        for target in targets:
            by_layer[target.cache.layer].append(target)
        plans = defaultdict(list)
        for name, parts in by_layer.items():
            layer_sources = [source for source in sources if source.cache.layer == name]
            if parts[0].cache.format_id.startswith(MAMBA_FORMAT_PREFIX):
                layer_sources = [source for source in layer_sources if source.cache.token_end == request.token_end]
                plan = plan_mamba_restore(layer_sources, parts)
            else:
                plan = self.planner.build(layer_sources, parts)
            plans[self.layer_id(name)].extend(plan)
        pending = set(plans)
        self.loads[request_id] = pending
        if not pending:
            self._finished_loads.add(request_id)
        for layer, plan in plans.items():
            self._load_layers[layer].append((request_id, plan))

    def load_layer(self, layer_id, finished, failed=None):
        """Execute only after the ordinary receive thread's slot/event gates."""
        for request_id, plan in self._load_layers.get(layer_id, ()):
            pending = self.loads.get(request_id)
            if pending is None or layer_id not in pending:
                continue
            try:
                self.reader.restore(plan)
            except Exception as exc:
                # The executor drains native reads before returning. Do not
                # publish a successful request after partially restoring it.
                pending.clear()
                self.loads.pop(request_id, None)
                if failed is None:
                    raise
                failed(request_id, exc)
                continue
            pending.remove(layer_id)
            if not pending:
                self._finished_loads.add(request_id)
                finished(request_id)

    def cancel_saves(self):
        # A receive failure disables this step's saving in the ordinary pool.
        # Set cancellation before returning the receive-finished event, so a
        # later model hook cannot commit bytes derived from an invalid prefix.
        for request_id, handle in self.writes.items():
            self.aborted_writes.add(request_id)
            self.writer.cancel(handle)
        self.aborted_writes.update(self._passive_snapshots)
        self._passive_snapshots.clear()

    def check_save_errors(self):
        for request_id, handle in self.writes.items():
            if request_id in self.aborted_writes:
                continue
            for future in (*handle.copy_futures, handle.terminal):
                if future is not None and future.done():
                    try:
                        future.result()
                    except BaseException as exc:
                        self._save_failures[request_id] = exc
                        raise RuntimeError(f"Ordinary pool write failed: {request_id}") from exc

    def finish_step(self):
        """Drain before replacing any plan or releasing model source buffers."""
        errors = []
        committed = {request_id: self._pending_snapshots[request_id] for request_id in self._passive_snapshots}
        for request_id, handle in list(self.writes.items()):
            try:
                future = handle.terminal
                if future is None:
                    future = self.writer.cancel(handle)
                future.result()
                if not handle.cancelled.is_set():
                    committed[request_id] = self._pending_snapshots[request_id]
            except BaseException as exc:
                if request_id not in self.aborted_writes or not handle.finalized:
                    errors.append((request_id, exc))
        if any(self.loads.values()):
            raise RuntimeError("Ordinary pool layer loads must drain before replacing their plans")
        for request_id, snapshot in committed.items():
            self._retire_snapshot(self.snapshots.get(request_id))
            self.snapshots[request_id] = snapshot
        self.writes.clear()
        self._pending_snapshots.clear()
        self._passive_snapshots.clear()
        self.loads.clear()
        self._save_layers.clear()
        self._load_layers.clear()
        self._submitted.clear()
        self._save_failures.clear()
        self._finished_loads.clear()
        self.aborted_writes.clear()
        # Cleanup is deferred to the next metadata binding, after the worker's
        # TP barrier: another replica may still be reading the preceding step.
        return errors

    def close(self):
        if self.writer is not None:
            self.writer.close()
        self.reader.close()
        # Both native I/O owners drained. Pending logical plans cannot keep
        # temporary pool objects alive after worker shutdown.
        for snapshot in self.snapshots.values():
            self._retire_snapshot(snapshot)
        self.snapshots.clear()
        self._cleanup_retired_snapshots()
