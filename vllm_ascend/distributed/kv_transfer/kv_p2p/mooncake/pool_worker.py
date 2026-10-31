# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Independent worker execution for layerwise-write, bulk-read pool PD."""

import threading
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.logger import logger
from vllm.v1.kv_cache_interface import (
    MambaSpec,
)

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_control import PoolControlServer, PoolManifestRegistry
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_metadata import (
    LayerwisePoolConnectorMetadata,
    PoolObjectRef,
    PoolSaveRequest,
    PoolSourceProfile,
    PoolTransferMetadata,
    PoolWorkerHandshake,
    PoolWorkerReport,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_stats import PoolPDStats
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import collect_configured_register_regions
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.executor import PoolBackendExecutor
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.bulk_reader import BulkReader
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_layout import (
    describe_cache_bindings,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    get_layerwise_kv_cache_specs,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_writer import (
    LayerwiseWriteHandle,
    LayerwiseWriter,
    LayerWriteRange,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.restore_planner import RestorePlanner
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.state_snapshot import (
    MAMBA_FORMAT_PREFIX,
    plan_mamba_restore,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend


from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_cache_layout import PoolCacheLayout


@dataclass
class _PrefillProgress:
    transfer_id: str
    attempt_id: str
    request_digest: str
    profile: PoolSourceProfile
    token_end: int = 0
    objects: dict[str, PoolObjectRef] = field(default_factory=dict)
    error: str = ""


class LayerwisePoolWorker:
    def __init__(
        self,
        backend: "Backend",
        cache_config: "KVCacheConfig",
        *,
        worker_id: str,
        tp_rank: int,
        tp_size: int,
        dcp_rank: int,
        dcp_size: int,
        lease_ttl_ms: int,
        pcp_size: int = 1,
        pcp_rank: int = 0,
        total_num_kv_heads: int | None = None,
        pp_rank: int = 0,
        base_layers: int = 0,
        pool_head_rank: int | None = None,
        pool_layer_offset: int = 0,
    ):
        if not 0 <= tp_rank < tp_size or not 0 <= dcp_rank < dcp_size or not 0 <= pcp_rank < pcp_size:
            raise ValueError("Invalid pool worker parallel coordinates")
        if dcp_size > 1 and (dcp_size % pcp_size or tp_size % (dcp_size // pcp_size)):
            raise ValueError("Pool worker DCP/PCP/TP topology is not supported by the V2 mapping")
        self.backend = backend
        self.cache_config = cache_config
        self.worker_id = worker_id
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.dcp_rank, self.dcp_size = dcp_rank, dcp_size
        self.pcp_size = pcp_size
        self.pcp_rank = pcp_rank
        self.total_num_kv_heads = total_num_kv_heads
        self.pp_rank = pp_rank
        self.base_layers = base_layers
        self.pool_head_rank = tp_rank if pool_head_rank is None else pool_head_rank
        self.pool_layer_offset = pool_layer_offset
        self.specs = get_layerwise_kv_cache_specs(cache_config)
        self.bindings = {}
        self._caches_registered = False
        if pcp_size > 1 and tp_size > 1 and any(isinstance(spec, MambaSpec) for spec in self.specs.values()):
            raise ValueError("Pool Mamba requires PCP=1 or TP=1, following Mooncake V2")
        self.executor = PoolBackendExecutor(backend, lease_ttl_ms=lease_ttl_ms)
        self.reader = BulkReader(self.executor)
        self.writer = LayerwiseWriter(backend, lease_ttl_ms=lease_ttl_ms)
        self.planner = RestorePlanner()
        self._writes: dict[str, tuple[LayerwiseWriteHandle, str, str, tuple[PoolObjectRef, ...]]] = {}
        self._reports: list[PoolWorkerReport] = []
        self._results = KVConnectorTransferResults()
        self._manifest_registry: PoolManifestRegistry | None = None
        self._control_server: PoolControlServer | None = None
        self._retained: dict[tuple[str, str], list[str]] = {}
        self._retention_lock = threading.RLock()
        self._stats = PoolPDStats(data={})
        self._prefill_progress: dict[str, _PrefillProgress] = {}
        self._save_requests: dict[str, PoolSaveRequest] = {}

    def enable_manifest_control(
        self, profile: PoolSourceProfile, host: str, *, retention_seconds: float
    ) -> PoolWorkerHandshake:
        if self._control_server is not None:
            raise RuntimeError("Pool manifest control was already started")
        registry = PoolManifestRegistry(
            profile.engine_incarnation, self._release_report, retention_seconds=retention_seconds
        )

        def control(request):
            if request.get("op") == "prefix":
                if request.get("incarnation") != profile.engine_incarnation:
                    raise ValueError("Stale prefix worker incarnation")
                return self.probe_prefix(request)
            return registry.handle(request)

        server = PoolControlServer(host, control, maintenance=registry.expire)
        self._manifest_registry, self._control_server = registry, server
        profile = replace(profile, control_endpoint=server.endpoint)
        self._profile = profile
        return PoolWorkerHandshake(self.worker_id, profile, server.endpoint)

    def probe_prefix(self, query):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.prefix_lookup import probe_pool_prefix

        return probe_pool_prefix(PoolCacheLayout.from_worker(self), self.backend, self._profile, self.worker_id, query)

    def _retain_report(self, report: PoolWorkerReport) -> None:
        if self._control_server is not None:
            self._control_server.check_health()
        keys = list(
            dict.fromkeys(item.key for ref in report.objects for item in ref.resolve(self._profile.key_identity))
        )
        if not keys:
            return
        attempt = (report.transfer_id, report.attempt_id)
        # Native calls can acquire some sessions before raising or returning a
        # malformed vector. Keep ownership until rollback succeeds; a failed
        # report's expiry/close path will retry unresolved cleanup.
        with self._retention_lock:
            previous = self._retained.setdefault(attempt, [])
            added = [key for key in keys if key not in previous]
            # Use the same process-key reference counts and renewal as reads.
            # Earlier chunks must remain protected throughout a long prefill.
            try:
                if added:
                    self.executor.acquire_read_protection(added)
            except Exception as exc:
                if not previous:
                    self._retained.pop(attempt, None)
                raise RuntimeError(f"Pool source retention failed: {exc}") from exc
            previous.extend(added)
            self.executor.check_read_protection()

    def _remove_retention(self, keys: list[str]) -> None:
        if not keys:
            return
        self.backend.set_device()
        self.executor.release_read_protection(keys)

    def _release_report(self, report: PoolWorkerReport) -> None:
        self._release_attempt(report.transfer_id, report.attempt_id)

    def _release_attempt(self, transfer_id: str, attempt_id: str) -> None:
        with self._retention_lock:
            key = (transfer_id, attempt_id)
            if key not in self._retained:
                self.executor.retry_pending_releases()
                return
            try:
                self._remove_retention(self._retained[key])
            finally:
                # Logical refs are consumed once. The executor owns uncertain
                # native releases and retries them without decrementing twice.
                self._retained[key] = []
            self.executor.retry_pending_releases()
            self._retained.pop(key, None)

    def register_kv_caches(self, caches: dict[str, Any], *, owned_layers: set[str] | None = None) -> None:
        if self._caches_registered:
            raise RuntimeError("Pool worker caches were already registered")
        bindings = describe_cache_bindings(self.cache_config, caches, owned_layers=owned_layers)
        regions = collect_configured_register_regions(self.cache_config, caches)
        self.backend.register_buffer(regions.ptrs, regions.lengths)
        self.bindings = bindings
        self._caches_registered = True

    def build_target_slices(self, block_ids, token_start, token_end, *, group_ids=None):
        layout = PoolCacheLayout.from_worker(self)
        if group_ids is not None:
            layout = replace(
                layout,
                bindings={name: parts for name, parts in self.bindings.items() if parts[0].group_id in group_ids},
            )
        return layout.build_target_slices(block_ids, token_start, token_end)

    def build_source_plan(self, request, profile, *, allow_empty_chunk=False):
        return PoolCacheLayout.from_worker(self).build_source_plan(
            request, profile, allow_empty_chunk=allow_empty_chunk
        )

    def prepare_save_request(self, request_id: str, request: PoolSaveRequest, profile: PoolSourceProfile) -> None:
        if self._manifest_registry is not None and profile != self._profile:
            raise ValueError("Pool save profile does not match this worker incarnation")
        progress = self._prefill_progress.get(request_id)
        identity = (request.transfer_id, request.attempt_id, request.request_digest)
        if progress is None:
            if request.chunk_start:
                raise ValueError("Pool prefill is missing its preceding committed chunks")
            progress = self._prefill_progress[request_id] = _PrefillProgress(*identity, profile)
        elif (progress.transfer_id, progress.attempt_id, progress.request_digest) != identity:
            raise ValueError("Pool prefill cannot replace an active request generation")
        if progress.error:
            return  # Preserve the first failure; no incomplete final handoff.
        if request.chunk_start != progress.token_end:
            raise ValueError("Pool prefill chunk is out of order or has a coverage gap")
        objects, layers, refs = self.build_source_plan(request, profile, allow_empty_chunk=True)
        if not objects:
            # KVPP can assign no persistent target layers to a rank. Its
            # explicit successful empty report still participates in the exact
            # worker-set commit barrier; it must not allocate a dummy object.
            self._complete_chunk(request_id, request, ())
            return
        self._save_requests[request_id] = request
        shared = frozenset(
            item.key
            for ref in refs
            if not ref.object_id.startswith("pool-pd-v1:")
            for item in ref.resolve(profile.key_identity)
        )
        self.prepare_write(
            request_id, request.transfer_id, request.attempt_id, objects, layers, refs, shared_objects=shared
        )

    def release_source_requests(self, request_ids) -> None:
        for request_id in request_ids:
            progress = self._prefill_progress.get(request_id)
            if progress is not None:
                if request_id in self._writes:
                    raise RuntimeError("Pool source cannot retire before its layer writes drain")
                self._release_attempt(progress.transfer_id, progress.attempt_id)
                del self._prefill_progress[request_id]

    def start_load_kv(self, metadata: LayerwisePoolConnectorMetadata) -> None:
        self.release_source_requests(metadata.released_sources)
        for request_id in metadata.cancelled:
            self.reader.cancel(request_id)
        for request_id, transfer in metadata.loads.items():
            if request_id in metadata.cancelled:
                if self.reader.active_attempt(request_id) is None:
                    # Abort can arrive after scheduler allocation but before
                    # this worker starts the restore. Complete without I/O;
                    # an already running attempt must report only after drain.
                    self._results.finished_recving.add(request_id)
                    self._stats.add("restore_cancelled")
                continue
            self.restore_request(
                request_id,
                transfer,
                metadata.local_block_ids[request_id],
                group_ids=metadata.prefix_group_ids.get(request_id),
            )

    def restore_request(
        self,
        request_id: str,
        transfer: PoolTransferMetadata,
        block_ids: tuple[list[int], ...],
        *,
        group_ids: tuple[int, ...] | None = None,
    ) -> None:
        active = self.reader.active_attempt(request_id)
        if active is not None:
            if active == transfer.attempt_id:
                return
            raise RuntimeError("Cannot replace a pool read attempt before native I/O drains")
        blocks = frozenset(block for group in block_ids for block in group)
        try:
            targets = self.build_target_slices(block_ids, transfer.token_start, transfer.token_end, group_ids=group_ids)
            required = any(group_ids is None or parts[0].group_id in group_ids for parts in self.bindings.values())
            if transfer.token_end > transfer.token_start and required and not targets:
                raise ValueError("Pool restore has no destination cache ranges")
            sources = transfer.sources()
            state_sources = [
                item
                for item in sources
                if item.cache.format_id.startswith(MAMBA_FORMAT_PREFIX) and item.cache.token_end == transfer.token_end
            ]
            state_targets = [item for item in targets if item.cache.format_id.startswith(MAMBA_FORMAT_PREFIX)]
            token_sources = [item for item in sources if not item.cache.format_id.startswith(MAMBA_FORMAT_PREFIX)]
            token_targets = [item for item in targets if not item.cache.format_id.startswith(MAMBA_FORMAT_PREFIX)]
            plan = self.planner.build(token_sources, token_targets) + plan_mamba_restore(state_sources, state_targets)
            self.reader.submit(request_id, transfer.attempt_id, plan, blocks)
        except Exception as exc:
            # No native I/O has started before submit. Framework sees a failed
            # receive rather than continuing with partially initialized KV.
            self._results.finished_recving.add(request_id)
            self._results.failed_recving.add(request_id)
            self._stats.add("restore_failed")
            logger.error(
                "Pool restore failed before submission: request=%s attempt=%s error=%s",
                request_id,
                transfer.attempt_id,
                exc,
            )

    def prepare_write(
        self,
        request_id: str,
        transfer_id: str,
        attempt_id: str,
        objects: dict[str, int],
        layers: dict[str, tuple[LayerWriteRange, ...]],
        refs: tuple[PoolObjectRef, ...],
        *,
        shared_objects: frozenset[str] = frozenset(),
    ) -> None:
        if request_id in self._writes:
            raise ValueError("Previous pool write must complete before replacing its source buffers")
        handle = (
            self.writer.prepare(objects, layers, shared_objects=shared_objects)
            if shared_objects
            else self.writer.prepare(objects, layers)
        )
        self._writes[request_id] = (handle, transfer_id, attempt_id, refs)

    def save_kv_layer(self, layer_name: str, ready_event: Any | None = None) -> None:
        handles = [
            entry[0]
            for entry in self._writes.values()
            if layer_name in entry[0].layers and layer_name not in entry[0].submitted
        ]
        if not handles:
            return
        if ready_event is None:
            ready_event = torch.npu.Event()
            ready_event.record()
        for handle in handles:
            self.writer.submit_layer(handle, layer_name, ready_event)

    def wait_for_save(self) -> None:
        for request_id, (handle, transfer_id, attempt_id, refs) in list(self._writes.items()):
            try:
                self.writer.finish(handle).result()
                request = self._save_requests.pop(request_id, None)
                published = self._complete_chunk(request_id, request, refs, transfer_id, attempt_id)
            except Exception as exc:
                # Incomplete layer submission still requires native rollback.
                try:
                    self.writer.cancel(handle).result()
                except Exception as cleanup_exc:
                    logger.error("Pool write cleanup reported failure: request=%s error=%s", request_id, cleanup_exc)
                progress = self._prefill_progress.get(request_id)
                if progress is not None:
                    progress.error = progress.error or str(exc)
                report = PoolWorkerReport(
                    transfer_id,
                    attempt_id,
                    self.worker_id,
                    False,
                    error=progress.error if progress is not None else str(exc),
                )
                if self._manifest_registry is not None:
                    self._manifest_registry.publish(report)
                self._reports.append(report)
                self._save_requests.pop(request_id, None)
                self._stats.add("save_failed")
            else:
                if published:
                    self._stats.add("save_committed")
                self._stats.add(
                    "save_bytes", sum(size for key, size in handle.objects.items() if key not in handle.borrowed)
                )
            del self._writes[request_id]

    def _complete_chunk(self, request_id, request, refs, transfer_id=None, attempt_id=None):
        """Expose committed chunks to the pool; hand off only the final union."""
        progress = self._prefill_progress.get(request_id) if request is not None else None
        if progress is not None:
            for ref in refs:
                key = ref.resolve(progress.profile.key_identity)[0].key
                progress.objects[key] = ref
            transfer_id, attempt_id = progress.transfer_id, progress.attempt_id
        report = PoolWorkerReport(transfer_id, attempt_id, self.worker_id, True, tuple(refs))
        retained_at = self._manifest_registry.clock() if self._manifest_registry is not None else None
        if self._manifest_registry is not None:
            self._retain_report(report)
        if progress is not None:
            progress.token_end = request.token_end
            if not request.is_final:
                return False
            # Recurrent/circular state is an exact-boundary snapshot. Earlier
            # checkpoints remain prefix objects, but cannot initialize final D.
            merged = []
            for ref in progress.objects.values():
                slices = tuple(
                    part
                    for part in ref.slices
                    if not (
                        part.cache.format_id.startswith(MAMBA_FORMAT_PREFIX)
                        or '"family":"circular-v2"' in part.cache.format_id
                    )
                    or part.cache.token_end == request.token_end
                )
                if slices:
                    merged.append(replace(ref, slices=slices))
            report = replace(report, objects=tuple(merged))
            # Validate the accumulated history before making a PD handoff
            # visible. This builds ranges only; it never copies back into P.
            sources = [part for ref in report.objects for part in ref.resolve(progress.profile.key_identity)]
            targets = self.build_target_slices(request.block_ids, 0, request.token_end)
            self.planner.build(
                [part for part in sources if not part.cache.format_id.startswith(MAMBA_FORMAT_PREFIX)],
                [part for part in targets if not part.cache.format_id.startswith(MAMBA_FORMAT_PREFIX)],
            )
            plan_mamba_restore(
                [part for part in sources if part.cache.format_id.startswith(MAMBA_FORMAT_PREFIX)],
                [part for part in targets if part.cache.format_id.startswith(MAMBA_FORMAT_PREFIX)],
            )
            keep = {part.key for part in sources}
            with self._retention_lock:
                attempt = (transfer_id, attempt_id)
                old = self._retained.get(attempt, [])
                obsolete = [key for key in old if key not in keep]
                try:
                    self._remove_retention(obsolete)
                finally:
                    if attempt in self._retained:
                        self._retained[attempt] = [key for key in old if key in keep]
                self.executor.retry_pending_releases()
        if self._manifest_registry is not None:
            self.executor.check_read_protection()
            self._manifest_registry.publish(report, retention_started_at=retained_at)
        self._reports.append(report)
        if progress is not None:
            del self._prefill_progress[request_id]
        return True

    def take_write_reports(self) -> list[PoolWorkerReport]:
        reports, self._reports = self._reports, []
        return reports

    def get_transfer_results(self) -> KVConnectorTransferResults:
        for result in self.reader.poll_results():
            self._results.finished_recving.add(result.request_id)
            if result.error:
                logger.error(
                    "Pool restore failed: worker=%s request=%s attempt=%s cancelled=%s error=%s",
                    self.worker_id,
                    result.request_id,
                    result.attempt_id,
                    result.cancelled,
                    result.error,
                )
            if result.success:
                self._stats.add("restore_completed")
                self._stats.add("restore_bytes", result.copied_bytes)
            elif result.cancelled:
                self._stats.add("restore_cancelled")
            else:
                self._stats.add("restore_failed")
            if not result.success:
                self._results.failed_recving.add(result.request_id)
        results, self._results = self._results, KVConnectorTransferResults()
        return results

    def get_kv_connector_stats(self) -> PoolPDStats | None:
        if self._stats.is_empty():
            return None
        stats, self._stats = self._stats, PoolPDStats(data={})
        return stats

    def get_block_ids_with_load_errors(self) -> set[int]:
        # Fail by request on every layout. The full destination table can
        # include shared local prefix blocks that were never written by this
        # restore. Invalidating those IDs would affect unrelated requests.
        # failed_recving is aggregated only once every worker has drained.
        return set()

    def shutdown(self) -> None:
        # If joining control callbacks is interrupted, leave the backend open:
        # closing it under an in-flight native operation is unsafe.
        if self._control_server is not None:
            self._control_server.close()
        errors = []
        for close in (self.writer.close, self.reader.close):
            try:
                close()
            except Exception as exc:
                errors.append(exc)
        if self._manifest_registry is not None:
            try:
                self._manifest_registry.close()
            except Exception as exc:
                errors.append(exc)
        for progress in list(getattr(self, "_prefill_progress", {}).values()):
            try:
                self._release_attempt(progress.transfer_id, progress.attempt_id)
            except Exception as exc:
                errors.append(exc)
        try:
            self.executor.close()
        except Exception as exc:
            errors.append(exc)
        if errors:
            # Failed cleanup still owns native objects/sessions. Keep their
            # client open so shutdown can be retried; do not destroy the only
            # handle capable of releasing them.
            raise RuntimeError(f"Pool worker shutdown has {len(errors)} unresolved cleanup failures") from errors[0]
        self._writes.clear()
        getattr(self, "_prefill_progress", {}).clear()
        getattr(self, "_save_requests", {}).clear()
        self.backend.close()
