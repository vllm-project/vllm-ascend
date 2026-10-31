# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPD scheduling for the independent layerwise-pool connector.

This class does not retain P HBM for a D acknowledgement. P workers drain
their own writes before returning the model output; request_finished checks
their committed manifests before exposing source objects to the proxy.
"""

import math
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.request import RequestStatus

from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_control import pool_control_request
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_metadata import (
    LayerwisePoolConnectorMetadata,
    PoolCommitTracker,
    PoolSaveRequest,
    PoolTransferMetadata,
    PoolWorkerHandshake,
    PoolWorkerMetadata,
    PoolWorkerReport,
)
from vllm_ascend.distributed.kv_transfer.utils.prefill_boundary import (
    needs_prefill_token_truncation,
    truncate_request_for_prefill,
    unique_group_specs,
)
from vllm_ascend.distributed.kv_transfer.utils.request_identity import pool_request_digest

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


@dataclass
class _SourceRequest:
    request: "Request"
    digest: str
    transfer_id: str
    attempt_id: str
    save: PoolSaveRequest | None = None
    tracker: PoolCommitTracker | None = None
    prefix_failed: bool = False


class LayerwisePoolScheduler:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        cache_config: "KVCacheConfig",
        *,
        control_request: Callable[..., dict[str, Any]] = pool_control_request,
    ):
        transfer_config = vllm_config.kv_transfer_config
        if transfer_config is None or transfer_config.is_kv_producer == transfer_config.is_kv_consumer:
            raise ValueError("LayerwisePoolConnector requires exactly one P/D role")
        self.producer = transfer_config.is_kv_producer
        self.cache_config = cache_config
        parallel = vllm_config.parallel_config
        self.expected_worker_ids = frozenset(
            f"pp{pp}/pcp{pcp}/tp{tp}"
            for pp in range(parallel.pipeline_parallel_size)
            for pcp in range(parallel.prefill_context_parallel_size)
            for tp in range(parallel.tensor_parallel_size)
        )
        self.expected_workers = len(self.expected_worker_ids)
        self.dcp_size = parallel.decode_context_parallel_size
        self.need_truncate = needs_prefill_token_truncation(
            vllm_config.model_config, [unique_group_specs(group) for group in cache_config.kv_cache_groups]
        )
        self._control_request = control_request
        self._workers: dict[str, PoolWorkerHandshake] = {}
        self._source: dict[str, _SourceRequest] = {}
        self._loads: dict[str, tuple[PoolTransferMetadata, tuple[list[int], ...]]] = {}
        self._receiving: dict[str, PoolTransferMetadata] = {}
        self._cancelled: set[str] = set()
        self.hash_block_size: int | None = None
        self._notifications = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pool-retention-release")
        self.prefix_group_ids = tuple(
            index
            for index, group in enumerate(cache_config.kv_cache_groups)
            if is_prefix_cacheable(group.kv_cache_spec)
        )
        specs = [spec for group in cache_config.kv_cache_groups for spec in unique_group_specs(group)]
        self._prefix_enabled = bool(self.prefix_group_ids) and all(
            not isinstance(spec, MambaSpec) or spec.mamba_cache_mode == "align" for spec in specs
        )
        self._prefix_granularity = (
            math.lcm(
                *(
                    cache_config.kv_cache_groups[index].kv_cache_spec.block_size * self.dcp_size
                    for index in self.prefix_group_ids
                )
            )
            if self.prefix_group_ids
            else 1
        )
        self._prefix_hits: dict[str, PoolTransferMetadata] = {}
        self._prefix_receiving: set[str] = set()
        self._prefix_queries = ThreadPoolExecutor(
            max_workers=min(8, self.expected_workers), thread_name_prefix="pool-prefix-query"
        )

    def set_xfer_handshake_metadata(self, metadata: Mapping[Any, Any]) -> None:
        workers = {}
        for value in metadata.values():
            if not isinstance(value, PoolWorkerHandshake):
                raise TypeError("Pool scheduler received another connector's worker handshake")
            if value.worker_id in workers:
                raise ValueError("Duplicate pool worker identity in handshake")
            workers[value.worker_id] = value
        if workers.keys() != self.expected_worker_ids:
            missing = sorted(self.expected_worker_ids - workers.keys())
            unexpected = sorted(workers.keys() - self.expected_worker_ids)
            raise ValueError(
                "Pool handshake does not cover every local PP/PCP/TP worker: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if self._source:
            raise RuntimeError("Cannot replace pool worker incarnations with active transfers")
        sizes = {worker.profile.group_block_sizes for worker in workers.values()}
        if any(sizes):
            if len(sizes) != 1 or () in sizes:
                raise ValueError("Source workers disagree on logical cache group block sizes")
            group_sizes = next(iter(sizes))
            if len(group_sizes) != len(self.cache_config.kv_cache_groups):
                raise ValueError("Source logical cache groups differ from the scheduler")
            self._prefix_granularity = (
                math.lcm(*(group_sizes[index] for index in self.prefix_group_ids)) if self.prefix_group_ids else 1
            )
        self._workers = workers

    def on_new_request(self, request: "Request") -> None:
        params = request.kv_transfer_params
        if not params:
            return
        if self.producer and params.get("do_remote_decode"):
            # Fingerprint BEFORE changing P's prompt boundary. MTP settings do
            # not participate: D's proposal is computed locally.
            digest = pool_request_digest(request)
            self._source[request.request_id] = _SourceRequest(request, digest, uuid4().hex, uuid4().hex)
            if self.need_truncate:
                truncate_request_for_prefill(request)

    def _incoming(self, request: "Request") -> PoolTransferMetadata | None:
        params = request.kv_transfer_params
        if self.producer or not params or not params.get("do_remote_prefill"):
            return None
        try:
            transfer = PoolTransferMetadata.from_wire(params.get("pool_transfer"))
            if transfer.request_digest != pool_request_digest(request):
                raise ValueError("Pool handoff does not match the original request inputs")
            expected_end = request.num_prompt_tokens - int(self.need_truncate and request.num_prompt_tokens > 1)
            if transfer.token_start != 0 or transfer.token_end != expected_end:
                raise ValueError("Pool handoff has an incompatible prompt/state boundary")
            return transfer
        except (ValueError, TypeError, KeyError) as exc:
            # A malformed/stale CPD handoff must never turn into an external
            # hit. Leave D to compute this request locally.
            logger.error("Pool handoff rejected: request=%s error=%s", request.request_id, exc)
            params["do_remote_prefill"] = False
            return None

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        if self.producer:
            self._prefix_hits.pop(request.request_id, None)
            transfer = self._lookup_prefix(request, num_computed_tokens)
            if transfer is None:
                return 0, False
            self._prefix_hits[request.request_id] = transfer
            return transfer.token_end - num_computed_tokens, True
        transfer = self._incoming(request)
        if transfer is None:
            return 0, False
        count = max(0, transfer.token_end - num_computed_tokens)
        return count, count > 0

    def _lookup_prefix(self, request, start):
        state = self._source.get(request.request_id)
        if (
            not self._prefix_enabled
            or state is None
            or state.prefix_failed
            or len(self._workers) != self.expected_workers
        ):
            return None
        step = self._prefix_granularity
        end = (request.num_prompt_tokens - 1) // step * step
        if end <= start or self.hash_block_size is None:
            return None
        attempt = uuid4().hex
        data = {
            "transfer_id": state.transfer_id,
            "attempt_id": attempt,
            "request_digest": state.digest,
            "token_start": start,
            "token_end": end,
            "group_hashes": [
                list(self._source_hashes(request, index)) for index in range(len(self.cache_config.kv_cache_groups))
            ],
        }

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.prefix_lookup import lookup_pool_prefix

        try:
            return lookup_pool_prefix(
                self._workers.values(), data, self.prefix_group_ids, step, self._prefix_queries, self._control_request
            )
        except Exception:
            logger.exception("Pool prefix lookup failed; computing locally: request=%s", request.request_id)
            return None

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int) -> None:
        if self.producer:
            transfer = self._prefix_hits.pop(request.request_id, None)
            if num_external_tokens:
                if transfer is None or num_external_tokens != transfer.token_end - transfer.token_start:
                    raise ValueError("P prefix allocation does not match its validated hit")
                self._loads[request.request_id] = (transfer, tuple(list(group) for group in blocks.get_block_ids()))
                self._prefix_receiving.add(request.request_id)
            return
        transfer = self._incoming(request)
        if transfer is None:
            if num_external_tokens:
                raise ValueError("Pool allocation has external tokens without a validated handoff")
            return
        request.kv_transfer_params["do_remote_prefill"] = False
        if num_external_tokens:
            # Full logical tables (including null padding) preserve positions
            # under CP and window eviction. Worker coverage is checked before I/O.
            tables = tuple(list(group) for group in blocks.get_block_ids())
            restore = replace(transfer, token_start=transfer.token_end - num_external_tokens)
            self._loads[request.request_id] = (restore, tables)
            self._receiving[request.request_id] = transfer
        else:
            self._release_later(transfer)

    def _source_hashes(self, request: "Request", group_id: int) -> tuple[str, ...]:
        if self.hash_block_size is None:
            raise RuntimeError("Pool scheduler must bind the local block pool before saving")
        spec = self.cache_config.kv_cache_groups[group_id].kv_cache_spec
        # The worker's private attempt keys also include the full request
        # digest and coordinates. For non-token state groups no prefix hashes
        # are assigned; their snapshot identity is the request boundary.
        group_size = (
            next(iter(self._workers.values())).profile.group_block_sizes[group_id]
            if self._workers and next(iter(self._workers.values())).profile.group_block_sizes
            else spec.block_size * self.dcp_size
        )
        if group_size % self.hash_block_size:
            return ()
        hashes = BlockHashListWithBlockSize(request.block_hashes, self.hash_block_size, group_size)
        return tuple(value.hex() for value in hashes)

    def build_connector_meta(self, output: "SchedulerOutput") -> LayerwisePoolConnectorMetadata:
        meta = LayerwisePoolConnectorMetadata()
        for req_id, (transfer, blocks) in self._loads.items():
            meta.loads[req_id], meta.local_block_ids[req_id] = transfer, blocks
            if req_id in self._prefix_receiving:
                meta.prefix_group_ids[req_id] = self.prefix_group_ids
        self._loads.clear()
        meta.cancelled, self._cancelled = self._cancelled, set()
        meta.released_sources = set(output.finished_req_ids)
        preempted = set(getattr(output, "preempted_req_ids", ()) or ())
        meta.released_sources.update(preempted)
        for req_id in preempted:
            state = self._source.get(req_id)
            if state is not None:
                # Drop active pins under pressure and restart from a new
                # attempt. Resident immutable objects can still be borrowed.
                state.attempt_id = uuid4().hex
                state.save = state.tracker = None
        for req_id in output.finished_req_ids:
            self._source.pop(req_id, None)
        for req_id, count in output.num_scheduled_tokens.items():
            state = self._source.get(req_id)
            if state is None or count <= 0:
                continue
            request = state.request
            end = min(request.num_computed_tokens + count, request.num_prompt_tokens)
            previous_end = state.save.token_end if state.save is not None else 0
            if end <= previous_end:
                continue  # Already stored before preemption or duplicate metadata.
            if len(self._workers) != self.expected_workers:
                raise RuntimeError("Pool source worker handshake is incomplete")
            block_state = output.kv_connector_block_state
            blocks = None if block_state is None else block_state.get_block_ids(req_id)
            if blocks is None:
                raise RuntimeError("Pool source requires authoritative full block tables")
            state.save = PoolSaveRequest(
                state.transfer_id,
                state.attempt_id,
                state.digest,
                0,
                end,
                tuple(tuple(group) for group in blocks),
                tuple(self._source_hashes(request, group) for group in range(len(blocks))),
                chunk_start=previous_end,
                is_final=end == request.num_prompt_tokens,
            )
            state.tracker = PoolCommitTracker(state.transfer_id, state.attempt_id, self.expected_worker_ids)
            meta.saves[req_id] = state.save
        return meta

    def request_finished(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        if request.request_id in self._prefix_receiving:
            self._cancelled.add(request.request_id)
        self._prefix_hits.pop(request.request_id, None)
        state = self._source.pop(request.request_id, None)
        if state is None:
            if request.request_id in self._receiving:
                # Scheduler keeps D blocks alive until reader completion.
                self._cancelled.add(request.request_id)
            elif (transfer := self._incoming(request)) is not None:
                # A queued D request may be cancelled before allocation or
                # the first cache lookup. No reader can own its objects yet.
                # Release the committed P retention instead of waiting for TTL.
                request.kv_transfer_params["do_remote_prefill"] = False
                self._release_later(transfer)
            return False, None
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED or state.save is None or state.tracker is None:
            return False, None
        try:
            # vLLM processes worker metadata after request_finished. Query the
            # completed manifests directly, rather than publishing before commit.
            for worker in self._workers.values():
                wire = self._control_request(
                    worker.control_endpoint,
                    {
                        "op": "manifest",
                        "incarnation": worker.profile.engine_incarnation,
                        "transfer_id": state.transfer_id,
                        "attempt_id": state.attempt_id,
                    },
                )
                report = PoolWorkerReport.from_wire(wire)
                if report.worker_id != worker.worker_id or not state.tracker.record(report):
                    raise ValueError("Pool worker returned a stale/mismatched manifest")
            if not state.tracker.committed:
                raise RuntimeError(state.tracker.failure or "Pool source commit is incomplete")
            transfer = PoolTransferMetadata(
                state.transfer_id,
                state.attempt_id,
                state.digest,
                state.save.token_start,
                state.save.token_end,
                state.attempt_id,
                tuple(worker.profile for worker in self._workers.values()),
                tuple(item for report in state.tracker.reports.values() for item in report.objects),
            )
            params = {"do_remote_prefill": True, "do_remote_decode": False, "pool_transfer": transfer.to_wire()}
            if request.output_token_ids:
                params["last_token_id"] = request.output_token_ids[-1]
            return False, params
        except Exception as exc:
            logger.error("Pool source commit not published: request=%s error=%s", request.request_id, exc)
            return False, {"do_remote_prefill": False, "do_remote_decode": False}

    def _release_later(self, transfer: PoolTransferMetadata) -> None:
        self._notifications.submit(self._release, transfer)

    def _release(self, transfer: PoolTransferMetadata) -> None:
        for profile in transfer.profiles:
            if not profile.control_endpoint:
                continue
            try:
                self._control_request(
                    profile.control_endpoint,
                    {
                        "op": "release",
                        "incarnation": profile.engine_incarnation,
                        "transfer_id": transfer.transfer_id,
                        "attempt_id": transfer.attempt_id,
                    },
                )
            except Exception:
                logger.exception(
                    "Pool retention release failed; source expiry must reclaim attempt=%s", transfer.attempt_id
                )

    def update_connector_output(self, output: "KVConnectorOutput") -> None:
        for request_id in output.failed_recving or ():
            if request_id in self._prefix_receiving:
                state = self._source.get(request_id)
                if state is not None:
                    # A readable object can still fail native GET. Recompute
                    # this request instead of repeatedly admitting the same
                    # failed prefix; other requests may still use the pool.
                    state.prefix_failed = True
        reports = output.kv_connector_worker_meta
        if isinstance(reports, PoolWorkerMetadata):
            active = {(state.transfer_id, state.attempt_id): state for state in self._source.values()}
            for report in reports.reports:
                state = active.get((report.transfer_id, report.attempt_id))
                if state is not None and state.tracker is not None:
                    state.tracker.record(report)
        # This callback receives the framework's ALL-worker aggregated result.
        # ACK releases pool retention only, never producer HBM blocks.
        for req_id in output.finished_recving or ():
            self._prefix_receiving.discard(req_id)
            transfer = self._receiving.pop(req_id, None)
            if transfer is not None:
                self._release_later(transfer)

    def shutdown(self) -> None:
        self._prefix_queries.shutdown(wait=True)
        self._notifications.shutdown(wait=True)
