# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PD completion and compact tails over the common pool executor."""

import time

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_p2p.pd_pool.metadata import HANDOFF_DEADLINE_MARGIN_S
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import block_hash_to_str
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_io import MemcacheStoreLayoutIO

_HANDOFF_RELEASE_GRACE_S = 2.0
_HANDOFF_LEASE_GRACE_S = 30.0


class PoolPDWorker:
    def __init__(self, pool):
        self.pool = pool
        self.transfer = pool.shared_pool
        if self.transfer is None:
            raise ValueError("Pool PD requires the shared Store format")
        self.tails = []
        self.handoff_requests = []
        self.handoff_holds = []
        self.results = KVConnectorTransferResults()

    def _plans(self, request, *, tail_only, writing):
        plans = []
        # Async receive reserves only the remotely computed prompt prefix.
        # The final prompt token is recomputed locally; if it is alone in a
        # new block, that block has not been allocated at receive time.
        receive_end = max(0, request.prompt_tokens - 1)
        cache_end = request.prompt_tokens if request.cache_tokens is None else request.cache_tokens
        if not writing and request.local_tokens >= receive_end:
            return ()
        for gid, layout in self.transfer.request_states.items():
            if writing and layout.writer_rank != 0:
                continue
            blocks = request.block_ids[gid]
            if len(blocks) != 1:
                raise ValueError("PD circular state requires exactly one block per request")
            plans.extend(layout.snapshot(blocks[0], request.prompt_digest, cache_end))
        for gid, layout in self.transfer.layouts.items():
            if writing and layout.replicated and layout.writer_rank != 0:
                continue
            complete, remainder = divmod(cache_end, layout.chunk_size)
            blocks = request.block_ids[gid]
            if layout.state:
                index = (cache_end - 1) // layout.block_size
                if index >= len(blocks) or blocks[index] <= 0:
                    raise ValueError("PD recurrent state has no allocated checkpoint")
                # PD snapshots use the exact stateful P boundary, including
                # non-aligned request tails. They are not ordinary checkpoints.
                plans.extend(layout.objects(0, blocks[index], f"pd_state_{cache_end}_{request.prompt_digest}"))
                continue
            first = request.local_tokens // layout.chunk_size
            unhashed = None
            if not writing and request.unhashed_block_ids is not None:
                # The scheduler reports the deepest attention hit. A different
                # group may have missed part of that prefix, so decide which
                # blocks to restore from each group's actual allocation.
                unhashed = set(request.unhashed_block_ids[gid])
                first = 0
            if not tail_only:
                for index in range(first, complete):
                    block_id = blocks[index]
                    if block_id <= 0:
                        raise ValueError("PD cannot restore a null full-attention block")
                    if unhashed is not None and block_id not in unhashed:
                        continue
                    digest = block_hash_to_str(
                        request.block_hashes[(index + 1) * layout.chunk_size // layout.hash_block_size - 1]
                    )
                    plans.extend(layout.objects(index * layout.chunk_size, block_id, digest))
            needs_tail = writing or receive_end > complete * layout.chunk_size
            if remainder and needs_tail and request.local_tokens < request.prompt_tokens:
                if complete >= len(blocks) or blocks[complete] <= 0:
                    raise ValueError("PD tail has no allocated block")
                if unhashed is not None and blocks[complete] not in unhashed:
                    raise ValueError("PD tail requires a private destination block")
                plans.extend(
                    layout.partial_objects(
                        complete * layout.chunk_size, blocks[complete], request.prompt_digest, remainder
                    )
                )
        return tuple(plans)

    def _reap_handoffs(self, *, all_holds=False):
        remaining = []
        for expiry, io, session in self.handoff_holds:
            if all_holds or time.monotonic() >= expiry:
                io.abort(session)
            else:
                remaining.append((expiry, io, session))
        self.handoff_holds = remaining

    def _protect_handoffs(self):
        for request in self.handoff_requests:
            remaining = request.handoff_deadline - time.time()
            if remaining <= HANDOFF_DEADLINE_MARGIN_S:
                raise RuntimeError("Pool PD handoff protection expired before acquisition")
            # Use an independent local owner for each handoff. The SDK tracks
            # operation leases per client/key; no stale GVA is reused. Native
            # TTL exceeds the bounded hold, including clock-skew allowance.
            io = MemcacheStoreLayoutIO(
                self.transfer.io.backend, lease_ttl_ms=int((remaining + _HANDOFF_LEASE_GRACE_S) * 1000)
            )
            plans = self._plans(request, tail_only=False, writing=True)
            session = io.begin(plans, writing=False)
            self.handoff_holds.append((time.monotonic() + remaining + _HANDOFF_RELEASE_GRACE_S, io, session))
            if time.time() + HANDOFF_DEADLINE_MARGIN_S >= request.handoff_deadline:
                raise RuntimeError("Pool PD handoff protection expired during acquisition")
            logger.info(
                "Pool PD HOLD req=%s objects=%d deadline=%.3f", request.request_id, len(plans), request.handoff_deadline
            )

    def prepare(self, metadata):
        self._reap_handoffs()
        self.handoff_requests = [r for r in metadata.tails if r.handoff_deadline is not None]
        if self.tails:
            raise RuntimeError("Previous PD tail saves have not drained")
        self.transfer.prepare_step(metadata.pool)
        seen = set()
        try:
            for request in metadata.tails:
                plans = tuple(
                    plan for plan in self._plans(request, tail_only=True, writing=True) if plan.key not in seen
                )
                seen.update(plan.key for plan in plans)
                self.tails.append(self.transfer.io.begin(plans, writing=True))
        except Exception:
            try:
                self.abort()
            finally:
                self.transfer.abort_step()
            raise

    def start_load(self, metadata):
        self.transfer.start_load(metadata.pool)
        for request in metadata.receives:
            session = None
            try:
                if (
                    request.handoff_deadline is not None
                    and time.time() + HANDOFF_DEADLINE_MARGIN_S >= request.handoff_deadline
                ):
                    raise RuntimeError("Pool PD handoff protection expired before receive")
                session = self.transfer.io.begin(self._plans(request, tail_only=False, writing=False), writing=False)
                if (
                    request.handoff_deadline is not None
                    and time.time() + HANDOFF_DEADLINE_MARGIN_S >= request.handoff_deadline
                ):
                    raise RuntimeError("Pool PD handoff protection expired during receive acquisition")
                self.transfer.io.copy(session)
                self.transfer.io.finish(session)
                logger.info(
                    "Pool PD GET req=%s cache_tokens=%s objects=%d bytes=%d",
                    request.request_id,
                    request.cache_tokens,
                    len(session.plans),
                    sum(plan.size for plan in session.plans),
                )
            except Exception:
                # Eviction between P completion and D acquisition is a miss,
                # not permission to consume partly restored cache.
                logger.exception("Pool PD receive failed for %s", request.request_id)
                self.results.failed_recving.add(request.request_id)
            finally:
                if session is not None:
                    self.transfer.io.abort(session)
            self.results.finished_recving.add(request.request_id)

    def save_layer(self, name):
        self.transfer.save_layer(name, self.tails)

    def finish(self):
        try:
            self.transfer.finish_step()
            if self.tails and not self.transfer.layerwise:
                event = torch.npu.Event()
                event.record()
                event.synchronize()
            for session in self.tails:
                if not self.transfer.layerwise:
                    self.transfer.io.copy(session)
                self.transfer.io.finish(session)
            self._protect_handoffs()
        finally:
            self.abort()

    def abort(self):
        errors = []
        try:
            self.transfer.drain_saves()
        except Exception as error:
            errors.append(error)
        for session in self.tails:
            try:
                self.transfer.io.abort(session)
            except Exception as error:
                errors.append(error)
        self.tails.clear()
        self.handoff_requests.clear()
        if errors:
            raise RuntimeError("PD tail lease cleanup failed") from errors[0]

    def take_results(self):
        result, self.results = self.results, KVConnectorTransferResults()
        return result

    def shutdown(self):
        try:
            self.abort()
            self._reap_handoffs(all_holds=True)
        finally:
            self.pool.shutdown()
