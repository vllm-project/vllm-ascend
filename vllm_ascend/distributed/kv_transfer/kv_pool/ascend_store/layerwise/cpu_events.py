"""Publish layerwise offloads only after the complete prefix is readable."""

import time
from collections import OrderedDict

from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    block_hash_to_str,
    get_block_hashes,
)

MAX_PENDING_CPU_EVENT_REQUESTS = 256
PENDING_CPU_EVENT_TIMEOUT_SECONDS = 300


class LayerwiseCpuEvents:
    """Worker-owned pending metadata, polled by the existing event drain.

    Readability checks include every required group/rank and paired draft
    object. They never advertise an allocation or an incomplete layer PUT.
    No additional device synchronization or background thread is introduced.
    """

    def __init__(self, worker):
        self.worker = worker
        self.pending = OrderedDict()
        self.published = {}

    def observe(self, requests, finished_req_ids):
        for request in requests:
            if request.can_save is False or not request.token_ids or request.save_end_token <= 0:
                continue
            if request.req_id not in self.pending and len(self.pending) >= MAX_PENDING_CPU_EVENT_REQUESTS:
                continue
            self.pending[request.req_id] = (request, time.monotonic())
        for req_id in finished_req_ids:
            if req_id not in self.pending:
                self.published.pop(req_id, None)

    def _confirmed_tokens(self, request):
        worker = self.worker
        coordinator = worker.cache_coordinator
        if coordinator is None or worker.layerwise_keys is None:
            return 0
        draft_keys = getattr(worker, "_dspark_prefix_keys", None)
        draft_groups = frozenset(getattr(worker.kv_cache_config, "dspark_draft_only_group_ids", ()))

        def query_group_hits(group_id, hashes, lookup_mask):
            keys_by_block = []
            selected_hashes = []
            for index, block_hash in enumerate(hashes):
                if lookup_mask is not None and not (index < len(lookup_mask) and lookup_mask[index]):
                    continue
                hash_hex = block_hash_to_str(block_hash)
                if draft_keys is not None and group_id in draft_groups:
                    keys = draft_keys.make_hit_check_keys(hash_hex)
                else:
                    keys = worker.layerwise_keys.make_hit_check_keys(
                        group_id,
                        hash_hex,
                        worker.get_group_tp_size(group_id),
                    )
                    if draft_keys is not None:
                        keys.extend(draft_keys.make_hit_check_keys(hash_hex))
                keys_by_block.append(keys)
                selected_hashes.append(block_hash)
            keys = [key for block_keys in keys_by_block for key in block_keys]
            if not keys:
                return []
            readable = worker.m_store.batch_is_readable(keys)
            if len(readable) != len(keys) or any(type(value) is not bool for value in readable):
                raise RuntimeError("Invalid layerwise CPU-event readability response")
            hits = []
            offset = 0
            for block_hash, block_keys in zip(selected_hashes, keys_by_block, strict=True):
                if all(readable[offset : offset + len(block_keys)]):
                    hits.append(block_hash)
                offset += len(block_keys)
            return hits

        return coordinator.find_reachable_hit_tokens(
            request.block_hashes,
            min(request.save_end_token, len(request.token_ids)),
            query_group_hits,
            log_context=f"layerwise CPU events: req={request.req_id}",
        )

    def drain(self):
        worker = self.worker
        if worker.kv_send_thread is None or worker.kv_send_thread.save_disabled.is_set():
            self.pending.clear()
            self.published.clear()
            return []
        if not self.pending or not worker.cacheable_group_ids:
            return []
        events = []
        now = time.monotonic()
        # Use the first cacheable main-attention group, not a sliding-state
        # group's page size or the scheduler's cross-group LCM.
        group_id = min(worker.cacheable_group_ids)
        block_size = worker.grouped_block_size[group_id]
        for req_id, (request, created) in list(self.pending.items()):
            if now - created > PENDING_CPU_EVENT_TIMEOUT_SECONDS:
                self.pending.pop(req_id)
                self.published.pop(req_id, None)
                continue
            try:
                confirmed = self._confirmed_tokens(request)
            except Exception:
                logger.exception("Layerwise CPU-event confirmation failed for %s", req_id)
                continue
            hashes = get_block_hashes(request.block_hashes, block_size, worker.hash_block_size)
            first = self.published.get(req_id, 0) // block_size
            end = min(confirmed // block_size, len(hashes))
            for index in range(first, end):
                start_token = index * block_size
                events.append(
                    BlockStored(
                        block_hashes=[maybe_convert_block_hash(hashes[index])],
                        parent_block_hash=maybe_convert_block_hash(hashes[index - 1]) if index else None,
                        token_ids=request.token_ids[start_token : start_token + block_size],
                        block_size=block_size,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                    )
                )
            if end > first:
                self.published[req_id] = end * block_size
                logger.info(
                    "Layerwise CPU events request=%s confirmed_tokens=%d blocks=%d grain=%d",
                    req_id,
                    confirmed,
                    end - first,
                    block_size,
                )
            if confirmed >= request.save_end_token:
                self.pending.pop(req_id)
                self.published.pop(req_id, None)
        return events
