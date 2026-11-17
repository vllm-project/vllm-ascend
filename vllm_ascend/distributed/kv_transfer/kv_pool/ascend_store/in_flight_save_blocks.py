from vllm.logger import logger
from vllm.v1.core.block_pool import BlockPool

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ReqMeta


class InFlightSaveBlockTracker:
    """Keep bulk-save source blocks alive until every worker reports completion."""

    def __init__(self, expected_worker_count: int) -> None:
        self.expected_worker_count = expected_worker_count
        self.next_event_id = 0
        self.sending_blocks: dict[int, list[int]] = {}
        self.sending_events: dict[int, int] = {}

    def retain(
        self,
        req_meta: ReqMeta,
        speculative_blocks_by_group: dict[int, int],
        block_pool: BlockPool | None,
        *,
        retain_all: bool = False,
    ) -> None:
        if not req_meta.can_save:
            return
        if req_meta.event_id is not None:
            # Finish-time handoffs are pinned before request teardown and then
            # emitted on the next scheduler step. Never take a second pin.
            assert req_meta.event_id in self.sending_events
            return

        block_ids = [put.block_id for put in req_meta.boundary_puts or []]
        assert all(block_id > 0 for block_id in block_ids), "Null block cannot back a boundary save"
        groups = (
            (
                group_id
                for group_id in range(len(req_meta.block_ids_by_group))
                if group_id not in speculative_blocks_by_group
            )
            if retain_all
            else speculative_blocks_by_group
        )
        for group_id in groups:
            num_speculative_blocks = speculative_blocks_by_group.get(group_id, 0)
            group_block_ids = req_meta.block_ids_by_group[group_id]
            # Speculative scratch slots must remain exclusively owned for relocation.
            non_spec_block_ids = (
                group_block_ids[:-num_speculative_blocks] if num_speculative_blocks > 0 else group_block_ids
            )
            block_ids.extend(block_id for block_id in non_spec_block_ids if block_id > 0)

        block_ids = list(dict.fromkeys(block_ids))
        if not block_ids:
            return

        assert block_pool is not None
        event_id = self.next_event_id
        self.next_event_id += 1
        req_meta.event_id = event_id
        req_meta.store_job_id = event_id
        logger.debug("event: %s touch blocks: %s", event_id, block_ids)
        block_pool.touch([block_pool.blocks[block_id] for block_id in block_ids])
        self.sending_blocks[event_id] = block_ids
        self.sending_events[event_id] = 0

    def complete(
        self,
        completed_events: dict[int, int],
        block_pool: BlockPool,
    ) -> None:
        for event_id, count in completed_events.items():
            logger.debug("event %s update with %s", event_id, count)
            total = self.sending_events.get(event_id)
            if total is None:
                logger.warning("worker reports an invalid event: %s, count %s", event_id, count)
                continue
            total += count
            if total > self.expected_worker_count:
                raise ValueError(f"Too many completions for save event {event_id}")
            if total < self.expected_worker_count:
                self.sending_events[event_id] = total
                continue

            block_ids = self.sending_blocks.pop(event_id, [])
            del self.sending_events[event_id]
            if block_ids:
                logger.debug("event %s free blocks: %s", event_id, block_ids)
                # Release the tail first so a shared prefix reaches the eviction queue last.
                block_pool.free_blocks([block_pool.blocks[block_id] for block_id in reversed(block_ids)])

    def has_pending_work(self) -> bool:
        return bool(self.sending_events)

    def has_tracked_blocks(self) -> bool:
        return bool(self.sending_events or self.sending_blocks)
