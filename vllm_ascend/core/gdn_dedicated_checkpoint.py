# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Track independently allocated GDN pages through an executing prefill."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request

    from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager


# (KV cache group, prompt position, physical page).
GDNCheckpointWrites = dict[str, tuple[tuple[int, int, int], ...]]


@dataclass
class GDNCheckpointAcknowledgement:
    writes: GDNCheckpointWrites
    layers: set[str] = field(default_factory=set)


def confirmed_writes(plans: GDNCheckpointWrites, attn_metadata, cache_groups) -> GDNCheckpointWrites:
    """Return only requests for which every planned layer ran its exporter."""
    ready: GDNCheckpointWrites = {}
    for req_id, writes in plans.items():
        for write in writes:
            group = cache_groups[write[0]]
            if not group.layer_names:
                break
            for layer in group.layer_names:
                checkpoint = getattr(attn_metadata.get(layer), "checkpoint", None)
                ack = getattr(checkpoint, "dedicated_ack", None)
                if ack is None or layer not in ack.layers or write not in ack.writes.get(req_id, ()):
                    break
            else:
                continue
            break
        else:
            ready[req_id] = writes
    return ready


@dataclass(frozen=True)
class _ReservedCheckpoint:
    request: "Request"
    preemptions: int
    pages: tuple[tuple["AscendMambaManager", int, "KVCacheBlock"], ...]

    @property
    def writes(self) -> tuple[tuple[int, int, int], ...]:
        return tuple((manager.kv_cache_group_id, position, block.block_id) for manager, position, block in self.pages)


class DedicatedGDNCheckpoints:
    def __init__(self, managers: tuple["AscendMambaManager", ...]) -> None:
        self.managers = managers
        self.inflight: dict[int, list[_ReservedCheckpoint]] = {}

    def begin_step(self) -> None:
        for manager in self.managers:
            manager.dedicated_plans.clear()

    def plan(
        self,
        request_id: str,
        start: int,
        position: int,
        end: int,
        hash_size: int,
        block_size: int,
        required_blocks: int,
    ) -> bool:
        if start % hash_size or position - start < hash_size:
            return False
        for manager in self.managers:
            alignment = manager.kv_cache_spec.prefill_checkpoint_alignment
            if manager.block_size != block_size or not alignment or (position - start) % alignment:
                return False
        if self.managers[0].block_pool.get_num_free_blocks() < required_blocks + len(self.managers):
            return False
        for manager in self.managers:
            manager.dedicated_plans[request_id] = (position, end)
        return True

    def dispatch(self, requests: dict[str, "Request"], output: "SchedulerOutput") -> None:
        by_request: dict[str, list[tuple[AscendMambaManager, int, KVCacheBlock]]] = {}
        for manager in self.managers:
            for req_id, (position, block) in manager.dedicated_pages.items():
                by_request.setdefault(req_id, []).append((manager, position, block))
            manager.dedicated_pages.clear()
        if not by_request:
            return
        reservations = []
        for req_id, pages in by_request.items():
            assert req_id in output.num_scheduled_tokens and len(pages) == len(self.managers)
            request = requests[req_id]
            reservations.append(_ReservedCheckpoint(request, request.num_preemptions, tuple(pages)))
        output.gdn_dedicated_checkpoints = {item.request.request_id: item.writes for item in reservations}
        self.inflight[id(output)] = reservations

    def publish(self, requests: dict[str, "Request"], step: "SchedulerOutput", output: "ModelRunnerOutput") -> None:
        reservations = self.inflight.pop(id(step), ())
        if not reservations:
            return
        completed = set(output.req_ids)
        written = getattr(output, "gdn_dedicated_checkpoint_writes", {})
        try:
            for item in reservations:
                req_id = item.request.request_id
                request = requests.get(req_id)
                if (
                    request is item.request
                    and not request.is_finished()
                    and not request.num_stale_output_tokens
                    and request.num_preemptions == item.preemptions
                    and req_id in completed
                    and set(item.writes) == set(written.get(req_id, ()))
                ):
                    for manager, position, block in item.pages:
                        manager.block_pool.cache_partial_block(
                            request, block, position, manager.kv_cache_group_id, manager.block_size
                        )
        finally:
            for item in reservations:
                for manager, _, block in item.pages:
                    manager.block_pool.free_blocks([block])


def allocation_bound(cache_managers, tokens: int, watermark_blocks: int) -> int:
    # For a waiting local hit, count even the prefix pages: currently unused
    # hits are taken off the same free list when the normal allocation pins them.
    return watermark_blocks + sum(
        cdiv(tokens, manager.block_size) + getattr(manager, "num_speculative_blocks", 0) for manager in cache_managers
    )
