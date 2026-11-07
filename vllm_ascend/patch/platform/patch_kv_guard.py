# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport allocation generations to the Worker L1 state machine.

No lifecycle trace, extra publisher, token hashes or per-token log messages.
Changed bindings, releases and compact per-request write ranges use the existing channel.
"""

import functools
import inspect
import uuid
from typing import Any, cast

from vllm_ascend.debug.kv_guard_config import KVGuardConfig


def install_scheduler_guard(scheduler_cls):
    if getattr(scheduler_cls.__init__, "_ascend_kv_guard", False):
        return
    original_init = scheduler_cls.__init__

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not KVGuardConfig.from_vllm_config(self.vllm_config).enabled:
            return
        pool = self.kv_cache_manager.block_pool
        counters = [0] * len(pool.blocks)
        pool_id = uuid.uuid4().hex
        allocate = pool.get_new_blocks

        @functools.wraps(allocate)
        def allocated(*args, **kwargs):
            blocks = allocate(*args, **kwargs)
            for block in blocks:
                counters[block.block_id] += 1
            return blocks

        pool.get_new_blocks = allocated
        schedule = self.schedule
        mappings: dict[str, tuple] = {}
        released: set[str] = set()
        loads: dict[str, dict] = {}
        replays: dict[str, tuple[int, int]] = {}
        load_sequence = 0

        def block_refs(request_id):
            return [
                [(block, counters[block]) for block in group]
                for group in self.kv_cache_manager.get_block_ids(request_id)
            ]

        allocate_slots = getattr(self.kv_cache_manager, "allocate_slots", None)
        if allocate_slots is not None:
            signature = inspect.signature(allocate_slots)
            external_position = list(signature.parameters).index("num_external_computed_tokens") - 1

            @functools.wraps(allocate_slots)
            def slots_allocated(request, *args, **kwargs):
                nonlocal load_sequence
                external = kwargs.get(
                    "num_external_computed_tokens", args[external_position] if len(args) > external_position else 0
                )
                if not external:
                    return allocate_slots(request, *args, **kwargs)
                arguments = signature.bind(request, *args, **kwargs).arguments
                local_end = request.num_computed_tokens + arguments.get("num_new_computed_tokens", 0)
                result = allocate_slots(request, *args, **kwargs)
                if result is not None:
                    load_sequence += 1
                    loads[request.request_id] = {
                        "pool_id": pool_id,
                        "load_id": load_sequence,
                        "request_id": request.request_id,
                        "block_table": block_refs(request.request_id),
                        "token_start": local_end,
                        "token_end": min(
                            local_end + external, getattr(self.kv_cache_manager, "max_model_len", local_end + external)
                        ),
                        "is_async": arguments.get("delay_cache_blocks", False),
                    }
                return result

            self.kv_cache_manager.allocate_slots = slots_allocated

        def record_replay(request, start, end):
            if start < end:
                previous = replays.get(request.request_id, (start, end))
                replays[request.request_id] = min(start, previous[0]), max(end, previous[1])

        mark_replay = getattr(self, "_mark_prefix_replay", None)
        if mark_replay is not None:

            @functools.wraps(mark_replay)
            def replay_marked(request, num_hit_tokens, *args, **kwargs):
                count = mark_replay(request, num_hit_tokens, *args, **kwargs)
                record_replay(request, num_hit_tokens - count, num_hit_tokens)
                return count

            self._mark_prefix_replay = replay_marked

        update_waiting = getattr(self, "_update_waiting_for_remote_kv", None)
        if update_waiting is not None:

            @functools.wraps(update_waiting)
            def waiting_updated(request, *args, **kwargs):
                before = request.num_computed_tokens
                result = update_waiting(request, *args, **kwargs)
                record_replay(request, request.num_computed_tokens, before)
                return result

            self._update_waiting_for_remote_kv = waiting_updated

        free = self.kv_cache_manager.free

        @functools.wraps(free)
        def freed(request, *args, **kwargs):
            result = free(request, *args, **kwargs)
            released.add(request.request_id)
            loads.pop(request.request_id, None)
            replays.pop(request.request_id, None)
            return result

        self.kv_cache_manager.free = freed

        @functools.wraps(schedule)
        def scheduled(*args, **kwargs):
            output = schedule(*args, **kwargs)
            computed = {r.req_id: r.num_computed_tokens for r in output.scheduled_new_reqs}
            cached = output.scheduled_cached_reqs
            computed.update(zip(cached.req_ids, cached.num_computed_tokens))
            changed = {}
            for request_id in output.finished_req_ids | released:
                mappings.pop(request_id, None)
                if request_id in output.finished_req_ids:
                    replays.pop(request_id, None)
            for request_id in dict.fromkeys((*output.num_scheduled_tokens, *loads)):
                refs = block_refs(request_id)
                signature = tuple(tuple(group) for group in refs)
                if mappings.get(request_id) != signature:
                    changed[request_id] = refs
                    mappings[request_id] = signature
            output._ascend_kv_guard_context = {
                "pool_id": pool_id,
                "requests": changed,
                "released_request_ids": sorted(output.finished_req_ids | released),
                "scheduled_requests": {
                    request_id: (computed.get(request_id), count)
                    for request_id, count in output.num_scheduled_tokens.items()
                },
            }
            if loads:
                output._ascend_kv_guard_context["loads"] = list(loads.values())
                loads.clear()
            replay_ranges = []
            for request_id in output.num_scheduled_tokens:
                replay = replays.pop(request_id, None)
                if replay is not None:
                    replay_ranges.append(
                        {
                            "pool_id": pool_id,
                            "request_id": request_id,
                            "block_table": block_refs(request_id),
                            "token_start": replay[0],
                            "token_end": replay[1],
                        }
                    )
            if replay_ranges:
                output._ascend_kv_guard_context["recompute_ranges"] = replay_ranges
            released.clear()
            return output

        self.schedule = scheduled

    cast(Any, initialize)._ascend_kv_guard = True
    scheduler_cls.__init__ = initialize


def apply_patch():
    from vllm.v1.core.sched.scheduler import Scheduler

    install_scheduler_guard(Scheduler)
