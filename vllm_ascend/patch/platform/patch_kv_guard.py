# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport allocation generations to the Worker L1 state machine.

No lifecycle trace, extra publisher, token hashes or per-token log messages.
Changed bindings, releases and compact per-request write ranges use the existing channel.
"""

import functools
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
        free = self.kv_cache_manager.free

        @functools.wraps(free)
        def freed(request, *args, **kwargs):
            result = free(request, *args, **kwargs)
            released.add(request.request_id)
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
            for request_id in output.num_scheduled_tokens:
                blocks = [list(group) for group in self.kv_cache_manager.get_block_ids(request_id)]
                refs = [[(block, counters[block]) for block in ids] for ids in blocks]
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
            released.clear()
            return output

        self.schedule = scheduled

    cast(Any, initialize)._ascend_kv_guard = True
    scheduler_cls.__init__ = initialize


def apply_patch():
    from vllm.v1.core.sched.scheduler import Scheduler

    install_scheduler_guard(Scheduler)
