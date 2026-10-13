# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport allocation generations to the Worker L1 state machine.

No lifecycle trace, extra publisher, token hashes or per-token log messages.
Only changed bindings and finished requests are sent on the existing channel.
"""

import functools
import uuid
from typing import Any, cast


def install_scheduler_guard(scheduler_cls):
    if getattr(scheduler_cls.__init__, "_ascend_kv_guard", False):
        return
    original_init = scheduler_cls.__init__

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if getattr(getattr(self, "vllm_config", None), "use_v2_model_runner", False):
            raise ValueError("KV L1 guard supports Model Runner V1; set VLLM_USE_V2_MODEL_RUNNER=0")
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

        @functools.wraps(schedule)
        def scheduled(*args, **kwargs):
            output = schedule(*args, **kwargs)
            computed = {r.req_id: r.num_computed_tokens for r in output.scheduled_new_reqs}
            cached = output.scheduled_cached_reqs
            computed.update(zip(cached.req_ids, cached.num_computed_tokens))
            changed = []
            for request_id in output.finished_req_ids:
                mappings.pop(request_id, None)
            for request_id in output.num_scheduled_tokens:
                blocks = [list(group) for group in self.kv_cache_manager.get_block_ids(request_id)]
                refs = [
                    {
                        "pool_id": pool_id,
                        "alloc_epoch": str(counters[block]) if counters[block] else None,
                        "group_id": group,
                        "block_id": block,
                        "logical_block_index": index,
                    }
                    for group, ids in enumerate(blocks)
                    for index, block in enumerate(ids)
                    if block != pool.null_block.block_id
                ]
                signature = tuple(tuple(g) for g in blocks), tuple(ref["alloc_epoch"] for ref in refs)
                if mappings.get(request_id) != signature:
                    changed.append(
                        {
                            "request_id": request_id,
                            "block_ids": blocks,
                            "block_refs": refs,
                            "num_computed_tokens": computed.get(request_id, 0),
                        }
                    )
                    mappings[request_id] = signature
            output._ascend_kv_guard_context = {
                "requests": changed,
                "finished_request_ids": sorted(output.finished_req_ids),
            }
            return output

        self.schedule = scheduled

    cast(Any, initialize)._ascend_kv_guard = True
    scheduler_cls.__init__ = initialize


def apply_patch():
    from vllm.v1.core.sched.scheduler import Scheduler

    install_scheduler_guard(Scheduler)
