# SPDX-License-Identifier: Apache-2.0
"""Version-checked Ascend worker call site; remove when upstream exposes a hook."""

import inspect
from functools import wraps
from pathlib import Path

from . import resident_updates

WORKER_SHA = "1953b3e582502d16df38bba4a5966e0b98492feb67b65cdac7f2b0397ca4b229"
RUNTIME_SHA = "89b940424066b62bf042bcdade3b7efacf03968d9f449888c12b351c636cb926"


def install():
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse import worker
    from vllm.v1.hisparse import runtime

    from .optimization_profile import verify_core_source

    original = worker.update_hisparse_residency
    if getattr(original, "_ascend_direct_Resident_update", False):
        return
    if original is not runtime.update_hisparse_residency:
        raise RuntimeError("Another owner changed the Resident worker call site")
    for obj, digest in ((worker.HiSparseConnectorWorker.start_step, WORKER_SHA), (original, RUNTIME_SHA)):
        path = Path(inspect.getsourcefile(inspect.unwrap(obj)))
        verify_core_source(path, digest)

    @wraps(original)
    def update(residency, updates, request_ids, request_state_indices):
        if residency.device.type != "npu":
            return original(residency, updates, request_ids, request_state_indices)
        # Current Core owns [group,state,page]; the existing stride-aware kernel
        # consumes [state,group,page]. A view preserves its allocation and values.
        state_first = residency.permute(1, 0, 2)

        def fallback(_view, updates, request_ids, request_state_indices):
            return original(residency, updates, request_ids, request_state_indices)

        return resident_updates.candidate_update(state_first, updates, request_ids, request_state_indices, fallback)

    update._ascend_direct_Resident_update = True
    update._ascend_original = original
    worker.update_hisparse_residency = update
