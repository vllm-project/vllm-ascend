# SPDX-License-Identifier: Apache-2.0
"""Detached candidate: omit an unused lazy Target row snapshot, not count D2H.

Scope is one synchronous start_load_kv call. Unknown/Prefill/pending-draft
states retain the original staging and waits. No DMA/page ownership or MTP
ordering changes. Requires actual model qualification before deployment.
"""

import inspect
import threading
from functools import wraps
from pathlib import Path

WORKER_SHA = "1953b3e582502d16df38bba4a5966e0b98492feb67b65cdac7f2b0397ca4b229"
CONNECTOR_SHA = "78610e02ebdd622dc8272a98d75c2a75a41d8b19ce1e72ac582bdedd19d11f13"
FLAG = "_ascend_certified_lazy_mtp_mirror"


def scope_for(worker, attn_metadata, num_tokens, all_resident, eligible):
    """Read current Host metadata; never retain values across proposals."""
    state = worker._slot_mapping_staging
    if (
        all_resident is not True
        or type(num_tokens) is not int
        or num_tokens <= 0
        or state is None
        or state.num_tokens != 0
        or worker._draft_mirror_pending
        or worker._per_layer_mirrored
        or worker._submitted_mirror_layers
        or worker.shared_host_region is not None
        or not isinstance(attn_metadata, dict)
    ):
        return None
    if not worker.cache_handles or not worker._group_leaders:
        return None
    for handle in worker.cache_handles:
        if handle.runtime.device.type != "npu" or handle.runtime.eager_host_mirror is not False:
            return None
    cache = {}
    for name, handle in worker._group_leaders:
        metadata = attn_metadata.get(name)
        capacity = handle.runtime.max_swap_rows
        key = (id(metadata), capacity)
        if key not in cache:
            if metadata is None or metadata.num_actual_tokens != num_tokens or not eligible(metadata, capacity):
                return None
            # Strong reference plus call-scoped lifetime prevents identity
            # reuse. Common prepare_for_batch reads these fields without
            # changing the metadata; batch values are re-read next call.
            cache[key] = (metadata, True)
    return dict(worker=worker, state=state, num_tokens=num_tokens, eligibility=cache, skipped=0)


def make_bindings(original_load, original_stage, original_eligible):
    """Build the exact call-scoped wrappers, also exercised by Host contracts."""
    local = threading.local()

    @wraps(original_eligible)
    def eligible(metadata, max_rows):
        scope = getattr(local, "scope", None)
        if scope is not None:
            record = scope["eligibility"].get((id(metadata), max_rows))
            if record is not None and record[0] is metadata:
                return record[1]
        return original_eligible(metadata, max_rows)

    @wraps(original_stage)
    def stage(worker, num_tokens):
        scope = getattr(local, "scope", None)
        if scope is None or scope["worker"] is not worker:
            return original_stage(worker, num_tokens)
        if (
            scope["state"] is not worker._slot_mapping_staging
            or scope["state"].num_tokens != 0
            or scope["num_tokens"] != num_tokens
            or scope["skipped"]
        ):
            raise RuntimeError("Changed/reentrant certified lazy row-staging scope")
        scope["skipped"] += 1
        # state.num_tokens==0 means no new snapshot was submitted. The old
        # source only resets it after the old staging event has completed.
        # finish_mirror_phase consequently has no snapshot consumer or wait;
        # grouped device invalidation and sealed-page DMA remain unchanged.

    @wraps(original_load)
    def load(connector, forward_context, **kwargs):
        if getattr(local, "scope", None) is not None:
            raise RuntimeError("Reentrant HiSparse start_load_kv is unsupported")
        worker = connector.connector_worker
        metadata = connector._get_connector_metadata()
        scope = scope_for(
            worker,
            kwargs.get("attn_metadata"),
            kwargs.get("num_tokens"),
            metadata.all_context_pages_resident,
            original_eligible,
        )
        local.scope = scope
        try:
            result = original_load(connector, forward_context, **kwargs)
            if scope is not None:
                assert scope["skipped"] == 1 and scope["state"].num_tokens == 0
                active = [h for h in worker.cache_handles if h.num_actual_tokens]
                if not active or any(
                    getattr(h, FLAG, False) is not True or h.host_mirror_required or h.decode_batch for h in active
                ):
                    raise RuntimeError("Actual prepare differs from certified lazy row-snapshot scope")
            return result
        finally:
            local.scope = None

    load._ascend_unused_lazy_staging = True
    load._ascend_original = original_load
    stage._ascend_unused_lazy_staging = True
    stage._ascend_original = original_stage
    return eligible, stage, load


def install():
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import HiSparseConnector
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker

    from vllm_ascend.distributed.kv_transfer.hisparse.optimization_profile import verify_core_source

    from . import patch_hisparse_lazy_mtp_mirror as policy

    original_load = HiSparseConnector.start_load_kv
    if getattr(original_load, "_ascend_unused_lazy_staging", False):
        return
    original_stage = HiSparseConnectorWorker._stage_row_mirror_mapping
    for function, expected in ((original_load, CONNECTOR_SHA), (original_stage, WORKER_SHA)):
        path = Path(inspect.getsourcefile(inspect.unwrap(function)))
        verify_core_source(path, expected)
    if not getattr(HiSparseConnectorWorker._enqueue_host_mirror, "_ascend_certified_lazy_mtp_mirror_patch", False):
        raise RuntimeError("Install the reviewed D-only lazy mirror policy first")
    eligible, stage, load = make_bindings(original_load, original_stage, policy._eligible)
    policy._eligible = eligible
    HiSparseConnector.start_load_kv = load
    HiSparseConnectorWorker._stage_row_mirror_mapping = stage
