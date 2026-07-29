# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Thin v0.23 model-runner adaptation for DSA sparse offload.

The v0.19-derived implementation copied ``GPUModelRunner._update_states``.
That is intentionally avoided here: v0.23 owns request construction, MTP
draft correction, streaming sessions, batch condensation and backend reorder.
DSA only projects the two pieces that differ from native behavior:

* a sparse MLA block-table update is a replacement, not an append;
* the full-block hash ledger is transported as a snapshot plus suffix deltas.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_ascend.dsa_sparse.dsa_block_hash_delta import (
    apply_context_full_block_hash_delta,
)
from vllm_ascend.dsa_sparse.dsa_spec_utils import is_dsa_mla_resident_spec
from vllm_ascend.dsa_sparse.dsa_types import INVALID_SLOT, ReqStage

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def dsa_request_finished_in_worker(
    self: "NPUModelRunner",
    request_id: str,
) -> None:
    dsa_mgr = self.dsa_worker_mgr
    if dsa_mgr is not None:
        dsa_mgr.request_finished_in_worker(request_id)


def dsa_request_preempted_in_worker(
    self: "NPUModelRunner",
    request_id: str,
) -> None:
    dsa_mgr = self.dsa_worker_mgr
    if dsa_mgr is not None:
        dsa_mgr.request_preempted_in_worker(request_id)


def normalize_dsa_decode_block_ids(
    self: "NPUModelRunner",
    req_id: str,
    req_state: CachedRequestState,
    new_block_ids: tuple[list[int], ...] | None,
    *,
    resumed_from_preemption: bool,
) -> tuple[list[int], ...] | None:
    """Normalize a sparse scheduler snapshot to every configured KV group.

    The split coordinator normally returns all groups.  The one-group branch
    remains for compatibility with an older scheduler producer that emitted
    only the MLA group during steady sparse decode.
    """
    if new_block_ids is None:
        return None

    expected_num_groups = len(req_state.block_ids)
    if expected_num_groups == 0 or len(new_block_ids) == expected_num_groups:
        return tuple(list(group) for group in new_block_ids)

    kv_cache_groups = self.kv_cache_config.kv_cache_groups
    if len(kv_cache_groups) != expected_num_groups:
        raise RuntimeError(
            "DSA decode block-id normalization got inconsistent KV groups "
            f"for req {req_id}: expected_groups={expected_num_groups}, "
            f"kv_cache_groups={len(kv_cache_groups)}, "
            f"new_block_groups={len(new_block_ids)}")

    full_group_ids = [
        group_id
        for group_id, kv_cache_group in enumerate(kv_cache_groups)
        if is_dsa_mla_resident_spec(kv_cache_group.kv_cache_spec)
    ]
    if len(new_block_ids) == 1 and len(full_group_ids) == 1:
        if resumed_from_preemption:
            raise RuntimeError(
                "Resumed sparse request must refresh every KV group, but got "
                f"only one block-id group for req {req_id}; "
                f"expected_groups={expected_num_groups}")
        normalized = [list(group) for group in req_state.block_ids]
        normalized[full_group_ids[0]] = list(new_block_ids[0])
        return tuple(normalized)

    raise RuntimeError(
        "DSA decode block-id normalization could not map scheduler output "
        f"for req {req_id}: expected_groups={expected_num_groups}, "
        f"new_block_groups={len(new_block_ids)}, "
        f"full_group_ids={full_group_ids}")


def _is_sparse_cached_request(
    scheduler_output: "SchedulerOutput",
    req_id: str,
    req_state: CachedRequestState,
) -> bool:
    stages = getattr(scheduler_output, "req_dsa_stage", None) or {}
    resident_lens = (
        getattr(scheduler_output, "req_dsa_resident_valid_seq_len", None)
        or {})
    resident_len = int(resident_lens.get(req_id, INVALID_SLOT))
    fallback = (
        ReqStage.SPARSE_DECODE
        if resident_len != INVALID_SLOT else ReqStage.DENSE_DECODE)
    stage = ReqStage.coerce(stages.get(req_id, fallback))
    return bool(
        req_state.output_token_ids
        and stage.is_sparse_decode
        and resident_len != INVALID_SLOT)


def update_states(
    self: "NPUModelRunner",
    scheduler_output: "SchedulerOutput",
    native_update_states: Callable[["SchedulerOutput"], Callable | None],
) -> Callable | None:
    """Run v0.23 native state update with DSA block-table projections.

    ``native_update_states`` is the bound ``super()._update_states`` callable
    supplied by ``NPUModelRunner``.  Keeping that call authoritative preserves
    every v0.23 MTP and request-lifecycle correction.
    """
    if self.dsa_worker_mgr is None:
        return native_update_states(scheduler_output)

    for req_id in scheduler_output.finished_req_ids:
        dsa_request_finished_in_worker(self, req_id)
    for req_id in scheduler_output.preempted_req_ids or ():
        dsa_request_preempted_in_worker(self, req_id)

    req_data = scheduler_output.scheduled_cached_reqs
    original_new_block_ids = req_data.new_block_ids
    projected_new_block_ids = list(original_new_block_ids)
    sparse_replacements: dict[
        str, tuple[list[int], ...]
    ] = {}

    for index, req_id in enumerate(req_data.req_ids):
        req_state = self.requests.get(req_id)
        if req_state is None or not _is_sparse_cached_request(
                scheduler_output, req_id, req_state):
            continue
        new_block_ids = original_new_block_ids[index]
        if new_block_ids is None:
            continue
        resumed = req_id in req_data.resumed_req_ids
        normalized = normalize_dsa_decode_block_ids(
            self,
            req_id,
            req_state,
            new_block_ids,
            resumed_from_preemption=resumed,
        )
        if normalized is None:
            continue
        if not resumed:
            # Native treats cached non-resumed IDs as a delta.  Sparse
            # allocation instead returns a complete Indexer+MLA snapshot after
            # the MLA table may have shrunk, so suppress native append and
            # install the snapshot after batch condensation/reorder.
            sparse_replacements[req_id] = normalized
            projected_new_block_ids[index] = None
        else:
            projected_new_block_ids[index] = normalized

    req_data.new_block_ids = projected_new_block_ids
    try:
        deferred_correction = native_update_states(scheduler_output)
    finally:
        # Other post-execute consumers still observe the scheduler payload.
        req_data.new_block_ids = original_new_block_ids

    for req_id, block_ids in sparse_replacements.items():
        req_state = self.requests.get(req_id)
        if req_state is None:
            continue
        req_state.block_ids = tuple(list(group) for group in block_ids)
        req_index = self.input_batch.req_id_to_index.get(req_id)
        if req_index is not None:
            self.input_batch.block_table.add_row(block_ids, req_index)

    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_state = self.requests.get(new_req_data.req_id)
        if req_state is not None:
            req_state.context_full_blk_hashes = list(
                getattr(new_req_data, "block_hashes", None) or ())

    hash_starts = getattr(req_data, "block_hash_starts", ())
    hash_deltas = getattr(req_data, "block_hashes", ())
    for index, req_id in enumerate(req_data.req_ids):
        if index >= len(hash_deltas):
            break
        req_state = self.requests.get(req_id)
        if req_state is None:
            continue
        ledger = getattr(req_state, "context_full_blk_hashes", None)
        if ledger is None:
            ledger = []
            req_state.context_full_blk_hashes = ledger
        start = int(hash_starts[index]) if index < len(hash_starts) else 0
        apply_context_full_block_hash_delta(
            ledger,
            start,
            hash_deltas[index],
        )

    return deferred_correction


def update_streaming_request_hashes(
    req_state: CachedRequestState,
    new_req_data: Any,
) -> None:
    """Reset the DSA hash ledger after a native streaming-session refresh."""
    req_state.context_full_blk_hashes = list(
        getattr(new_req_data, "block_hashes", None) or ())
