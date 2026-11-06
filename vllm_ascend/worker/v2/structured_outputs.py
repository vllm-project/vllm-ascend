# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[Role] Backport vLLM #54442 grammar acceptance limits to Ascend MRv2.
[Invariants] Mask verification rows using actual device logits boundaries.
[Timeouts & Locks] No explicit synchronization, device reads, or locks.
[Side-Effect Policy] Allocate masks bounded by the active batch/logit rows.
"""

import numpy as np
import torch
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.input_batch import InputBatch


def grammar_invalid_drafts(
    input_batch: InputBatch,
    grammar_req_ids: list[str],
    num_acceptable_drafts: list[int] | None,
) -> torch.Tensor | None:
    """Draft i is at local row i+1; reject rows beyond the constrained prefix.

    Device boundaries reflect adaptive verification's compaction, unlike the
    scheduler's upper bounds. Explicit output_size avoids a device-to-host
    shape read in repeat_interleave.
    """
    if not grammar_req_ids or input_batch.num_draft_tokens == 0:
        return None
    req_id_to_idx = {req_id: i for i, req_id in enumerate(input_batch.req_ids)}
    limit = np.full(input_batch.num_reqs, np.iinfo(np.int32).max, dtype=np.int32)
    for i, req_id in enumerate(grammar_req_ids):
        req_idx = req_id_to_idx.get(req_id)
        if req_idx is not None:
            limit[req_idx] = num_acceptable_drafts[i] if num_acceptable_drafts is not None else 0
    num_drafts = input_batch.num_draft_tokens_per_req
    if num_drafts is not None and (limit >= num_drafts).all():
        return None
    local_pos = input_batch.expanded_local_pos
    boundaries = input_batch.cu_num_logits
    row_limit = torch.repeat_interleave(
        async_tensor_h2d(limit, device=local_pos.device),
        boundaries[1:] - boundaries[:-1],
        output_size=local_pos.shape[0],
    )
    return local_pos > row_limit
