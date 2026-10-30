# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import numpy as np
import torch
import xgrammar as xgr
from vllm.v1.core.sched.output import GrammarOutput

from vllm_ascend._310p.worker.v2.input_batch import Ascend310PInputBatch


def apply_grammar_bitmask(
    logits: torch.Tensor,
    input_batch: Ascend310PInputBatch,
    grammar_output: GrammarOutput,
) -> np.ndarray | None:
    """Apply scheduler masks on CPU, as in 310P MRV1, without Triton.

    Masks have a fixed stride per grammar request; logits contain only the
    positions actually scheduled, in runner batch order. Return invalid draft
    positions in that same compact layout for MTP verification.
    """
    grammar_req_ids = grammar_output.structured_output_request_ids
    if not grammar_req_ids or logits.shape[0] == 0:
        return None

    bitmask = grammar_output.grammar_bitmask
    mask_stride = bitmask.shape[0] // len(grammar_req_ids)
    sorted_bitmask = torch.full((logits.shape[0], bitmask.shape[1]), -1, dtype=torch.int32)
    req_id_to_idx = {req_id: i for i, req_id in enumerate(input_batch.req_ids)}
    invalid_drafts = np.zeros(logits.shape[0], dtype=bool) if input_batch.num_draft_tokens else None
    acceptable = grammar_output.num_acceptable_drafts
    out_indices = []
    for grammar_idx, req_id in enumerate(grammar_req_ids):
        req_idx = req_id_to_idx.get(req_id)
        if req_idx is None:
            continue
        start, end = map(int, input_batch.cu_num_logits_np[req_idx : req_idx + 2])
        mask_start = grammar_idx * mask_stride
        sorted_bitmask[start:end].copy_(torch.from_numpy(bitmask[mask_start : mask_start + end - start]))
        out_indices.extend(range(start, end))
        if invalid_drafts is not None:
            # Draft i occupies input/logit position i + 1. A missing bound
            # means no draft can safely be accepted (upstream semantics).
            limit = acceptable[grammar_idx] if acceptable is not None else 0
            invalid_drafts[start + limit + 1 : end] = True

    logits_cpu = logits.to(device="cpu", dtype=torch.float32)
    xgr.apply_token_bitmask_inplace(logits_cpu, sorted_bitmask, indices=out_indices)
    logits.copy_(logits_cpu)
    if invalid_drafts is not None and not invalid_drafts.any():
        return None
    return invalid_drafts
