# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optional AV budget policy shared by verification and physical-K scoring."""

from types import MethodType

import numpy as np


def select_av_budget(expected_output: np.ndarray, scores: np.ndarray, tolerance: float) -> int:
    best = int(np.argmax(scores))
    if not tolerance or scores[best] <= 0:
        return best

    # Only larger budgets can improve expected output because prefix gains are nonnegative.
    eligible = np.flatnonzero(
        (scores[best + 1 :] >= scores[best] * (1 - tolerance)) & (expected_output[best + 1 :] > expected_output[best])
    )
    return best + 1 + int(eligible[-1]) if eligible.size else best


def _get_num_tokens(manager, num_tokens_per_req, draft_tokens) -> int:
    # Mirror the input and output contract of vLLM AdaptiveVerificationManager.get_num_tokens.
    # Remove this adapter when upstream exposes a budget selection hook.
    req_ids = list(num_tokens_per_req)
    num_reqs = len(req_ids)
    scheduled = np.fromiter(num_tokens_per_req.values(), dtype=np.int32, count=num_reqs)
    drafts = np.fromiter((len(draft_tokens.get(r, ())) for r in req_ids), dtype=np.int32, count=num_reqs)
    slots = np.fromiter(map(manager.req_states.req_id_to_index.__getitem__, req_ids), dtype=np.int32, count=num_reqs)
    non_draft = scheduled - drafts
    total_non_draft = int(non_draft.sum())
    confidence = manager._stale_confidences[manager._stale_idx].np[slots]
    survival = np.cumprod(confidence.astype(np.float64), axis=1)
    valid = np.arange(manager.num_speculative_steps)[None, :] < drafts[:, None]
    limit = min(int(drafts.sum()), max(0, manager._max_total_logits - num_reqs * manager.num_bonus_tokens))
    ranked = np.sort(survival[valid])[::-1][:limit]
    sampling_reqs = np.count_nonzero(
        manager.req_states.num_computed_tokens_np[slots] + non_draft >= manager.req_states.prefill_len.np[slots]
    )
    expected = np.concatenate(([sampling_reqs], sampling_reqs + np.cumsum(ranked)))
    assert manager.cost_tables is not None
    draft_cost, verify_cost = manager.cost_tables
    costs = draft_cost[num_reqs] + verify_cost[total_non_draft : total_non_draft + limit + 1]
    budget = select_av_budget(expected, expected / costs, manager._ascend_av_score_tolerance)
    manager._batch_budget = (
        {r: int(n) for r, n in zip(req_ids, drafts, strict=True)},
        {r: int(n) for r, n in zip(req_ids, non_draft, strict=True)},
        budget,
    )
    return total_non_draft + budget


def configure_av_budget(manager, tolerance: float):
    if not 0 <= tolerance < 1:
        raise ValueError("VLLM_ASCEND_AV_SCORE_TOLERANCE must be in [0, 1)")
    if tolerance:
        manager._ascend_av_score_tolerance = tolerance
        manager.get_num_tokens = MethodType(_get_num_tokens, manager)
    return manager
