# SPDX-License-Identifier: Apache-2.0
"""Ascend adapter for fixed-K adaptive verification.

Keeps upstream vLLM's graph-aware cost model and draft-token allocator.
When additional_config.av_score_tolerance > 0, allows a larger verification budget
within the specified relative loss of *predicted* throughput.

Example:
    --additional-config '{"av_score_tolerance": 0.01}'

When unset or zero, vLLM's original budget-selection method is unchanged.
"""

import logging
from types import MethodType

import numpy as np
import torch
from torch.overrides import TorchFunctionMode
from vllm.v1.worker.gpu.spec_decode import adaptive_verification

logger = logging.getLogger(__name__)


def _index_fill(tensor, dim, index, value):
    # Import lazily to avoid circular imports during plugin startup.
    from vllm_ascend.device.device_op import DeviceOperator

    return DeviceOperator.index_fill(tensor, dim, index, value)


class _IndexFillMode(TorchFunctionMode):
    """Route index_fill_ through Ascend without a device-index host sync."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is torch.Tensor.index_fill_:
            return _index_fill(*args, **kwargs)
        return func(*args, **kwargs)


# Preserve the existing Ascend A5 workaround for per-request allocation.
_original_assign_draft_token_budget = adaptive_verification._assign_draft_token_budget


def _assign_draft_token_budget_ascend(*args, **kwargs):
    with _IndexFillMode():
        return _original_assign_draft_token_budget(*args, **kwargs)


adaptive_verification._assign_draft_token_budget_compiled = (
    _assign_draft_token_budget_ascend
)


def configure_adaptive_verification(manager, score_tolerance: float) -> None:
    """Bind the validated startup option to this runner's manager only.

    Called after GPUModelRunner.initialize_kv_cache creates the manager, before
    profiling or inference. Zero leaves the upstream method untouched.
    """
    if manager is not None and score_tolerance > 0:
        manager.av_score_tolerance = score_tolerance
        manager.get_num_tokens = MethodType(_get_num_tokens_ascend, manager)
        logger.info("Ascend adaptive verification: score tolerance = %.4f", score_tolerance)


def _select_draft_budget(
    estimated_output: np.ndarray,
    estimated_scores: np.ndarray,
    score_tolerance: float,
) -> int:
    """Choose the largest useful budget within tolerance of peak predicted TPS.

    A candidate must increase expected output over the unconstrained
    throughput-optimal budget. Selecting the largest such budget maximizes
    expected output because accepted-prefix gains are nonnegative.
    """
    speed_budget = int(np.argmax(estimated_scores))
    if not score_tolerance or estimated_scores[speed_budget] <= 0:
        return speed_budget

    eligible = np.flatnonzero(
        (estimated_scores >= estimated_scores[speed_budget] * (1 - score_tolerance))
        & (estimated_output > estimated_output[speed_budget])
    )
    return int(eligible[-1]) if eligible.size else speed_budget


def _get_num_tokens_ascend(
    self,
    num_tokens_per_req: dict[str, int],
    draft_tokens: dict[str, list[int]],
) -> int:
    """Upstream budget calculation with only the final selection rule changed.

    Physical draft K, cost-table creation, compact_batch(), reallocate_drafts(),
    and the device-side top-k draft allocation remain upstream implementations.
    """
    assert self.cost_tables is not None
    req_ids = list(num_tokens_per_req)
    num_reqs = len(req_ids)
    scheduled_tokens = np.fromiter(
        num_tokens_per_req.values(), dtype=np.int32, count=num_reqs
    )
    scheduled_drafts = np.fromiter(
        (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
        dtype=np.int32,
        count=num_reqs,
    )
    num_non_draft_tokens = scheduled_tokens - scheduled_drafts
    slots = np.fromiter(
        map(self.req_states.req_id_to_index.__getitem__, req_ids),
        dtype=np.int32,
        count=num_reqs,
    )
    stale_confidences = self._stale_confidences[self._stale_idx].np[slots]
    survival_probability = np.cumprod(stale_confidences.astype(np.float64), axis=1)
    steps = np.arange(self.num_speculative_steps)
    valid = steps[None, :] < scheduled_drafts[:, None]
    scores = np.sort(survival_probability[valid])[::-1]
    num_non_draft_tokens_total = int(num_non_draft_tokens.sum())
    max_draft_budget = min(
        int(scheduled_drafts.sum()),
        max(0, self._max_total_logits - num_reqs * self.num_bonus_tokens),
    )
    scores = scores[:max_draft_budget]
    draft_cost_ms, verify_cost_ms = self.cost_tables
    num_sampling_requests = np.count_nonzero(
        self.req_states.num_computed_tokens_np[slots] + num_non_draft_tokens
        >= self.req_states.prefill_len.np[slots]
    )
    estimated_output = np.concatenate(
        ([num_sampling_requests], num_sampling_requests + np.cumsum(scores))
    )
    costs = (
        draft_cost_ms[num_reqs]
        + verify_cost_ms[
            num_non_draft_tokens_total : num_non_draft_tokens_total
            + max_draft_budget
            + 1
        ]
    )
    num_drafts_per_req = {
        req_id: int(num_drafts)
        for req_id, num_drafts in zip(req_ids, scheduled_drafts, strict=True)
    }
    num_non_draft_tokens_per_req = {
        req_id: int(num_tokens)
        for req_id, num_tokens in zip(req_ids, num_non_draft_tokens, strict=True)
    }

    # This is the only change from the upstream global budget decision:
    # upstream uses argmax(estimated_output / costs) unconditionally.
    budget = _select_draft_budget(
        estimated_output, estimated_output / costs, self.av_score_tolerance
    )
    self._batch_budget = (
        num_drafts_per_req,
        num_non_draft_tokens_per_req,
        budget,
    )
    return sum(num_non_draft_tokens_per_req.values()) + budget
