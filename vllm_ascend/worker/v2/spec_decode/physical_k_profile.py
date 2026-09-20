# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Extend AV startup profiling with K-indexed draft costs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from types import MethodType
from typing import Any

import numpy as np
from vllm.logger import logger

from vllm_ascend.dynamic_spec import resolve_physical_k

_PROFILE_K: ContextVar[int | None] = ContextVar("ascend_profile_physical_k", default=None)


@contextmanager
def physical_k_profile_scope(physical_k: int | None):
    token = _PROFILE_K.set(physical_k)
    try:
        yield
    finally:
        _PROFILE_K.reset(token)


def profiling_physical_k() -> int | None:
    return _PROFILE_K.get()


def _batch_bucket(batch_size: int) -> int:
    return 1 << (max(int(batch_size), 1) - 1).bit_length()


def _candidate_k(vllm_config: Any, max_k: int) -> tuple[int, ...]:
    dynamic = (getattr(vllm_config, "additional_config", None) or {}).get("dynamic_spec_config", {})
    params = resolve_physical_k(dynamic) or {}
    if not params.get("enabled") or not params.get("auto_tune_enabled"):
        return ()
    values = params.get("capture_k") or range(int(params.get("min_k", 1)), max_k + 1)
    return tuple(sorted({max(1, min(int(k), max_k)) for k in values} | {max_k}))


def configure_physical_k_profiling(manager: Any, vllm_config: Any):
    """Add physical-K samples and recommendations to an upstream AV manager."""

    max_k = int(manager.num_speculative_steps)
    candidates = _candidate_k(vllm_config, max_k)
    if not candidates:
        return manager
    original_batches = manager.batches_to_profile
    original_set_curves = manager.set_initial_cost_curves
    original_get_num_tokens = manager.get_num_tokens
    manager._physical_k_profile_cases = []
    manager._physical_k_draft_costs = None
    manager._physical_k_recommendation = None
    manager._physical_k_last_by_bucket = {}
    manager._physical_k_last_logged = {}

    def batches_to_profile(self, capture_sizes) -> Iterator[dict[str, int]]:
        base = list(original_batches(capture_sizes))
        self._physical_k_profile_cases.clear()
        for physical_k in candidates:
            for batch in base:
                case = dict(batch)
                case["profile_physical_k"] = physical_k
                self._physical_k_profile_cases.append(physical_k)
                yield case

    def set_initial_cost_curves(self, samples) -> None:
        cases = self._physical_k_profile_cases
        if len(samples) != len(cases):
            raise RuntimeError(
                f"physical-K profile mismatch: {len(samples)} timings for {len(cases)} cases"
            )
        full_samples = [sample for sample, k in zip(samples, cases) if k == max_k]
        original_set_curves(full_samples)
        grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
        for sample, physical_k in zip(samples, cases):
            # Match upstream AV: eager-target steps inflate drafter event
            # timings while the CPU is still launching target kernels.  They
            # are useful for the verify curve, but not for pricing draft K.
            if not getattr(sample, "full_cudagraph", True):
                continue
            grouped[(physical_k, _batch_bucket(sample.num_reqs))].append(float(sample.drafter_ms))
        costs: dict[int, dict[int, float]] = defaultdict(dict)
        for (physical_k, bucket), values in grouped.items():
            costs[physical_k][bucket] = float(np.median(values))
        # A TP step completes at the slowest rank. Aggregate that maximum
        # rather than trusting rank 0, which can hide a slow-rank regression.
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            from vllm.distributed.parallel_state import get_tp_group

            tp_group = get_tp_group()
            if tp_group.world_size > 1:
                gathered: list[dict[int, dict[int, float]] | None] = [
                    None
                ] * tp_group.world_size
                dist.all_gather_object(
                    gathered,
                    dict(costs),
                    group=tp_group.cpu_group,
                )
                merged: dict[int, dict[int, float]] = defaultdict(dict)
                for rank_costs in gathered:
                    assert rank_costs is not None
                    for physical_k, buckets in rank_costs.items():
                        for bucket, value in buckets.items():
                            merged[physical_k][bucket] = max(
                                merged[physical_k].get(bucket, 0.0),
                                value,
                            )
                costs = merged
        self._physical_k_draft_costs = dict(costs)
        logger.info("ASCEND_AV_PHYSICAL_K_COSTS candidates=%s costs=%s", candidates, dict(costs))

    def lookup_cost(self, physical_k: int, batch_size: int) -> float | None:
        table = self._physical_k_draft_costs
        if not table or physical_k not in table:
            return None
        points = table[physical_k]
        xs = np.asarray(sorted(points), dtype=np.float64)
        ys = np.asarray([points[int(x)] for x in xs], dtype=np.float64)
        return float(np.interp(_batch_bucket(batch_size), xs, ys))

    def score_k(
        self,
        survival,
        physical_k,
        batch_size,
        num_reqs,
        non_draft,
        num_sampling_requests,
    ):
        draft_cost = lookup_cost(self, physical_k, batch_size)
        if draft_cost is None:
            return None
        verify_cost = self.cost_tables[1]
        if non_draft >= len(verify_cost):
            return None
        scores = np.sort(survival[:, :physical_k].reshape(-1))[::-1]
        logits_budget = max(
            0,
            self._max_total_logits - num_reqs * self.num_bonus_tokens,
        )
        max_budget = max(
            0,
            min(
                len(scores),
                logits_budget,
                len(verify_cost) - non_draft - 1,
            ),
        )
        scores = scores[:max_budget]
        expected = np.concatenate(
            (
                [num_sampling_requests],
                num_sampling_requests + np.cumsum(scores),
            )
        )
        total_cost = draft_cost + verify_cost[non_draft : non_draft + max_budget + 1]
        return float(np.max(expected / total_cost))

    def recommend(self, num_tokens_per_req, draft_tokens):
        if self._physical_k_draft_costs is None or self.cost_tables is None:
            return None
        req_ids = list(num_tokens_per_req)
        num_reqs = len(req_ids)
        active = [req_id for req_id in req_ids if draft_tokens.get(req_id)]
        batch_size = len(active)
        if not batch_size:
            return None
        all_slots = np.fromiter(
            (self.req_states.req_id_to_index[req_id] for req_id in req_ids),
            dtype=np.intp,
            count=num_reqs,
        )
        slots = np.fromiter(
            (self.req_states.req_id_to_index[req_id] for req_id in active),
            dtype=np.intp,
            count=batch_size,
        )
        fresh_width = min(len(draft_tokens[req_id]) for req_id in active)
        confidence = self._stale_confidences[self._stale_idx].np[slots].astype(np.float64)
        survival = np.cumprod(confidence, axis=1)
        num_non_draft = np.fromiter(
            (
                num_tokens_per_req[req_id] - len(draft_tokens.get(req_id, ()))
                for req_id in req_ids
            ),
            dtype=np.int32,
            count=num_reqs,
        )
        non_draft = int(num_non_draft.sum())
        num_sampling_requests = int(
            np.count_nonzero(
                self.req_states.num_computed_tokens_np[all_slots] + num_non_draft
                >= self.req_states.prefill_len.np[all_slots]
            )
        )
        scored = [
            (score, k)
            for k in candidates
            if k <= fresh_width
            and (
                score := score_k(
                    self,
                    survival,
                    k,
                    batch_size,
                    num_reqs,
                    non_draft,
                    num_sampling_requests,
                )
            )
            is not None
        ]
        if not scored:
            return None
        score, selected = max(scored)
        bucket = _batch_bucket(batch_size)
        current = self._physical_k_last_by_bucket.get(bucket, max_k)
        if current <= fresh_width and current != selected:
            current_score = score_k(
                self,
                survival,
                current,
                batch_size,
                num_reqs,
                non_draft,
                num_sampling_requests,
            )
            if current_score is not None and score < current_score * 1.02:
                score, selected = current_score, current
        self._physical_k_last_by_bucket[bucket] = selected
        if self._physical_k_last_logged.get(bucket) != selected:
            logger.info(
                "ASCEND_AV_PHYSICAL_K_DECISION batch_bucket=%d physical_k=%d "
                "score=%.6f fresh_width=%d",
                bucket,
                selected,
                score,
                fresh_width,
            )
            self._physical_k_last_logged[bucket] = selected
        return batch_size, selected, score

    def get_num_tokens(self, num_tokens_per_req, draft_tokens) -> int:
        result = original_get_num_tokens(num_tokens_per_req, draft_tokens)
        self._physical_k_recommendation = recommend(self, num_tokens_per_req, draft_tokens)
        return result

    manager.batches_to_profile = MethodType(batches_to_profile, manager)
    manager.set_initial_cost_curves = MethodType(set_initial_cost_curves, manager)
    manager.get_num_tokens = MethodType(get_num_tokens, manager)
    return manager
