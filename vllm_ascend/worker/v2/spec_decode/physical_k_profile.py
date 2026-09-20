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

from vllm_ascend.dynamic_spec import (
    PHYSICAL_K_MIN_TUNED_BATCH_SIZE,
    resolve_physical_k,
)

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
    if not params.get("auto_tune"):
        return ()
    values = range(int(params.get("min_k", 3)), max_k + 1)
    return tuple(sorted({max(1, min(int(k), max_k)) for k in values} | {max_k}))


def _is_tp_rank_zero() -> bool:
    """Only the output-producing TP rank needs to score runtime K."""

    try:
        from vllm.distributed.parallel_state import get_tp_group

        return int(get_tp_group().rank_in_group) == 0
    except (AssertionError, AttributeError, RuntimeError):
        # Unit tests and non-distributed startup do not initialize a TP group.
        return True


def _sparse_profile_batches(base: list[dict[str, int]]) -> list[dict[str, int]]:
    """Keep two timing replays per shape for non-max-K candidates."""

    counts: dict[tuple[tuple[str, int], ...], int] = defaultdict(int)
    result: list[dict[str, int]] = []
    for batch in base:
        key = tuple(sorted(batch.items()))
        if counts[key] >= 2:
            continue
        counts[key] += 1
        result.append(batch)
    return result


def _extend_profile_batches(
    base: list[dict[str, int]],
    max_num_reqs: int,
    min_batch_size: int,
) -> tuple[list[dict[str, int]], list[bool]]:
    """Add sparse power-of-two request counts through max_num_reqs.

    Upstream AV already derives most of these cases from graph capture sizes.
    Explicitly filling gaps makes coverage independent of a particular capture
    list. Samples that cannot use a FULL graph are harmless: they remain useful
    upstream warmups but are excluded from physical-K draft pricing.
    """

    result = list(base)
    is_upstream = [True] * len(base)
    if not base or max_num_reqs < min_batch_size:
        return result, is_upstream
    existing = {int(batch["num_tokens"]) for batch in base}
    wanted: set[int] = {max_num_reqs}
    size = _batch_bucket(min_batch_size)
    while size < max_num_reqs:
        wanted.add(size)
        size *= 2
    template = base[0]
    for num_tokens in sorted(wanted - existing):
        for _ in range(2):
            result.append({**template, "num_tokens": num_tokens})
            is_upstream.append(False)
    return result, is_upstream


def configure_physical_k_profiling(manager: Any, vllm_config: Any):
    """Add physical-K samples and recommendations to an upstream AV manager."""

    max_k = int(manager.num_speculative_steps)
    candidates = _candidate_k(vllm_config, max_k)
    if not candidates:
        return manager
    original_batches = manager.batches_to_profile
    original_set_curves = manager.set_initial_cost_curves
    original_get_num_tokens = manager.get_num_tokens
    min_batch_size = PHYSICAL_K_MIN_TUNED_BATCH_SIZE
    manager._physical_k_profile_cases = []
    manager._physical_k_profile_is_upstream = []
    manager._physical_k_draft_costs = None
    manager._physical_k_recommendation = None
    manager._physical_k_last_logged = {}

    def batches_to_profile(self, capture_sizes) -> Iterator[dict[str, int]]:
        base = list(original_batches(capture_sizes))
        covered, covered_is_upstream = _extend_profile_batches(
            base,
            int(getattr(self.req_states, "max_num_reqs", 0)),
            min_batch_size,
        )
        sparse = _sparse_profile_batches(
            [case for case in covered if case["num_tokens"] >= min_batch_size]
        )
        self._physical_k_profile_cases.clear()
        self._physical_k_profile_is_upstream.clear()
        for physical_k in candidates:
            # Preserve the complete upstream max-K profile so its verify-cost
            # curves are unchanged. Lower physical widths only need robust
            # medians for each distinct graph shape.
            batches = covered if physical_k == max_k else sparse
            upstream_flags = (
                covered_is_upstream
                if physical_k == max_k
                else [False] * len(batches)
            )
            for batch, is_upstream in zip(batches, upstream_flags):
                case = dict(batch)
                case["profile_physical_k"] = physical_k
                self._physical_k_profile_cases.append(physical_k)
                self._physical_k_profile_is_upstream.append(is_upstream)
                yield case

    def set_initial_cost_curves(self, samples) -> None:
        cases = self._physical_k_profile_cases
        upstream_cases = self._physical_k_profile_is_upstream
        if len(samples) != len(cases) or len(upstream_cases) != len(cases):
            raise RuntimeError(
                "physical-K profile mismatch: "
                f"{len(samples)} timings, {len(upstream_cases)} flags, "
                f"and {len(cases)} cases"
            )
        full_samples = [sample for sample, upstream in zip(samples, upstream_cases) if upstream]
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
        profiled_max = min(
            (max(buckets, default=0) for buckets in costs.values()),
            default=0,
        )
        required_max = _batch_bucket(int(getattr(self.req_states, "max_num_reqs", 1)))
        if profiled_max < required_max:
            logger.warning(
                "ASCEND_AV_PHYSICAL_K_COVERAGE profiled_max_batch=%d "
                "configured_max_batch=%d action=full_k_outside_profile",
                profiled_max,
                required_max,
            )
        logger.info("ASCEND_AV_PHYSICAL_K_COSTS candidates=%s costs=%s", candidates, dict(costs))

    def lookup_cost(self, physical_k: int, batch_size: int) -> float | None:
        table = self._physical_k_draft_costs
        if not table or physical_k not in table:
            return None
        points = table[physical_k]
        xs = np.asarray(sorted(points), dtype=np.float64)
        ys = np.asarray([points[int(x)] for x in xs], dtype=np.float64)
        bucket = _batch_bucket(batch_size)
        # Never price a large RL batch with the edge of a smaller profile.
        # The safe behavior outside measured coverage is full physical K.
        if not len(xs) or bucket < xs[0] or bucket > xs[-1]:
            return None
        return float(np.interp(bucket, xs, ys))

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

    def cost_floor(self, batch_size: int) -> int | None:
        """Return the shortest K not dominated by a wider draft graph.

        A wider K that costs no more can emit every proposal available to the
        shorter K, while AV remains free not to verify its tail. Treat a 2%
        cost difference as equivalent to avoid selecting a noisy micro-win.
        """

        measured = {
            k: cost
            for k in candidates
            if (cost := lookup_cost(self, k, batch_size)) is not None
        }
        if len(measured) != len(candidates):
            return None
        for physical_k in candidates:
            cost = measured[physical_k]
            if not any(
                measured[wider_k] <= cost * 1.02
                for wider_k in candidates
                if wider_k > physical_k
            ):
                return physical_k
        return max_k

    def recommend(self, num_tokens_per_req, draft_tokens):
        if self._physical_k_draft_costs is None or self.cost_tables is None:
            return None
        req_ids = list(num_tokens_per_req)
        num_reqs = len(req_ids)
        active = [req_id for req_id in req_ids if draft_tokens.get(req_id)]
        batch_size = len(active)
        if not batch_size:
            return None
        # Small batches deliberately stay at max K. Avoid confidence copies,
        # cumprod, and sorting entirely on this latency-sensitive path.
        if batch_size < min_batch_size:
            return None
        if not _is_tp_rank_zero():
            return None
        fresh_width = min(len(draft_tokens[req_id]) for req_id in active)
        # A narrowed physical step has no fresh confidence for wider
        # candidates. Keep the last full-width recommendation until the next
        # periodic K=max probe instead of creating a self-reinforcing low K.
        if fresh_width < max_k:
            return None
        minimum_useful_k = cost_floor(self, batch_size)
        if minimum_useful_k is None:
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
            if minimum_useful_k <= k <= fresh_width
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
        return batch_size, selected

    def get_num_tokens(self, num_tokens_per_req, draft_tokens) -> int:
        result = original_get_num_tokens(num_tokens_per_req, draft_tokens)
        self._physical_k_recommendation = recommend(self, num_tokens_per_req, draft_tokens)
        return result

    manager.batches_to_profile = MethodType(batches_to_profile, manager)
    manager.set_initial_cost_curves = MethodType(set_initial_cost_curves, manager)
    manager.get_num_tokens = MethodType(get_num_tokens, manager)
    return manager
