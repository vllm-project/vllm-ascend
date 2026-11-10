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
    PHYSICAL_K_MAX_TP_SIZE,
    PHYSICAL_K_MIN_TUNED_BATCH_SIZE,
    PHYSICAL_K_RECOMMEND_INTERVAL,
    physical_k_runtime_enabled,
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
    if not physical_k_runtime_enabled(vllm_config):
        return ()
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
    """Take three timing replays per shape for a cold-start-robust median.

    Two observations let one slow first replay inflate the median by half of
    its cost. A third replay makes the same median insensitive to one cold
    graph/allocator outlier without changing upstream max-K profiling.
    """

    counts: dict[tuple[tuple[str, int], ...], int] = defaultdict(int)
    first_by_shape: dict[tuple[tuple[str, int], ...], dict[str, int]] = {}
    result: list[dict[str, int]] = []
    for batch in base:
        key = tuple(sorted(batch.items()))
        first_by_shape.setdefault(key, batch)
        if counts[key] >= 3:
            continue
        counts[key] += 1
        result.append(batch)
    for key, batch in first_by_shape.items():
        result.extend(dict(batch) for _ in range(3 - counts[key]))
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
        for _ in range(3):
            result.append({**template, "num_tokens": num_tokens})
            is_upstream.append(False)
    return result, is_upstream


def configure_physical_k_profiling(manager: Any, vllm_config: Any):
    """Add physical-K samples and recommendations to an upstream AV manager."""

    dynamic = (getattr(vllm_config, "additional_config", None) or {}).get("dynamic_spec_config", {})
    params = resolve_physical_k(dynamic)
    if params is not None and not physical_k_runtime_enabled(vllm_config):
        tp_size = int(
            getattr(
                getattr(vllm_config, "parallel_config", None),
                "tensor_parallel_size",
                1,
            )
        )
        if tp_size > PHYSICAL_K_MAX_TP_SIZE:
            logger.info(
                "ASCEND_AV_PHYSICAL_K_FALLBACK tensor_parallel_size=%d max_supported_tp=%d mode=adaptive_verification",
                tp_size,
                PHYSICAL_K_MAX_TP_SIZE,
            )
        else:
            logger.info(
                "ASCEND_AV_PHYSICAL_K_NOOP min_k=%d configured_k=%d mode=adaptive_verification",
                params["min_k"],
                int(
                    getattr(
                        getattr(vllm_config, "speculative_config", None),
                        "num_speculative_tokens",
                        0,
                    )
                ),
            )
        return manager
    max_k = int(manager.num_speculative_steps)
    candidates = _candidate_k(vllm_config, max_k)
    if not candidates:
        return manager
    original_batches = manager.batches_to_profile
    original_set_curves = manager.set_initial_cost_curves
    original_get_num_tokens = manager.get_num_tokens
    original_record_confidences = manager.record_confidences
    min_batch_size = PHYSICAL_K_MIN_TUNED_BATCH_SIZE
    manager._physical_k_profile_cases = []
    manager._physical_k_profile_is_upstream = []
    manager._physical_k_draft_costs = None
    manager._physical_k_recommendation = None
    manager._physical_k_cost_saving = None
    manager._physical_k_last_logged = {}
    manager._physical_k_score_steps = defaultdict(int)
    manager._physical_k_seen_narrow_confidence = defaultdict(bool)
    manager._physical_k_global_confidence_widths = np.zeros(int(manager.req_states.max_num_reqs), dtype=np.int32)
    manager._physical_k_confidence_widths = [
        np.zeros(int(manager.req_states.max_num_reqs), dtype=np.int32) for _ in manager._stale_confidences
    ]
    manager._physical_k_single_request_id = None
    manager._physical_k_score_new_request = False

    def batches_to_profile(self, capture_sizes) -> Iterator[dict[str, int]]:
        base = list(original_batches(capture_sizes))
        covered, covered_is_upstream = _extend_profile_batches(
            base,
            int(getattr(self.req_states, "max_num_reqs", 0)),
            min_batch_size,
        )
        sparse = _sparse_profile_batches([case for case in covered if case["num_tokens"] >= min_batch_size])
        self._physical_k_profile_cases.clear()
        self._physical_k_profile_is_upstream.clear()
        per_width_cases: dict[int, list[tuple[dict[str, int], bool]]] = {}
        for physical_k in candidates:
            # Preserve the complete upstream max-K profile so its verify-cost
            # curves are unchanged. Lower physical widths only need robust
            # medians for each distinct graph shape.
            batches = covered if physical_k == max_k else sparse
            upstream_flags = covered_is_upstream if physical_k == max_k else [False] * len(batches)
            per_width_cases[physical_k] = list(zip(batches, upstream_flags))
        # A sequential K sweep confounds width with device warmup/clock drift.
        # Interleave nearby candidate timings while keeping the max-K samples
        # in their original relative order for upstream AV cost curves.
        for index in range(max(map(len, per_width_cases.values()))):
            for physical_k in candidates:
                if index >= len(per_width_cases[physical_k]):
                    continue
                batch, is_upstream = per_width_cases[physical_k][index]
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
                gathered: list[dict[int, dict[int, float]] | None] = [None] * tp_group.world_size
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

    def record_confidences(self, confidence_probs, input_batch) -> None:
        # Native record flips the readable index before writing the current
        # snapshot. Tag its write buffer, without adding a copy or event wait.
        write_idx = self._stale_idx
        if self._pending_resets:
            self._physical_k_global_confidence_widths[self._pending_resets] = 0
            for widths in self._physical_k_confidence_widths:
                widths[self._pending_resets] = 0
        original_record_confidences(confidence_probs, input_batch)
        produced_k = getattr(input_batch, "_vllm_ascend_produced_confidence_k", 0)
        if type(produced_k) is not int or not 0 < produced_k <= max_k:
            produced_k = 0
        slots = input_batch.idx_mapping_np[: input_batch.num_reqs]
        self._physical_k_global_confidence_widths[slots] = produced_k
        # Native snapshots the entire global confidence table, including
        # requests absent from this batch. Its CPU provenance must do so too.
        np.copyto(self._physical_k_confidence_widths[write_idx], self._physical_k_global_confidence_widths)

    def readable_confidence_width(self, req_ids, draft_tokens) -> int | None:
        widths = self._physical_k_confidence_widths[self._stale_idx]
        available = []
        for req_id in req_ids:
            if not draft_tokens.get(req_id):
                continue
            slot = self.req_states.req_id_to_index[req_id]
            # Untracked slots, including requests beyond measured coverage,
            # cannot establish the provenance of a full-width prediction.
            if not 0 <= slot < len(widths):
                return 0
            available.append(int(widths[slot]))
        return min(available, default=None)

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
        verify_costs = verify_cost[non_draft : non_draft + max_budget + 1]
        # Native AV still selects its budget using the unchanged max-K draft
        # curve. Price the budget it will actually execute, rather than an
        # independently optimized budget based on this candidate's draft cost.
        av_draft_cost = self.cost_tables[0][num_reqs]
        budget = int(np.argmax(expected / (av_draft_cost + verify_costs)))
        return float(expected[budget] / (draft_cost + verify_costs[budget]))

    def recommend(self, num_tokens_per_req, draft_tokens):
        if self._physical_k_draft_costs is None or self.cost_tables is None:
            return None
        req_ids = list(num_tokens_per_req)
        num_reqs = len(req_ids)
        active = [req_id for req_id in req_ids if draft_tokens.get(req_id)]
        batch_size = len(active)
        if not batch_size:
            return None
        # Only non-empty active batches are scored. Cost tables cover every
        # power-of-two request bucket, including small RL rollout tails.
        if batch_size < min_batch_size:
            return None
        if not _is_tp_rank_zero():
            return None
        # Current drafts and CPU confidence snapshots belong to different
        # producer steps. Only the readable buffer's own label can establish
        # that all candidate prefixes were actually evaluated at full width.
        fresh_width = readable_confidence_width(self, req_ids, draft_tokens)
        if fresh_width is None or fresh_width < max_k:
            return None
        # Require complete measured coverage. Do not prune a narrow K solely
        # by draft cost: wider K can make native AV choose a more expensive
        # verification budget even when the two draft graphs cost the same.
        if any(lookup_cost(self, k, batch_size) is None for k in candidates):
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
            (num_tokens_per_req[req_id] - len(draft_tokens.get(req_id, ())) for req_id in req_ids),
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
        full_cost = lookup_cost(self, max_k, batch_size)
        self._physical_k_cost_saving = next(
            (
                k
                for k in candidates
                if k < max_k
                and (cost := lookup_cost(self, k, batch_size)) is not None
                and full_cost is not None
                and cost <= 0.75 * full_cost
            ),
            None,
        )
        bucket = _batch_bucket(batch_size)
        if self._physical_k_last_logged.get(bucket) != selected:
            logger.info(
                "ASCEND_AV_PHYSICAL_K_DECISION batch_bucket=%d physical_k=%d "
                "score=%.6f confidence_width=%d candidate_scores=%s",
                bucket,
                selected,
                score,
                fresh_width,
                {k: round(value, 6) for value, k in scored},
            )
            self._physical_k_last_logged[bucket] = selected
        return batch_size, selected

    def get_num_tokens(self, num_tokens_per_req, draft_tokens) -> int:
        result = original_get_num_tokens(num_tokens_per_req, draft_tokens)
        # Avoid request-list construction only for an empty batch. Every
        # non-empty concurrency can now use a bucket-specific recommendation.
        # Confidence scoring is amortized per bucket: score on first entry and
        # then once every N decode steps, while scheduler acceptance feedback
        # continues to update the controller on every step.
        if len(num_tokens_per_req) < min_batch_size:
            self._physical_k_recommendation = None
            self._physical_k_cost_saving = None
        else:
            if len(num_tokens_per_req) == 1:
                request_id = next(iter(num_tokens_per_req))
                if request_id != self._physical_k_single_request_id:
                    # The scheduler resets the width when this new request's
                    # first output arrives. Price its own confidence on the
                    # first subsequent full-width draft, not 128 steps later.
                    self._physical_k_single_request_id = request_id
                    self._physical_k_score_new_request = True
            else:
                self._physical_k_single_request_id = None
                self._physical_k_score_new_request = False
            bucket = _batch_bucket(len(num_tokens_per_req))
            steps = self._physical_k_score_steps[bucket]
            self._physical_k_score_steps[bucket] = steps + 1
            probe_due = False
            confidence_width = readable_confidence_width(self, num_tokens_per_req, draft_tokens)
            if confidence_width is not None:
                if confidence_width < max_k:
                    self._physical_k_seen_narrow_confidence[bucket] = True
                elif self._physical_k_seen_narrow_confidence[bucket]:
                    # A full probe becomes CPU-readable on a later record
                    # call, even if the newly published drafts are narrow.
                    probe_due = True
                    self._physical_k_seen_narrow_confidence[bucket] = False
            new_request_ready = (
                self._physical_k_score_new_request
                and len(num_tokens_per_req) == 1
                and confidence_width is not None
                and confidence_width >= max_k
            )
            if steps % PHYSICAL_K_RECOMMEND_INTERVAL == 0 or probe_due or new_request_ready:
                self._physical_k_cost_saving = None
                self._physical_k_recommendation = recommend(self, num_tokens_per_req, draft_tokens)
                if self._physical_k_recommendation is not None:
                    self._physical_k_score_new_request = False
                if self._physical_k_recommendation is not None and self._physical_k_recommendation[1] == max_k:
                    self._physical_k_seen_narrow_confidence[bucket] = False
            else:
                self._physical_k_recommendation = None
                self._physical_k_cost_saving = None
        return result

    manager.batches_to_profile = MethodType(batches_to_profile, manager)
    manager.set_initial_cost_curves = MethodType(set_initial_cost_curves, manager)
    manager.get_num_tokens = MethodType(get_num_tokens, manager)
    manager.record_confidences = MethodType(record_confidences, manager)
    return manager
