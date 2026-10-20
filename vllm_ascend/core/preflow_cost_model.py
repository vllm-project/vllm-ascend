# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profiled prefill-cost model used by the PREFLOW scheduler."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PREFLOW_PROFILE_METADATA_ATTR = "_vllm_ascend_preflow_profile_metadata"
PREFLOW_PROFILE_ELAPSED_MS_ATTR = "_vllm_ascend_preflow_profile_elapsed_ms"
PREFLOW_CALIBRATION_REQUEST_ATTR = "_vllm_ascend_preflow_calibration_request"
PREFLOW_CALIBRATION_REQUEST_PREFIX = "__vllm_ascend_preflow_calibration__"

_MIN_COST = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class PreflowProfileSample:
    """One startup-profile observation."""

    history: int
    chunk_size: int
    elapsed_ms: float
    is_final: bool
    discard: bool = False


@dataclass(frozen=True)
class ProfiledPrefillCostModel:
    """Additive, history-aware prefill runtime model measured in milliseconds.

    Full chunks use a quadratic in normalized history. Short final chunks use
    monotone interpolation between measured sizes plus the history-dependent
    part of the full-chunk curve.
    """

    chunk_size: int
    history_scale: float
    theta_0: float
    theta_1: float
    theta_2: float
    short_chunk_sizes: tuple[int, ...]
    short_chunk_costs: tuple[float, ...]
    final_overhead: float

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        scalar_parameters = (
            self.history_scale,
            self.theta_0,
            self.theta_1,
            self.theta_2,
            self.final_overhead,
        )
        if not all(math.isfinite(value) for value in scalar_parameters):
            raise ValueError("profiled cost-model parameters must be finite")
        if self.history_scale <= 0:
            raise ValueError("history_scale must be positive")
        if self.final_overhead < 0:
            raise ValueError("final_overhead must be non-negative")
        if not self.short_chunk_sizes or len(self.short_chunk_sizes) != len(self.short_chunk_costs):
            raise ValueError("short chunk sizes and costs must be non-empty and have equal length")
        if any(lower >= upper for lower, upper in zip(self.short_chunk_sizes, self.short_chunk_sizes[1:])):
            raise ValueError("short chunk sizes must be strictly increasing")
        if self.short_chunk_sizes[0] <= 0:
            raise ValueError("short chunk sizes must be positive")
        if self.short_chunk_sizes[-1] != self.chunk_size:
            raise ValueError("short chunk observations must end at chunk_size")
        if any(not math.isfinite(cost) or cost <= 0 for cost in self.short_chunk_costs):
            raise ValueError("short chunk costs must be finite and positive")
        if any(lower > upper for lower, upper in zip(self.short_chunk_costs, self.short_chunk_costs[1:])):
            raise ValueError("short chunk costs must be non-decreasing")

    def _full_chunk_cost(self, history: int | float) -> float:
        normalized_history = max(0.0, float(history)) / self.history_scale
        return max(
            _MIN_COST,
            self.theta_0 + self.theta_1 * normalized_history + self.theta_2 * normalized_history * normalized_history,
        )

    def _short_chunk_cost(self, chunk_size: int) -> float:
        if chunk_size <= 0:
            return 0.0
        if chunk_size <= self.short_chunk_sizes[0]:
            return self.short_chunk_costs[0]
        for index, upper_size in enumerate(self.short_chunk_sizes[1:], start=1):
            if chunk_size <= upper_size:
                lower_size = self.short_chunk_sizes[index - 1]
                fraction = (chunk_size - lower_size) / (upper_size - lower_size)
                lower_cost = self.short_chunk_costs[index - 1]
                return lower_cost + fraction * (self.short_chunk_costs[index] - lower_cost)
        return self.short_chunk_costs[-1]

    def chunk_cost(self, history: int, chunk_size: int, *, is_final: bool = False) -> float:
        history = max(0, int(history))
        chunk_size = max(0, int(chunk_size))
        if chunk_size == 0:
            return 0.0
        if chunk_size == self.chunk_size:
            cost = self._full_chunk_cost(history)
        elif chunk_size < self.chunk_size:
            history_adjustment = self._full_chunk_cost(history) - self._full_chunk_cost(0)
            cost = self._short_chunk_cost(chunk_size) + chunk_size / self.chunk_size * history_adjustment
        else:
            # Preserve additivity for an unusual runtime budget larger than the
            # calibrated chunk instead of extrapolating the sparse short curve.
            cost = self.interval_cost(history, history + chunk_size, include_final=False)
        if is_final:
            cost += self.final_overhead
        return max(_MIN_COST, cost)

    def interval_cost(self, start_history: int, end_history: int, *, include_final: bool = True) -> float:
        start = max(0, int(start_history))
        end = max(start, int(end_history))
        distance = end - start
        if distance == 0:
            return 0.0

        full_chunks, tail = divmod(distance, self.chunk_size)
        total = 0.0
        if full_chunks:
            count = float(full_chunks)
            step = float(self.chunk_size)
            base = float(start)
            history_sum = count * base + step * count * (count - 1.0) / 2.0
            history_squared_sum = (
                count * base * base
                + base * step * count * (count - 1.0)
                + step * step * count * (count - 1.0) * (2.0 * count - 1.0) / 6.0
            )
            total += (
                count * self.theta_0
                + self.theta_1 * history_sum / self.history_scale
                + self.theta_2 * history_squared_sum / (self.history_scale * self.history_scale)
            )
        if tail:
            total += self.chunk_cost(start + full_chunks * self.chunk_size, tail)
        if include_final:
            total += self.final_overhead
        return max(0.0, total)


def _project_monotone_quadratic_slopes(
    linear: float,
    quadratic: float,
    maximum_normalized_history: float,
) -> tuple[float, float]:
    """Project coefficients onto a non-negative derivative over the domain."""
    if linear >= 0.0 and linear + 2.0 * quadratic * maximum_normalized_history >= 0.0:
        return linear, quadratic

    candidates: list[tuple[float, float]] = [(0.0, 0.0), (0.0, max(0.0, quadratic))]
    boundary_normal = (1.0, 2.0 * maximum_normalized_history)
    denominator = boundary_normal[0] ** 2 + boundary_normal[1] ** 2
    projection = (linear * boundary_normal[0] + quadratic * boundary_normal[1]) / denominator
    boundary_linear = linear - projection * boundary_normal[0]
    boundary_quadratic = quadratic - projection * boundary_normal[1]
    if boundary_linear >= 0.0:
        candidates.append((boundary_linear, boundary_quadratic))
    return min(
        candidates,
        key=lambda value: (value[0] - linear) ** 2 + (value[1] - quadratic) ** 2,
    )


def fit_profiled_prefill_cost_model(
    samples: list[PreflowProfileSample],
    *,
    chunk_size: int,
    calibration_history: int,
    maximum_history: int,
) -> ProfiledPrefillCostModel:
    """Fit a deterministic monotone cost model from startup measurements."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    usable = [sample for sample in samples if not sample.discard]
    if any(not math.isfinite(sample.elapsed_ms) or sample.elapsed_ms <= 0 for sample in usable):
        raise ValueError("profile samples must have finite, positive elapsed times")

    full_nonfinal = [sample for sample in usable if sample.chunk_size == chunk_size and not sample.is_final]
    if len(full_nonfinal) < 3:
        raise RuntimeError(
            f"PREFLOW profiling needs at least three measured non-final full chunks; collected {len(full_nonfinal)}."
        )

    history_scale = float(max(1, calibration_history))
    full_by_history: dict[int, list[float]] = {}
    for sample in full_nonfinal:
        full_by_history.setdefault(sample.history, []).append(sample.elapsed_ms)
    ordered_histories = sorted(full_by_history)
    histories = np.asarray([history / history_scale for history in ordered_histories], dtype=np.float64)
    latencies = np.asarray(
        [np.median(np.asarray(full_by_history[history], dtype=np.float64)) for history in ordered_histories],
        dtype=np.float64,
    )
    design = np.column_stack((np.ones_like(histories), histories, histories * histories))
    coefficients, _, _, _ = np.linalg.lstsq(design, latencies, rcond=None)
    theta_0, theta_1, theta_2 = (float(value) for value in coefficients)
    if not all(math.isfinite(value) for value in (theta_0, theta_1, theta_2)):
        raise RuntimeError("PREFLOW profiling produced non-finite full-chunk coefficients.")

    maximum_normalized_history = max(1.0, maximum_history / history_scale)
    theta_1, theta_2 = _project_monotone_quadratic_slopes(
        theta_1,
        theta_2,
        maximum_normalized_history,
    )
    theta_0 = max(
        float(np.mean(latencies - theta_1 * histories - theta_2 * histories * histories)),
        _MIN_COST,
    )

    def full_cost(history: int) -> float:
        normalized_history = max(0.0, float(history)) / history_scale
        return max(
            _MIN_COST,
            theta_0 + theta_1 * normalized_history + theta_2 * normalized_history * normalized_history,
        )

    final_full = [sample for sample in usable if sample.chunk_size == chunk_size and sample.is_final]
    if not final_full:
        raise RuntimeError("PREFLOW profiling did not collect a final full-chunk sample.")
    final_overhead = max(
        0.0,
        float(np.median([sample.elapsed_ms - full_cost(sample.history) for sample in final_full])),
    )

    short_observations: dict[int, list[float]] = {}
    for sample in usable:
        if sample.is_final and sample.history == 0:
            short_observations.setdefault(sample.chunk_size, []).append(
                max(_MIN_COST, sample.elapsed_ms - final_overhead)
            )
    short_observations.setdefault(chunk_size, []).append(full_cost(0))
    if len(short_observations) < 2:
        raise RuntimeError("PREFLOW profiling did not collect enough short-query sizes.")

    sizes = sorted(short_observations)
    raw_costs = [float(np.median(short_observations[size])) for size in sizes]
    anchor = full_cost(0)
    projected_costs: list[float] = []
    running_cost = _MIN_COST
    for size, raw_cost in zip(sizes, raw_costs):
        bounded_cost = min(raw_cost, anchor) if size < chunk_size else anchor
        running_cost = max(running_cost, bounded_cost)
        projected_costs.append(running_cost)
    if sizes[-1] != chunk_size:
        sizes.append(chunk_size)
        projected_costs.append(anchor)
    else:
        projected_costs[-1] = anchor
    for index in range(len(projected_costs) - 2, -1, -1):
        projected_costs[index] = min(projected_costs[index], projected_costs[index + 1])

    return ProfiledPrefillCostModel(
        chunk_size=chunk_size,
        history_scale=history_scale,
        theta_0=theta_0,
        theta_1=theta_1,
        theta_2=theta_2,
        short_chunk_sizes=tuple(sizes),
        short_chunk_costs=tuple(projected_costs),
        final_overhead=final_overhead,
    )


__all__ = [
    "PREFLOW_CALIBRATION_REQUEST_ATTR",
    "PREFLOW_CALIBRATION_REQUEST_PREFIX",
    "PREFLOW_PROFILE_ELAPSED_MS_ATTR",
    "PREFLOW_PROFILE_METADATA_ATTR",
    "PreflowProfileSample",
    "ProfiledPrefillCostModel",
    "fit_profiled_prefill_cost_model",
]
