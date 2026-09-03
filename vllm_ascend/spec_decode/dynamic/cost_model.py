# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small, immutable cost tables for Ascend verification steps."""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


def _as_int_keyed_mapping(values: Mapping[Any, Any], name: str) -> dict[int, float]:
    result: dict[int, float] = {}
    for key, value in values.items():
        try:
            int_key = int(key)
            result[int_key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must have integer keys and numeric values") from exc
    if any(key <= 0 for key in result):
        raise ValueError(f"{name} keys must be > 0")
    if any(value <= 0 for value in result.values()):
        raise ValueError(f"{name} values must be > 0")
    return result


@dataclass(frozen=True)
class HardwareProfileCollector:
    """Collect a startup latency profile from real verification steps.

    The collector deliberately stays independent of the model runner.  The
    runner supplies ``measure_step(batch_size, verify_k)`` and is responsible
    for executing the actual NPU dummy step.  This makes the collection logic
    unit-testable on CPU and keeps runner-specific graph/input handling out of
    the dynamic scheduler.

    The resulting table is keyed by the physical verification token count
    ``batch_size * (verify_k + 1)``.  The extra token is the target bonus token
    already accounted for by :class:`HardwareAwarePrefixPolicy`.
    """

    batch_sizes: tuple[int, ...]
    verify_token_sizes: tuple[int, ...]
    warmup_runs: int = 1
    measure_runs: int = 3

    def __post_init__(self) -> None:
        if not self.batch_sizes or any(size <= 0 for size in self.batch_sizes):
            raise ValueError("profile batch_sizes must contain positive integers")
        if not self.verify_token_sizes or any(size < 0 for size in self.verify_token_sizes):
            raise ValueError("profile verify_token_sizes must contain non-negative integers")
        if self.warmup_runs < 0:
            raise ValueError("profile warmup_runs must be >= 0")
        if self.measure_runs <= 0:
            raise ValueError("profile measure_runs must be > 0")

    @classmethod
    def from_params(
        cls,
        *,
        max_batch_size: int,
        max_draft_tokens: int,
        max_token_capacity: int | None = None,
        params: Mapping[str, Any],
    ) -> "HardwareProfileCollector":
        """Build a bounded collector from ``dynamic_spec_config`` params."""

        max_batch_size = max(int(max_batch_size), 1)
        max_draft_tokens = max(int(max_draft_tokens), 0)

        raw_batches = params.get("profile_batch_sizes")
        if raw_batches is None:
            # Keep startup cost bounded while still covering small, medium and
            # saturated batches. Users can provide an exact sweep when the
            # deployment has a known concurrency distribution.
            raw_batches = (1, min(4, max_batch_size), min(8, max_batch_size), max_batch_size)
        raw_k = params.get("profile_verify_tokens")
        if raw_k is None:
            raw_k = range(0, max_draft_tokens + 1)

        batch_sizes = sorted({max(1, min(int(size), max_batch_size)) for size in raw_batches})
        verify_token_sizes = sorted({max(0, min(int(size), max_draft_tokens)) for size in raw_k})

        if max_token_capacity is not None:
            capacity = max(int(max_token_capacity), 1)
            smallest_batch = min(batch_sizes)
            verify_token_sizes = [
                verify_k
                for verify_k in verify_token_sizes
                if smallest_batch * (verify_k + 1) <= capacity
            ]
            batch_sizes = [
                batch_size
                for batch_size in batch_sizes
                if all(batch_size * (verify_k + 1) <= capacity for verify_k in verify_token_sizes)
            ]
            verify_token_sizes = [
                verify_k
                for verify_k in verify_token_sizes
                if all(batch_size * (verify_k + 1) <= capacity for batch_size in batch_sizes)
            ]
            if not batch_sizes or not verify_token_sizes:
                raise ValueError(
                    "profile_batch_sizes/profile_verify_tokens exceed the model runner token capacity"
                )

        return cls(
            batch_sizes=tuple(batch_sizes),
            verify_token_sizes=tuple(verify_token_sizes),
            warmup_runs=int(params.get("profile_warmup_runs", 1)),
            measure_runs=int(params.get("profile_measure_runs", 3)),
        )

    def collect(
        self,
        measure_step: Callable[[int, int], float],
        *,
        fingerprint: Mapping[str, Any] | None = None,
        confidence_temperatures: Sequence[float] | None = None,
        source: str = "startup",
    ) -> dict[str, Any]:
        """Measure every configured shape and return a JSON-compatible profile."""

        samples_by_token_count: dict[int, list[float]] = {}
        shape_samples: dict[str, dict[str, Any]] = {}
        for batch_size in self.batch_sizes:
            for verify_k in self.verify_token_sizes:
                token_count = batch_size * (verify_k + 1)
                for _ in range(self.warmup_runs):
                    measure_step(batch_size, verify_k)
                samples = [
                    float(measure_step(batch_size, verify_k))
                    for _ in range(self.measure_runs)
                ]
                if any(sample <= 0 for sample in samples):
                    raise ValueError(
                        f"profile measurement must be > 0, got {samples} for "
                        f"batch={batch_size}, verify_k={verify_k}"
                    )
                samples_by_token_count.setdefault(token_count, []).extend(samples)
                shape_samples[f"{batch_size}x{verify_k}"] = {
                    "batch_size": batch_size,
                    "verify_tokens": verify_k,
                    "token_count": token_count,
                    "latency_ms": statistics.median(samples),
                    "samples_ms": samples,
                }

        payload: dict[str, Any] = {
            "schema_version": 1,
            "profile_kind": "startup_dummy_step",
            "source": source,
            "fingerprint": dict(fingerprint or {}),
            "latency_ms": {
                str(token_count): statistics.median(samples)
                for token_count, samples in sorted(samples_by_token_count.items())
            },
            "shapes": shape_samples,
            "profile_batch_sizes": list(self.batch_sizes),
            "profile_verify_tokens": list(self.verify_token_sizes),
            "profile_warmup_runs": self.warmup_runs,
            "profile_measure_runs": self.measure_runs,
        }
        if confidence_temperatures is not None:
            payload["confidence_temperatures"] = [float(value) for value in confidence_temperatures]
        return payload

    @staticmethod
    def save(payload: Mapping[str, Any], path: str | Path) -> None:
        """Persist a profile atomically so a failed startup cannot corrupt it."""

        profile_path = Path(path)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        # Each TP worker can finish startup profiling concurrently.  A shared
        # ``.tmp`` name lets one worker replace another worker's temporary
        # file, making the profile save fail nondeterministically.
        temporary_path = profile_path.with_name(
            f"{profile_path.name}.tmp.{os.getpid()}"
        )
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, profile_path)


@dataclass(frozen=True)
class HardwareCostModel:
    """Lookup table mapping target verify-token batch size to latency.

    ``latency_ms`` is intentionally the canonical representation.  Profiles
    may provide ``sps`` instead; it is converted to milliseconds at load time.
    The table is sparse because Ascend graph modes commonly have only a small
    set of captured shapes.  Lookup uses the nearest profiled shape at or
    above the requested size, then the largest shape for an over-limit query.
    """

    latency_ms: dict[int, float]
    fingerprint: dict[str, Any]
    confidence_temperatures: tuple[float, ...] = ()
    source: str = "inline"

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        expected_fingerprint: Mapping[str, Any] | None = None,
        strict_fingerprint: bool = True,
    ) -> "HardwareCostModel":
        profile_path = Path(path)
        with profile_path.open(encoding="utf-8") as profile_file:
            payload = json.load(profile_file)
        return cls.from_dict(
            payload,
            expected_fingerprint=expected_fingerprint,
            strict_fingerprint=strict_fingerprint,
            source=str(profile_path),
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_fingerprint: Mapping[str, Any] | None = None,
        strict_fingerprint: bool = True,
        source: str = "inline",
    ) -> "HardwareCostModel":
        if not isinstance(payload, Mapping):
            raise ValueError("hardware profile must be a JSON object")
        # ``offline_DSD_k_tuner`` stores the profile alongside its legacy
        # batch-size schedule. Accepting that complete artifact avoids a
        # needless post-processing step for deployments.
        if "hardware_profile" in payload and not any(
            key in payload for key in ("latency_ms", "sps", "curves")
        ):
            nested_profile = payload["hardware_profile"]
            if not isinstance(nested_profile, Mapping):
                raise ValueError("hardware_profile must be an object")
            payload = nested_profile

        fingerprint = dict(payload.get("fingerprint", {}))
        if expected_fingerprint:
            mismatches = {
                key: (expected_fingerprint[key], fingerprint.get(key))
                for key in expected_fingerprint
                if fingerprint.get(key) != expected_fingerprint[key]
            }
            if mismatches and strict_fingerprint:
                raise ValueError(f"hardware profile fingerprint mismatch: {mismatches}")

        latency_ms: dict[int, float]
        if "latency_ms" in payload:
            raw_latency = payload["latency_ms"]
            if not isinstance(raw_latency, Mapping):
                raise ValueError("latency_ms must be an object keyed by token batch size")
            latency_ms = _as_int_keyed_mapping(raw_latency, "latency_ms")
        elif "sps" in payload:
            raw_sps = payload["sps"]
            if not isinstance(raw_sps, Mapping):
                raise ValueError("sps must be an object keyed by token batch size")
            sps = _as_int_keyed_mapping(raw_sps, "sps")
            latency_ms = {key: 1000.0 / value for key, value in sps.items()}
        else:
            curves = payload.get("curves")
            if not isinstance(curves, Mapping):
                raise ValueError("hardware profile requires latency_ms, sps, or curves")
            curve = curves.get(payload.get("workload", "decode_only"), curves.get("decode_only"))
            if curve is None:
                # Allow a direct graph-mode curve such as curves.FULL when the
                # profile contains only one workload.
                curve = next(iter(curves.values()), None)
            if isinstance(curve, Mapping) and any(isinstance(value, Mapping) for value in curve.values()):
                graph_mode = payload.get("graph_mode")
                curve = curve.get(graph_mode) if graph_mode is not None else next(iter(curve.values()), None)
            if not isinstance(curve, Mapping):
                raise ValueError("curves must contain an integer-keyed latency mapping")
            latency_ms = _as_int_keyed_mapping(curve, "curves")

        if not latency_ms:
            raise ValueError("hardware profile latency curve must not be empty")
        raw_temperatures = payload.get("confidence_temperatures", ())
        if raw_temperatures is None:
            raw_temperatures = ()
        if not isinstance(raw_temperatures, (list, tuple)):
            raise ValueError("confidence_temperatures must be a list")
        confidence_temperatures = tuple(float(value) for value in raw_temperatures)
        if any(value <= 0 for value in confidence_temperatures):
            raise ValueError("confidence_temperatures values must be > 0")
        return cls(
            latency_ms=latency_ms,
            fingerprint=fingerprint,
            confidence_temperatures=confidence_temperatures,
            source=source,
        )

    def latency(self, token_batch_size: int) -> float:
        if token_batch_size <= 0:
            raise ValueError("token_batch_size must be > 0")
        keys = sorted(self.latency_ms)
        for key in keys:
            if key >= token_batch_size:
                return self.latency_ms[key]
        return self.latency_ms[keys[-1]]

    def supports_token_batch(self, token_batch_size: int) -> bool:
        """Return whether the profile covers a requested token batch.

        ``latency`` intentionally keeps the historical nearest-shape lookup
        for compatibility.  Using the largest profile point for a larger
        batch, however, is not a valid hardware estimate: a profile collected
        at batch 4 must not be reused as if it described batch 16.  Callers
        that make a runtime scheduling decision can use this predicate to
        fall back to a safe full-width path instead of paying for a decision
        based on an out-of-range curve.
        """

        if token_batch_size <= 0:
            raise ValueError("token_batch_size must be > 0")
        return token_batch_size <= max(self.latency_ms)

    def sps(self, token_batch_size: int) -> float:
        return 1000.0 / self.latency(token_batch_size)
