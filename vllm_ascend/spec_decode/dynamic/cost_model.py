# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small, immutable cost tables for Ascend verification steps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


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

    def sps(self, token_batch_size: int) -> float:
        return 1000.0 / self.latency(token_batch_size)
