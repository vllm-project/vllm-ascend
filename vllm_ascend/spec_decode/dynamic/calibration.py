# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Calibration for confidence estimates used by dynamic speculation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch


@dataclass(frozen=True)
class SequentialTemperatureScaler:
    """Apply one temperature per draft position.

    DSpark confidence values are conditional probabilities.  Consequently a
    single global calibration scalar is often insufficient: a small bias at an
    early position is multiplied into every later prefix probability.  The
    profile stores one positive temperature for each position.  Temperatures
    are applied to logits so the ordering of candidates is preserved.
    """

    temperatures: tuple[float, ...]
    _tensor_cache: dict[tuple[str, torch.dtype], torch.Tensor] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.temperatures:
            return
        if any(t <= 0.0 for t in self.temperatures):
            raise ValueError("confidence calibration temperatures must be > 0")

    @classmethod
    def identity(cls, num_positions: int) -> "SequentialTemperatureScaler":
        return cls(tuple(1.0 for _ in range(num_positions)))

    @classmethod
    def from_config(
        cls,
        values: Sequence[float] | None,
        num_positions: int,
    ) -> "SequentialTemperatureScaler":
        if values is None:
            return cls.identity(num_positions)
        temperatures = tuple(float(value) for value in values)
        if len(temperatures) != num_positions:
            raise ValueError(
                "confidence_temperatures must contain exactly "
                f"{num_positions} values, got {len(temperatures)}"
            )
        return cls(temperatures)

    def _temperature_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), dtype)
        tensor = self._tensor_cache.get(key)
        if tensor is None:
            tensor = torch.as_tensor(self.temperatures, device=device, dtype=dtype)
            self._tensor_cache[key] = tensor
        return tensor

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated logits with a final draft-position dimension."""
        if logits.shape[-1] > len(self.temperatures):
            raise ValueError(
                "confidence logits last dimension cannot exceed calibration "
                f"length ({len(self.temperatures)}), got {logits.shape[-1]}"
            )
        temperatures = self._temperature_tensor(
            device=logits.device, dtype=logits.dtype
        )[: logits.shape[-1]]
        return logits / temperatures

    def calibrate_probabilities(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Calibrate probabilities whose final dimension is draft position."""
        if probabilities.shape[-1] > len(self.temperatures):
            raise ValueError(
                "confidence probabilities last dimension cannot exceed calibration "
                f"length ({len(self.temperatures)}), got {probabilities.shape[-1]}"
            )
        if all(t == 1.0 for t in self.temperatures):
            return probabilities
        eps = torch.finfo(probabilities.dtype).eps
        clipped = probabilities.clamp(min=eps, max=1.0 - eps)
        logits = torch.log(clipped) - torch.log1p(-clipped)
        return torch.sigmoid(self.calibrate_logits(logits))

    @classmethod
    def fit(
        cls,
        predicted_logits: torch.Tensor,
        accepted_prefix: torch.Tensor,
        *,
        temperatures: Sequence[float] | None = None,
    ) -> "SequentialTemperatureScaler":
        """Fit temperatures with a small CPU-side grid search.

        This helper is intended for offline profile generation, not the decode
        hot path. ``accepted_prefix`` is a boolean tensor shaped like
        ``predicted_logits`` and contains whether each conditional draft token
        survived target verification.  The objective is binary NLL of the
        conditional events.  A caller can pass a custom grid; the default is
        deliberately conservative to avoid overfitting small calibration sets.
        """
        if predicted_logits.shape != accepted_prefix.shape:
            raise ValueError("predicted_logits and accepted_prefix must have the same shape")
        if predicted_logits.ndim != 2:
            raise ValueError("calibration inputs must have shape [samples, positions]")
        num_positions = predicted_logits.shape[1]
        grid = tuple(float(x) for x in (temperatures or (0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0)))
        if any(x <= 0 for x in grid):
            raise ValueError("temperature grid values must be > 0")

        logits = predicted_logits.detach().float().cpu()
        labels = accepted_prefix.detach().float().cpu()
        fitted: list[float] = []
        for position in range(num_positions):
            position_logits = logits[:, position]
            position_labels = labels[:, position]
            best_temperature = grid[0]
            best_loss = float("inf")
            for temperature in grid:
                probabilities = torch.sigmoid(position_logits / temperature).clamp(1e-6, 1.0 - 1e-6)
                loss = torch.nn.functional.binary_cross_entropy(probabilities, position_labels).item()
                if loss < best_loss:
                    best_loss = loss
                    best_temperature = temperature
            fitted.append(best_temperature)
        return cls(tuple(fitted))
