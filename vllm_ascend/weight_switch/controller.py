# SPDX-License-Identifier: Apache-2.0
"""Backend-side orchestration for switching one or more linear weights."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from .linear import WeightSwitchConfig, WeightSwitchMixin, WeightSwitchState


@dataclass(frozen=True)
class WeightSwitchTarget:
    """One backend-owned linear layer participating in a weight switch."""

    name: str
    layer: torch.nn.Module
    pool_key_suffix: tuple[Any, ...]
    clone_local_tensors: bool = False


@dataclass
class WeightSwitchHandle:
    """The linear method and runtime state for one switched target."""

    layer: torch.nn.Module
    method: WeightSwitchMixin
    state: WeightSwitchState


class WeightSwitchControllerMixin:
    """Reusable backend lifecycle for a set of switched linear weights.

    A backend declares the linear targets, the buffer pool, and when its
    full-weight path is required. The controller owns the common lifecycle:
    allocation, asynchronous all-gather, full/local pointer switching, and
    guaranteed restoration after a full-weight forward.
    """

    def _initialize_weight_switch(self, config: WeightSwitchConfig) -> None:
        """Set the parallel domain before weights are loaded or processed."""
        self.weight_switch_config = config
        self._weight_switch_enabled = False
        self._weight_switch_handles: dict[str, WeightSwitchHandle] = {}

    def _get_weight_switch_targets(self) -> tuple[WeightSwitchTarget, ...]:
        """Return the backend-owned linear targets participating in the switch."""
        raise NotImplementedError

    def _get_weight_switch_pool(self) -> dict[Any, torch.Tensor]:
        """Return reusable full-weight buffers owned by this backend."""
        raise NotImplementedError

    @staticmethod
    def _get_weight_switch_method(layer: torch.nn.Module) -> WeightSwitchMixin:
        quant_method = layer.quant_method
        linear_method = getattr(quant_method, "quant_method", quant_method)
        if (not isinstance(linear_method, WeightSwitchMixin)
                or not linear_method.supports_weight_switch):
            raise RuntimeError(
                "Weight switching requires a weight-switch capable linear "
                f"method, got {type(linear_method).__name__}.")
        return linear_method

    def _get_single_weight_switch_method(self) -> WeightSwitchMixin:
        """Return the method for a controller with exactly one target."""
        targets = self._get_weight_switch_targets()
        if len(targets) != 1:
            raise RuntimeError(
                "A backend with multiple switched layers must use "
                "_get_weight_switch_handle(name).")
        return self._get_weight_switch_method(targets[0].layer)

    def _enable_full_weight_switch(self) -> None:
        """Allocate local/full buffers for every declared linear target."""
        if self._weight_switch_enabled:
            return

        handles: dict[str, WeightSwitchHandle] = {}
        pool = self._get_weight_switch_pool()
        for target in self._get_weight_switch_targets():
            linear_method = self._get_weight_switch_method(target.layer)
            handles[target.name] = WeightSwitchHandle(
                layer=target.layer,
                method=linear_method,
                state=linear_method.enable_weight_switch(
                    target.layer,
                    self.weight_switch_config,
                    pool=pool,
                    pool_key_prefix=(
                        type(linear_method).__qualname__,
                        *target.pool_key_suffix,
                    ),
                    clone_local_tensors=target.clone_local_tensors,
                ),
            )

        self._weight_switch_handles = handles
        self._weight_switch_enabled = True
        self._after_weight_switch_enabled(handles)

    def _after_weight_switch_enabled(
        self,
        handles: dict[str, WeightSwitchHandle],
    ) -> None:
        """Hook for backend-specific compatibility aliases or metadata."""

    def _get_weight_switch_handle(self, name: str) -> WeightSwitchHandle:
        if not self._weight_switch_enabled:
            raise RuntimeError("Weight switching has not been enabled.")
        try:
            return self._weight_switch_handles[name]
        except KeyError as exc:
            raise RuntimeError(
                f"Weight switch has no target named {name!r}.") from exc

    def _all_gather_full_weights(self) -> None:
        """Start asynchronous full-weight gathers for every target."""
        self._enable_full_weight_switch()
        for handle in self._weight_switch_handles.values():
            handle.method.all_gather_weight(handle.state,
                                            self.weight_switch_config)

    def _maybe_all_gather_full_weights(self, enabled: bool) -> None:
        """Start gathers only for a backend-selected full-weight path."""
        if enabled:
            self._all_gather_full_weights()

    def _switch_to_full_weights(self) -> None:
        """Wait for gathers, then expose full weights for a forward."""
        if not self._weight_switch_enabled:
            raise RuntimeError("Weight switching has not been enabled.")
        for handle in self._weight_switch_handles.values():
            handle.method.wait_weight_all_gather(handle.state)
        for handle in self._weight_switch_handles.values():
            handle.method.switch_weight(handle.layer,
                                        handle.state,
                                        use_full_weight=True)

    def _switch_to_local_weights(self) -> None:
        """Restore every switched linear to its local weight storage."""
        if not self._weight_switch_enabled:
            raise RuntimeError("Weight switching has not been enabled.")
        for handle in self._weight_switch_handles.values():
            handle.method.switch_weight(handle.layer,
                                        handle.state,
                                        use_full_weight=False)

    @contextmanager
    def _use_full_weights(self) -> Iterator[None]:
        """Temporarily expose full weights and always restore local weights."""
        self._switch_to_full_weights()
        try:
            yield
        finally:
            self._switch_to_local_weights()
