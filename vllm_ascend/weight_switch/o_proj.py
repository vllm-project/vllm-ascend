# SPDX-License-Identifier: Apache-2.0
"""O-proj compatibility adapter for the generic weight-switch controller."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

from .controller import (
    WeightSwitchControllerMixin,
    WeightSwitchHandle,
    WeightSwitchTarget,
)
from .linear import WeightSwitchConfig, WeightSwitchMixin


# These aliases preserve existing O-proj backend declarations while sharing
# the generic target/handle implementation with every future weight type.
OProjWeightSwitchLayer = WeightSwitchTarget
OProjWeightSwitchHandle = WeightSwitchHandle


class OProjWeightSwitchMixin(WeightSwitchControllerMixin):
    """Compatibility adapter exposing the original O-proj lifecycle API."""

    @property
    def o_proj_weight_switch_config(self) -> WeightSwitchConfig:
        return self.weight_switch_config

    @o_proj_weight_switch_config.setter
    def o_proj_weight_switch_config(self, config: WeightSwitchConfig) -> None:
        self.weight_switch_config = config

    @property
    def _o_proj_weight_switch_enabled(self) -> bool:
        return self._weight_switch_enabled

    @_o_proj_weight_switch_enabled.setter
    def _o_proj_weight_switch_enabled(self, enabled: bool) -> None:
        self._weight_switch_enabled = enabled

    @property
    def _o_proj_weight_switch_handles(self) -> dict[str, OProjWeightSwitchHandle]:
        return self._weight_switch_handles

    @_o_proj_weight_switch_handles.setter
    def _o_proj_weight_switch_handles(
        self,
        handles: dict[str, OProjWeightSwitchHandle],
    ) -> None:
        self._weight_switch_handles = handles

    def _initialize_o_proj_weight_switch(
        self,
        config: WeightSwitchConfig,
    ) -> None:
        self._initialize_weight_switch(config)

    def _get_o_proj_weight_switch_layers(
        self,
    ) -> tuple[OProjWeightSwitchLayer, ...]:
        raise NotImplementedError

    def _get_weight_switch_targets(self) -> tuple[WeightSwitchTarget, ...]:
        return self._get_o_proj_weight_switch_layers()

    def _get_o_proj_weight_switch_pool(self) -> dict[Any, torch.Tensor]:
        return self.o_proj_full_pools

    def _get_weight_switch_pool(self) -> dict[Any, torch.Tensor]:
        return self._get_o_proj_weight_switch_pool()

    def _get_o_proj_weight_switch_method(self) -> WeightSwitchMixin:
        return self._get_single_weight_switch_method()

    def _enable_o_proj_full_weight_switch(self) -> None:
        self._enable_full_weight_switch()

    def _after_o_proj_weight_switch_enabled(
        self,
        handles: dict[str, OProjWeightSwitchHandle],
    ) -> None:
        """Hook for backend-specific O-proj compatibility aliases."""

    def _after_weight_switch_enabled(
        self,
        handles: dict[str, WeightSwitchHandle],
    ) -> None:
        self._after_o_proj_weight_switch_enabled(handles)

    def _get_o_proj_weight_switch_handle(
        self,
        name: str,
    ) -> OProjWeightSwitchHandle:
        return self._get_weight_switch_handle(name)

    def _all_gather_o_proj_full_weight(self) -> None:
        self._all_gather_full_weights()

    def _maybe_all_gather_o_proj_full_weight(self, enabled: bool) -> None:
        self._maybe_all_gather_full_weights(enabled)

    def _switch_o_proj_to_full_weight(self) -> None:
        self._switch_to_full_weights()

    def _switch_o_proj_to_local_weight(self) -> None:
        self._switch_to_local_weights()

    @contextmanager
    def _use_full_o_proj_weights(self) -> Iterator[None]:
        with self._use_full_weights():
            yield
