# SPDX-License-Identifier: Apache-2.0
"""Single-target O-proj weight-switch lifecycle."""

from collections.abc import Iterator
from contextlib import contextmanager

from .linear import WeightSwitchConfig, WeightSwitchMixin, WeightSwitchState


class OProjWeightSwitchMixin:
    """Manage weight switching for the backend's single ``o_proj`` layer."""

    o_proj_weight_switch_pool_key = "o_proj"

    def _initialize_o_proj_weight_switch(
        self,
        config: WeightSwitchConfig,
    ) -> None:
        self.o_proj_weight_switch_config = config
        self.o_proj_weight_state: WeightSwitchState | None = None
        self._o_proj_weight_switch_enabled = False

    def _get_o_proj_weight_switch_method(self) -> WeightSwitchMixin:
        quant_method = self.o_proj.quant_method
        linear_method = getattr(quant_method, "quant_method", quant_method)
        if not isinstance(linear_method, WeightSwitchMixin) or not linear_method.supports_weight_switch:
            raise RuntimeError(
                f"Weight switching requires a weight-switch capable linear method, got {type(linear_method).__name__}."
            )
        return linear_method

    def _enable_o_proj_full_weight_switch(self) -> None:
        if self._o_proj_weight_switch_enabled:
            return

        linear_method = self._get_o_proj_weight_switch_method()
        self.o_proj_weight_state = linear_method.enable_weight_switch(
            self.o_proj,
            self.o_proj_weight_switch_config,
            pool=self.o_proj_full_pools,
            pool_key_prefix=(
                type(linear_method).__qualname__,
                self.o_proj_weight_switch_pool_key,
            ),
        )
        self._o_proj_weight_switch_enabled = True

    def _get_o_proj_weight_switch_state(self) -> WeightSwitchState:
        state = self.o_proj_weight_state
        if not self._o_proj_weight_switch_enabled or state is None:
            raise RuntimeError("O-proj weight switching has not been enabled.")
        return state

    def _all_gather_o_proj_full_weight(self) -> None:
        self._enable_o_proj_full_weight_switch()
        self._get_o_proj_weight_switch_method().all_gather_weight(
            self._get_o_proj_weight_switch_state(),
            self.o_proj_weight_switch_config,
        )

    @contextmanager
    def _use_full_o_proj_weights(self) -> Iterator[None]:
        linear_method = self._get_o_proj_weight_switch_method()
        state = self._get_o_proj_weight_switch_state()
        linear_method.wait_weight_all_gather(state)
        linear_method.switch_weight(
            self.o_proj,
            state,
            use_full_weight=True,
        )
        try:
            yield
        finally:
            linear_method.switch_weight(
                self.o_proj,
                state,
                use_full_weight=False,
            )
