# SPDX-License-Identifier: Apache-2.0
"""O-projection weight-switch lifecycle and adapter."""

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any, Literal, TypeAlias

import torch

from .linear import (
    WeightLoadPartition,
    WeightSwitchConfig,
    WeightSwitchLoadState,
    WeightSwitchMixin,
    WeightSwitchState,
)

OProjWeightLoadSpec: TypeAlias = tuple[
    str,
    torch.nn.Module,
    WeightSwitchMixin,
    Literal["input", "output"],
]
OProjWeightLoadStates: TypeAlias = dict[str, WeightSwitchLoadState | None]


class OProjWeightSwitchMixin:
    """Manage temporary full-weight O-projection execution.

    The default adapter manages one o_proj layer. Composite projections such
    as DSA override the O-projection layer hook and may provide path-specific
    enable options.
    """

    o_proj: Any
    o_proj_full_pools: dict[Any, torch.Tensor]
    o_proj_weight_switch_pool_key = "o_proj"

    def _initialize_o_proj_weight_switch(self, config: WeightSwitchConfig | None) -> None:
        self._weight_switch_config = config
        self._weight_switch_states: tuple[tuple[torch.nn.Module, WeightSwitchMixin, WeightSwitchState], ...] | None = (
            None
        )
        self._weight_switch_enabled = False
        self._o_proj_use_full_weight = False
        self._o_proj_weight_load_states: OProjWeightLoadStates | None = None

    def _set_o_proj_use_full_weight(self, enabled: bool) -> bool:
        self._o_proj_use_full_weight = bool(enabled)
        return self._o_proj_use_full_weight

    def _o_proj_uses_full_weight(self) -> bool:
        return bool(getattr(self, "_o_proj_use_full_weight", False))

    @contextmanager
    def _o_proj_forward_context(self) -> Iterator[None]:
        self._set_o_proj_use_full_weight(False)
        try:
            yield
        except BaseException:
            self._finish_o_proj_forward(suppress_errors=True)
            raise
        else:
            self._finish_o_proj_forward()

    def _finish_o_proj_forward(self, *, suppress_errors: bool = False) -> None:
        first_error: Exception | None = None
        try:
            for _, method, state in getattr(self, "_weight_switch_states", None) or ():
                if not state.handles:
                    continue
                try:
                    method.wait_weight_all_gather(state)
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
        finally:
            self._set_o_proj_use_full_weight(False)
        if first_error is not None and not suppress_errors:
            raise first_error

    def _get_o_proj_weight_switch_config(self) -> WeightSwitchConfig:
        config = getattr(self, "_weight_switch_config", None)
        if config is None:
            raise RuntimeError("O-proj weight switching has not been initialized.")
        return config

    @staticmethod
    def _resolve_o_proj_weight_switch_method(layer: torch.nn.Module) -> WeightSwitchMixin:
        quant_method = getattr(layer, "quant_method", None)
        linear_method = getattr(quant_method, "quant_method", quant_method)
        if not isinstance(linear_method, WeightSwitchMixin) or not linear_method.supports_weight_switch:
            raise RuntimeError(
                f"O-proj weight switching requires a weight-switch capable method, got {type(linear_method).__name__}."
            )
        return linear_method

    def _get_o_proj_weight_switch_pool(self) -> dict[Any, torch.Tensor]:
        return self.o_proj_full_pools

    def _get_o_proj_weight_switch_pool_key(self) -> str:
        return self.o_proj_weight_switch_pool_key

    def _get_o_proj_weight_switch_method(self) -> WeightSwitchMixin:
        return self._resolve_o_proj_weight_switch_method(self.o_proj)

    def _get_o_proj_weight_switch_enable_kwargs(self) -> dict[str, Any]:
        return {}

    def _get_o_proj_weight_load_specs(self) -> tuple[OProjWeightLoadSpec, ...]:
        """Describe load-time sharding for the O-projection layers."""
        return (("o_proj", self.o_proj, self._get_o_proj_weight_switch_method(), "input"),)

    def _get_o_proj_weight_switch_layers(
        self,
    ) -> tuple[tuple[str, torch.nn.Module, WeightSwitchMixin], ...]:
        return tuple((name, layer, method) for name, layer, method, _ in self._get_o_proj_weight_load_specs())

    def _get_o_proj_weight_load_groups(self):
        from vllm.distributed import get_pcp_group, get_tp_group

        return get_tp_group(), get_pcp_group()

    def _get_o_proj_weight_load_partition(self) -> WeightLoadPartition:
        """Build the nested TP+PCP checkpoint partition used by O-proj loads."""
        tp_group, pcp_group = self._get_o_proj_weight_load_groups()
        return WeightLoadPartition.from_nested_groups(
            tp_group,
            pcp_group,
        )

    def _prepare_o_proj_weight_load_shards(self) -> OProjWeightLoadStates:
        """Prepare O-proj parameters before checkpoint loading."""
        load_states = getattr(self, "_o_proj_weight_load_states", None)
        if load_states is not None:
            return load_states

        config = self._get_o_proj_weight_switch_config()
        partition = self._get_o_proj_weight_load_partition()
        load_states: OProjWeightLoadStates = {}
        for name, layer, method, shard_axis in self._get_o_proj_weight_load_specs():
            if shard_axis == "input":
                load_states[name] = method.prepare_layer_for_parallel_weight_load(
                    layer,
                    config,
                    partition,
                )
            elif shard_axis == "output":
                method.prepare_layer_for_parallel_output_weight_load(
                    layer,
                    config,
                    partition,
                )
                load_states[name] = None
            else:
                raise ValueError(f"Unsupported O-proj load shard axis: {shard_axis!r}")

        self._o_proj_weight_load_states = load_states
        return load_states

    def _get_o_proj_weight_load_state(self, name: str) -> WeightSwitchLoadState:
        load_states = self._o_proj_weight_load_states
        if load_states is None or name not in load_states:
            raise RuntimeError(f"O-proj load state {name!r} has not been prepared.")
        state = load_states[name]
        if state is None:
            raise RuntimeError(f"O-proj load state {name!r} has no input-shard state.")
        return state

    def _prepare_o_proj_weight_switches(
        self,
    ) -> tuple[tuple[torch.nn.Module, WeightSwitchMixin, WeightSwitchState], ...]:
        switches = getattr(self, "_weight_switch_states", None)
        if switches is None:
            config = self._get_o_proj_weight_switch_config()
            enable_kwargs = self._get_o_proj_weight_switch_enable_kwargs()
            switches = tuple(
                (
                    layer,
                    method,
                    method.enable_weight_switch(
                        layer,
                        config,
                        pool=self._get_o_proj_weight_switch_pool(),
                        pool_key_prefix=(
                            type(method).__qualname__,
                            name,
                            self._get_o_proj_weight_switch_pool_key(),
                        ),
                        **enable_kwargs,
                    ),
                )
                for name, layer, method in self._get_o_proj_weight_switch_layers()
            )
            self._weight_switch_states = switches
        return switches

    def _enable_o_proj_weight_switch(self) -> None:
        if getattr(self, "_weight_switch_enabled", False):
            return
        self._prepare_o_proj_weight_switches()
        self._weight_switch_enabled = True

    def _require_o_proj_weight_switches(
        self,
    ) -> tuple[tuple[torch.nn.Module, WeightSwitchMixin, WeightSwitchState], ...]:
        switches = getattr(self, "_weight_switch_states", None)
        if switches is None:
            raise RuntimeError("O-proj weight switching has not been prepared.")
        return switches

    def _all_gather_o_proj_weight_switches(self) -> None:
        config = self._get_o_proj_weight_switch_config()
        switches = self._require_o_proj_weight_switches()
        try:
            for _, method, state in switches:
                method.all_gather_weight(state, config)
        except Exception:
            for _, method, state in switches:
                if state.handles:
                    with suppress(Exception):
                        method.wait_weight_all_gather(state)
            raise

    @contextmanager
    def _use_full_o_proj_weight_switches(self) -> Iterator[None]:
        switches = self._require_o_proj_weight_switches()
        switched_layers: list[tuple[torch.nn.Module, WeightSwitchMixin, WeightSwitchState]] = []
        try:
            for layer, method, state in switches:
                method.wait_weight_all_gather(state)
                switched_layers.append((layer, method, state))
                method.switch_weight(layer, state, use_full_weight=True)
            yield
        finally:
            for layer, method, state in reversed(switched_layers):
                method.switch_weight(layer, state, use_full_weight=False)
