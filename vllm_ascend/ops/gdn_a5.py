# SPDX-License-Identifier: Apache-2.0
"""A5 operator selection and adapters for Qwen gated delta networks."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

from vllm.logger import init_logger

logger = init_logger(__name__)


class GDNBackendMode(StrEnum):
    AUTO = "auto"
    FLA_NPU = "fla_npu"
    NATIVE = "native"


class GDNOperator(StrEnum):
    CAUSAL_CONV1D = "causal_conv1d"
    L2NORM_FWD = "l2norm_fwd"
    CHUNK_LOCAL_CUMSUM = "chunk_local_cumsum"
    CHUNK_SCALED_DOT_KKT = "chunk_scaled_dot_kkt"
    SOLVE_TRI = "solve_tri"
    RECOMPUTE_W_U_FWD = "recompute_w_u_fwd"
    CHUNK_GATED_DELTA_RULE_FWD_H = "chunk_gated_delta_rule_fwd_h"
    CHUNK_FWD_O = "chunk_fwd_o"
    RECURRENT_GATED_DELTA_RULE = "recurrent_gated_delta_rule"


@dataclass(frozen=True)
class GDNBackendConfig:
    mode: GDNBackendMode
    overrides: dict[GDNOperator, GDNBackendMode]

    def mode_for(self, operator: GDNOperator) -> GDNBackendMode:
        return self.overrides.get(operator, self.mode)


@dataclass(frozen=True)
class GDNRuntimeSignature:
    soc: str
    dtype: str
    state_dtype: str
    num_key_heads: int
    num_value_heads: int
    key_dim: int
    value_dim: int
    chunk_size: int = 64
    mtp: bool = False
    acl_graph: bool = False


@dataclass(frozen=True)
class GDNOperatorSelection:
    backend: GDNBackendMode
    operator: Callable[..., Any]
    symbol: str
    reason: str | None = None


_FLA_OPERATOR_PATHS: dict[GDNOperator, tuple[str, str]] = {
    GDNOperator.CAUSAL_CONV1D: ("fla_npu.ops.ascendc", "causal_conv1d"),
    GDNOperator.L2NORM_FWD: ("fla_npu.ops.triton", "l2norm_fwd"),
    GDNOperator.CHUNK_LOCAL_CUMSUM: ("fla_npu.ops.triton", "chunk_local_cumsum"),
    GDNOperator.CHUNK_SCALED_DOT_KKT: ("fla_npu.ops.triton", "chunk_scaled_dot_kkt_fwd"),
    GDNOperator.SOLVE_TRI: ("fla_npu.ops.ascendc", "solve_tri"),
    GDNOperator.RECOMPUTE_W_U_FWD: ("fla_npu.ops.ascendc", "recompute_w_u_fwd"),
    GDNOperator.CHUNK_GATED_DELTA_RULE_FWD_H: (
        "fla_npu.ops.ascendc",
        "chunk_gated_delta_rule_fwd_h",
    ),
    GDNOperator.CHUNK_FWD_O: ("fla_npu.ops.ascendc", "chunk_fwd_o"),
}


def _first_line(exc: BaseException) -> str:
    return str(exc).splitlines()[0] if str(exc) else type(exc).__name__


def resolve_fla_operator(operator: GDNOperator) -> tuple[Callable[..., Any], str]:
    """Resolve a public fla_npu operator without importing it on native paths."""

    if operator is GDNOperator.RECURRENT_GATED_DELTA_RULE:
        return _resolve_fla_recurrent_operator()
    module_name, attribute = _FLA_OPERATOR_PATHS[operator]
    module = importlib.import_module(module_name)
    resolved = getattr(module, attribute)
    return resolved, f"{module_name}.{attribute}"


def _resolve_fla_recurrent_operator() -> tuple[Callable[..., Any], str]:
    candidates = (
        ("fla_npu.ops.ascendc", "recurrent_gated_delta_rule"),
        ("torch_npu", "npu_recurrent_gated_delta_rule"),
    )
    errors: list[str] = []
    for module_name, attribute in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attribute), f"{module_name}.{attribute}"
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}.{attribute}: {_first_line(exc)}")

    try:
        import torch

        for namespace_name in ("ascend_ops", "_C_ascend"):
            namespace = getattr(torch.ops, namespace_name)
            attribute = (
                "recurrent_gated_delta_rule"
                if namespace_name == "ascend_ops"
                else "npu_recurrent_gated_delta_rule"
            )
            if hasattr(namespace, attribute):
                return getattr(namespace, attribute), f"torch.ops.{namespace_name}.{attribute}"
            errors.append(f"torch.ops.{namespace_name}.{attribute}: missing")
    except (ImportError, AttributeError) as exc:
        errors.append(f"torch.ops: {_first_line(exc)}")
    raise AttributeError("; ".join(errors))


class A5GDNOperatorDispatcher:
    """Select and cache normalized A5 GDN operator implementations."""

    def __init__(self, config: GDNBackendConfig, *, is_a5: bool) -> None:
        self.config = config
        self.is_a5 = is_a5
        self._selections: dict[tuple[GDNOperator, GDNRuntimeSignature], GDNOperatorSelection] = {}

    def select(
        self,
        operator: GDNOperator,
        signature: GDNRuntimeSignature,
        *,
        native: Callable[..., Any],
        native_symbol: str,
        fla_resolver: Callable[[], tuple[Callable[..., Any], str]] | None = None,
        probe: Callable[[Callable[..., Any]], bool | None] | None = None,
    ) -> GDNOperatorSelection:
        cache_key = (operator, signature)
        if cache_key in self._selections:
            return self._selections[cache_key]

        requested = self.config.mode_for(operator)
        if not self.is_a5 or requested is GDNBackendMode.NATIVE:
            selection = GDNOperatorSelection(GDNBackendMode.NATIVE, native, native_symbol)
            self._remember(operator, signature, selection)
            return selection

        resolver = fla_resolver or (lambda: resolve_fla_operator(operator))
        try:
            resolved, symbol = resolver()
        except Exception as exc:
            return self._fallback_or_raise(
                operator,
                signature,
                requested,
                native,
                native_symbol,
                stage="resolve",
                exc=exc,
            )

        if probe is not None:
            try:
                probe_result = probe(resolved)
                if probe_result is False:
                    raise RuntimeError("smoke probe returned false")
            except Exception as exc:
                return self._fallback_or_raise(
                    operator,
                    signature,
                    requested,
                    native,
                    native_symbol,
                    stage="smoke_probe",
                    exc=exc,
                )

        selection = GDNOperatorSelection(GDNBackendMode.FLA_NPU, resolved, symbol)
        self._remember(operator, signature, selection)
        return selection

    def execute(
        self,
        operator: GDNOperator,
        selection: GDNOperatorSelection,
        *args: Any,
        phase: str,
        layer_name: str,
        state_may_be_mutated: bool,
        **kwargs: Any,
    ) -> Any:
        try:
            return selection.operator(*args, **kwargs)
        except Exception:
            logger.exception(
                "GDN A5 operator execution failed: op=%s backend=%s symbol=%s phase=%s "
                "layer=%s state_may_be_mutated=%s",
                operator.value,
                selection.backend.value,
                selection.symbol,
                phase,
                layer_name,
                state_may_be_mutated,
            )
            raise

    def _fallback_or_raise(
        self,
        operator: GDNOperator,
        signature: GDNRuntimeSignature,
        requested: GDNBackendMode,
        native: Callable[..., Any],
        native_symbol: str,
        *,
        stage: str,
        exc: BaseException,
    ) -> GDNOperatorSelection:
        reason = _first_line(exc)
        if requested is GDNBackendMode.FLA_NPU:
            raise RuntimeError(
                f"GDN operator {operator.value} failed during {stage} in strict fla_npu mode: {reason}"
            ) from exc
        selection = GDNOperatorSelection(GDNBackendMode.NATIVE, native, native_symbol, reason)
        logger.warning(
            "GDN A5 operator fallback: op=%s requested=%s selected=native stage=%s "
            "exception=%s reason=%s",
            operator.value,
            requested.value,
            stage,
            type(exc).__name__,
            reason,
        )
        self._remember(operator, signature, selection)
        return selection

    def _remember(
        self,
        operator: GDNOperator,
        signature: GDNRuntimeSignature,
        selection: GDNOperatorSelection,
    ) -> None:
        self._selections[(operator, signature)] = selection
        logger.info(
            "GDN A5 operator selected: op=%s backend=%s symbol=%s soc=%s dtype=%s "
            "state_dtype=%s nk=%d nv=%d dk=%d dv=%d chunk_size=%d mtp=%s acl_graph=%s",
            operator.value,
            selection.backend.value,
            selection.symbol,
            signature.soc,
            signature.dtype,
            signature.state_dtype,
            signature.num_key_heads,
            signature.num_value_heads,
            signature.key_dim,
            signature.value_dim,
            signature.chunk_size,
            signature.mtp,
            signature.acl_graph,
        )


def _parse_mode(value: str) -> GDNBackendMode:
    try:
        return GDNBackendMode(value.strip().lower())
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in GDNBackendMode)
        raise ValueError(f"Invalid GDN backend mode {value!r}; expected one of: {valid}.") from exc


def parse_gdn_backend_config(mode: str, operator_overrides: str) -> GDNBackendConfig:
    """Parse global and per-operator backend configuration."""

    parsed_mode = _parse_mode(mode)
    overrides: dict[GDNOperator, GDNBackendMode] = {}
    if not operator_overrides.strip():
        return GDNBackendConfig(parsed_mode, overrides)

    for raw_entry in operator_overrides.split(","):
        entry = raw_entry.strip()
        if not entry or entry.count("=") != 1:
            raise ValueError(f"Invalid GDN operator backend override {raw_entry!r}; expected operator=backend.")
        raw_operator, raw_backend = (part.strip().lower() for part in entry.split("=", 1))
        try:
            operator = GDNOperator(raw_operator)
        except ValueError as exc:
            raise ValueError(f"Invalid GDN operator backend override {raw_entry!r}: unknown operator.") from exc
        try:
            backend = GDNBackendMode(raw_backend)
        except ValueError as exc:
            raise ValueError(f"Invalid GDN operator backend override {raw_entry!r}: unknown backend.") from exc
        if backend is GDNBackendMode.AUTO:
            raise ValueError(f"Invalid GDN operator backend override {raw_entry!r}: auto is only a global mode.")
        if operator in overrides:
            raise ValueError(f"Invalid GDN operator backend override {raw_entry!r}: duplicate operator.")
        overrides[operator] = backend

    return GDNBackendConfig(parsed_mode, overrides)
