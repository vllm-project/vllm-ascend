# SPDX-License-Identifier: Apache-2.0
"""A5 operator selection and adapters for Qwen gated delta networks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


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
