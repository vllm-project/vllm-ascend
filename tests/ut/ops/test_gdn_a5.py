# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_ascend.ops.gdn_a5 import (
    A5GDNOperatorDispatcher,
    GDNBackendMode,
    GDNOperator,
    GDNRuntimeSignature,
    parse_gdn_backend_config,
)


SIGNATURE = GDNRuntimeSignature(
    soc="ascend950",
    dtype="bfloat16",
    state_dtype="float32",
    num_key_heads=2,
    num_value_heads=4,
    key_dim=128,
    value_dim=128,
    chunk_size=64,
)


def test_parse_gdn_backend_config_defaults_to_auto_without_overrides():
    config = parse_gdn_backend_config("auto", "")

    assert config.mode is GDNBackendMode.AUTO
    assert config.overrides == {}


def test_parse_gdn_backend_config_accepts_explicit_global_modes():
    assert parse_gdn_backend_config("native", "").mode is GDNBackendMode.NATIVE
    assert parse_gdn_backend_config("fla_npu", "").mode is GDNBackendMode.FLA_NPU


def test_parse_gdn_backend_config_accepts_per_operator_overrides():
    config = parse_gdn_backend_config(
        "auto",
        "causal_conv1d=fla_npu,chunk_fwd_o=native",
    )

    assert config.overrides == {
        GDNOperator.CAUSAL_CONV1D: GDNBackendMode.FLA_NPU,
        GDNOperator.CHUNK_FWD_O: GDNBackendMode.NATIVE,
    }


@pytest.mark.parametrize("mode", ["", "invalid", "FLA"])
def test_parse_gdn_backend_config_rejects_invalid_global_mode(mode):
    with pytest.raises(ValueError, match="GDN backend mode"):
        parse_gdn_backend_config(mode, "")


@pytest.mark.parametrize(
    "overrides",
    [
        "unknown=native",
        "causal_conv1d=invalid",
        "causal_conv1d=auto",
        "causal_conv1d=native,causal_conv1d=fla_npu",
        "causal_conv1d",
        "=native",
    ],
)
def test_parse_gdn_backend_config_rejects_invalid_overrides(overrides):
    with pytest.raises(ValueError, match="GDN operator backend override"):
        parse_gdn_backend_config("auto", overrides)


def _native_operator(value):
    return ("native", value)


def _fla_operator(value):
    return ("fla_npu", value)


def test_auto_selects_fla_operator_after_successful_probe():
    dispatcher = A5GDNOperatorDispatcher(parse_gdn_backend_config("auto", ""), is_a5=True)

    selection = dispatcher.select(
        GDNOperator.CHUNK_FWD_O,
        SIGNATURE,
        native=_native_operator,
        native_symbol="native.chunk_fwd_o",
        fla_resolver=lambda: (_fla_operator, "fla_npu.ops.ascendc.chunk_fwd_o"),
        probe=lambda operator: operator("probe") == ("fla_npu", "probe"),
    )

    assert selection.backend is GDNBackendMode.FLA_NPU
    assert selection.symbol == "fla_npu.ops.ascendc.chunk_fwd_o"
    assert selection.operator("input") == ("fla_npu", "input")


def test_auto_falls_back_when_fla_symbol_is_missing():
    dispatcher = A5GDNOperatorDispatcher(parse_gdn_backend_config("auto", ""), is_a5=True)

    def missing_resolver():
        raise AttributeError("missing chunk_fwd_o")

    selection = dispatcher.select(
        GDNOperator.CHUNK_FWD_O,
        SIGNATURE,
        native=_native_operator,
        native_symbol="native.chunk_fwd_o",
        fla_resolver=missing_resolver,
    )

    assert selection.backend is GDNBackendMode.NATIVE
    assert selection.reason == "missing chunk_fwd_o"
    assert selection.operator("input") == ("native", "input")


def test_strict_fla_mode_does_not_hide_probe_failure():
    dispatcher = A5GDNOperatorDispatcher(parse_gdn_backend_config("fla_npu", ""), is_a5=True)

    with pytest.raises(RuntimeError, match="chunk_fwd_o.*smoke_probe"):
        dispatcher.select(
            GDNOperator.CHUNK_FWD_O,
            SIGNATURE,
            native=_native_operator,
            native_symbol="native.chunk_fwd_o",
            fla_resolver=lambda: (_fla_operator, "fla_npu.ops.ascendc.chunk_fwd_o"),
            probe=lambda operator: False,
        )


def test_native_mode_does_not_resolve_fla_operator():
    dispatcher = A5GDNOperatorDispatcher(parse_gdn_backend_config("native", ""), is_a5=True)

    selection = dispatcher.select(
        GDNOperator.CHUNK_FWD_O,
        SIGNATURE,
        native=_native_operator,
        native_symbol="native.chunk_fwd_o",
        fla_resolver=lambda: pytest.fail("native mode must not import fla_npu"),
    )

    assert selection.backend is GDNBackendMode.NATIVE


def test_non_a5_always_uses_native_operator():
    dispatcher = A5GDNOperatorDispatcher(parse_gdn_backend_config("fla_npu", ""), is_a5=False)

    selection = dispatcher.select(
        GDNOperator.CHUNK_FWD_O,
        SIGNATURE,
        native=_native_operator,
        native_symbol="native.chunk_fwd_o",
        fla_resolver=lambda: pytest.fail("non-A5 must not import fla_npu"),
    )

    assert selection.backend is GDNBackendMode.NATIVE


def test_selection_is_cached_for_the_same_operator_and_signature():
    dispatcher = A5GDNOperatorDispatcher(parse_gdn_backend_config("auto", ""), is_a5=True)
    resolves = 0

    def resolver():
        nonlocal resolves
        resolves += 1
        return _fla_operator, "fla_npu.ops.ascendc.chunk_fwd_o"

    first = dispatcher.select(
        GDNOperator.CHUNK_FWD_O,
        SIGNATURE,
        native=_native_operator,
        native_symbol="native.chunk_fwd_o",
        fla_resolver=resolver,
    )
    second = dispatcher.select(
        GDNOperator.CHUNK_FWD_O,
        SIGNATURE,
        native=_native_operator,
        native_symbol="native.chunk_fwd_o",
        fla_resolver=resolver,
    )

    assert first is second
    assert resolves == 1


def test_runtime_error_is_propagated_without_fallback():
    dispatcher = A5GDNOperatorDispatcher(parse_gdn_backend_config("auto", ""), is_a5=True)

    def failing_operator(value):
        raise RuntimeError(f"failed after receiving {value}")

    selection = dispatcher.select(
        GDNOperator.RECURRENT_GATED_DELTA_RULE,
        SIGNATURE,
        native=_native_operator,
        native_symbol="native.recurrent",
        fla_resolver=lambda: (failing_operator, "fla_npu.recurrent"),
    )

    with pytest.raises(RuntimeError, match="failed after receiving request"):
        dispatcher.execute(
            GDNOperator.RECURRENT_GATED_DELTA_RULE,
            selection,
            "request",
            phase="decode",
            layer_name="model.layers.0.linear_attn",
            state_may_be_mutated=True,
        )
