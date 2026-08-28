# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.ops.gdn_a5 import (
    A5GDNAdapter,
    A5GDNOperatorDispatcher,
    GDNBackendMode,
    GDNOperator,
    GDNPrefillMetadata,
    GDNRuntimeSignature,
    parse_gdn_backend_config,
    run_gdn_decode_pipeline,
    run_gdn_prefill_pipeline,
    log_solve_tri_debug,
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


def test_log_solve_tri_debug_is_gated_by_environment(monkeypatch):
    tensor = SimpleNamespace(
        shape=(1, 2, 64, 64),
        dtype="float32",
        device="npu:0",
        stride=lambda: (8192, 4096, 64, 1),
        is_contiguous=lambda: True,
    )
    with patch("vllm_ascend.ops.gdn_a5.logger.info") as info:
        monkeypatch.setenv("VLLM_ASCEND_GDN_DEBUG_SOLVE_TRI", "1")
        log_solve_tri_debug(
            tensor,
            output_dtype="float32",
            layout="tnd",
            cu_seqlens_host=[0, 64],
            chunk_indices_host=[0, 0],
        )
        info.assert_called_once()
        assert "solve_tri" in info.call_args.args[0]
        assert "cu_seqlens_host_values=%s" in info.call_args.args[0]


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
        "l2norm_fwd=fla_npu",
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


def test_fallback_log_identifies_operator_backend_stage_and_exception():
    dispatcher = A5GDNOperatorDispatcher(parse_gdn_backend_config("auto", ""), is_a5=True)

    def missing_resolver():
        raise ImportError("missing op_api library")

    with patch("vllm_ascend.ops.gdn_a5.logger.warning") as warning:
        dispatcher.select(
            GDNOperator.SOLVE_TRI,
            SIGNATURE,
            native=_native_operator,
            native_symbol="native.solve_tri",
            fla_resolver=missing_resolver,
        )

    message, operator, requested, stage, exception_name, reason = warning.call_args.args
    assert "op=%s" in message
    assert "requested=%s" in message
    assert "stage=%s" in message
    assert operator == "solve_tri"
    assert requested == "auto"
    assert stage == "resolve"
    assert exception_name == "ImportError"
    assert reason == "missing op_api library"


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


def test_strict_adapter_validation_aggregates_missing_symbols(monkeypatch):
    def resolver(operator):
        if operator in {GDNOperator.CAUSAL_CONV1D, GDNOperator.SOLVE_TRI}:
            raise ImportError(f"missing {operator.value}")
        return _fla_operator, f"fla_npu.{operator.value}"

    monkeypatch.setattr("vllm_ascend.ops.gdn_a5.resolve_fla_operator", resolver)
    with pytest.raises(RuntimeError) as error:
        A5GDNAdapter(
            parse_gdn_backend_config("fla_npu", ""),
            SIGNATURE,
            layer_name="model.layers.0.linear_attn",
            is_a5=True,
        )

    message = str(error.value)
    assert "causal_conv1d: missing causal_conv1d" in message
    assert "solve_tri: missing solve_tri" in message


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


def test_prefill_pipeline_matches_reference_order_and_layouts():
    calls = []
    batch, tokens, key_heads, value_heads, dim = 1, 3, 1, 2, 4
    q = torch.arange(batch * tokens * key_heads * dim, dtype=torch.float32).view(
        batch, tokens, key_heads, dim
    )
    k = q + 1
    v = torch.arange(batch * tokens * value_heads * dim, dtype=torch.float32).view(
        batch, tokens, value_heads, dim
    )
    g = torch.full((batch, tokens, value_heads), -0.25)
    beta = torch.full((batch, tokens, value_heads), 0.5)
    state = torch.arange(value_heads * dim * dim, dtype=torch.float32).view(1, value_heads, dim, dim)

    def l2norm(x):
        calls.append("l2norm_fwd")
        return x

    def cumsum(gate, **kwargs):
        calls.append("chunk_local_cumsum")
        return gate

    def kkt(key, gate, beta_value, **kwargs):
        calls.append("chunk_scaled_dot_kkt")
        assert key.shape == (batch, value_heads, tokens, dim)
        return torch.zeros((batch, tokens, value_heads, 64), dtype=key.dtype)

    def solve(a, **kwargs):
        calls.append("solve_tri")
        return a

    def recompute(key, value, beta_value, a, gate, **kwargs):
        calls.append("recompute_w_u_fwd")
        assert key.shape == value.shape == (batch, value_heads, tokens, dim)
        return key, value

    def fwd_h(key, w, u, gate, initial_state, **kwargs):
        calls.append("chunk_gated_delta_rule_fwd_h")
        assert initial_state.shape == (1, value_heads, dim, dim)
        h = torch.zeros((batch, value_heads, 1, dim, dim), dtype=key.dtype)
        return h, u, initial_state + 1

    def fwd_o(query, key, new_value, h, gate, **kwargs):
        calls.append("chunk_fwd_o")
        assert query.shape == key.shape == new_value.shape
        return new_value

    output, final_state = run_gdn_prefill_pipeline(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        has_initial_state=torch.tensor([True]),
        scale=0.5,
        metadata=GDNPrefillMetadata(
            cu_seqlens=torch.tensor([0, tokens], dtype=torch.int64),
            cu_seqlens_host=(0, tokens),
            chunk_indices=torch.tensor([[0, 0]], dtype=torch.int64),
            chunk_indices_host=(0, 0),
        ),
        operators={
            GDNOperator.L2NORM_FWD: l2norm,
            GDNOperator.CHUNK_LOCAL_CUMSUM: cumsum,
            GDNOperator.CHUNK_SCALED_DOT_KKT: kkt,
            GDNOperator.SOLVE_TRI: solve,
            GDNOperator.RECOMPUTE_W_U_FWD: recompute,
            GDNOperator.CHUNK_GATED_DELTA_RULE_FWD_H: fwd_h,
            GDNOperator.CHUNK_FWD_O: fwd_o,
        },
        chunk_size=64,
    )

    assert calls == [
        "l2norm_fwd",
        "l2norm_fwd",
        "chunk_local_cumsum",
        "chunk_scaled_dot_kkt",
        "solve_tri",
        "recompute_w_u_fwd",
        "chunk_gated_delta_rule_fwd_h",
        "chunk_fwd_o",
    ]
    assert output.shape == (batch, tokens, value_heads, dim)
    torch.testing.assert_close(output, v)
    torch.testing.assert_close(final_state, (state.transpose(-1, -2) + 1).transpose(-1, -2))


def test_prefill_pipeline_rejects_non_integral_grouped_heads():
    q = torch.zeros((1, 2, 2, 4))
    v = torch.zeros((1, 2, 3, 4))

    with pytest.raises(ValueError, match="value heads.*multiple of key heads"):
        run_gdn_prefill_pipeline(
            q=q,
            k=q,
            v=v,
            g=torch.zeros((1, 2, 3)),
            beta=torch.zeros((1, 2, 3)),
            initial_state=torch.zeros((1, 3, 4, 4)),
            has_initial_state=torch.tensor([True]),
            scale=0.5,
            metadata=GDNPrefillMetadata(None, None, None, None),
            operators={},
        )


def test_causal_conv_adapter_maps_stateful_arguments(monkeypatch):
    calls = []

    def causal_conv(x, weight, bias, conv_states, **kwargs):
        calls.append((x, weight, bias, conv_states, kwargs))
        conv_states.add_(1)
        return x + 2

    monkeypatch.setattr(
        "vllm_ascend.ops.gdn_a5.resolve_fla_operator",
        lambda operator: (causal_conv, "fla_npu.ops.ascendc.causal_conv1d"),
    )
    adapter = A5GDNAdapter(
        parse_gdn_backend_config("fla_npu", ""),
        SIGNATURE,
        layer_name="model.layers.0.linear_attn",
        is_a5=True,
    )
    x = torch.zeros((2, 8))
    weight = torch.zeros((4, 8))
    bias = torch.zeros((8,))
    state = torch.zeros((2, 8, 4))
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    cache_indices = torch.tensor([3, 7], dtype=torch.int32)
    initial_state_mode = torch.tensor([0, 1], dtype=torch.int32)

    output = adapter.causal_conv1d(
        x=x,
        weight=weight,
        bias=bias,
        conv_state=state,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        initial_state_mode=initial_state_mode,
        activation_mode=1,
        pad_slot_id=-1,
        run_mode=0,
    )

    # The first call is an isolated state clone smoke probe; the second call
    # applies the operator to the live cache only after the probe succeeds.
    assert len(calls) == 2
    assert calls[-1][4] == {
        "query_start_loc": query_start_loc,
        "cache_indices": cache_indices,
        "initial_state_mode": initial_state_mode,
        "num_accepted_tokens": None,
        "activation_mode": 1,
        "pad_slot_id": -1,
        "run_mode": 0,
        "head_num": 0,
    }
    torch.testing.assert_close(output, x + 2)
    torch.testing.assert_close(state, torch.ones_like(state))


def test_stateful_runtime_probe_falls_back_without_mutating_live_state():
    dispatcher = A5GDNOperatorDispatcher(parse_gdn_backend_config("auto", ""), is_a5=True)
    state = torch.zeros((1,))

    def failing_fla(value):
        value.add_(1)
        raise RuntimeError("OPP load failed")

    def native(value):
        value.add_(2)
        return value

    selection = dispatcher.select(
        GDNOperator.CAUSAL_CONV1D,
        SIGNATURE,
        native=native,
        native_symbol="native.causal_conv1d",
        fla_resolver=lambda: (failing_fla, "fla_npu.ops.ascendc.causal_conv1d"),
    )
    output = dispatcher.execute_with_runtime_probe(
        GDNOperator.CAUSAL_CONV1D,
        SIGNATURE,
        selection,
        state,
        native=native,
        native_symbol="native.causal_conv1d",
        phase="decode",
        layer_name="model.layers.0.linear_attn",
        state_may_be_mutated=True,
    )

    torch.testing.assert_close(state, torch.full_like(state, 2))
    torch.testing.assert_close(output, state)


def test_causal_conv_prefill_and_decode_are_probed_separately():
    dispatcher = A5GDNOperatorDispatcher(parse_gdn_backend_config("auto", ""), is_a5=True)
    calls = 0

    def causal(input_tensor, weight, bias, conv_state, **kwargs):
        nonlocal calls
        del weight, bias, kwargs
        calls += 1
        conv_state.add_(1)
        return input_tensor

    selection = dispatcher.select(
        GDNOperator.CAUSAL_CONV1D,
        SIGNATURE,
        native=causal,
        native_symbol="native.causal_conv1d",
        fla_resolver=lambda: (causal, "fla_npu.ops.ascendc.causal_conv1d"),
    )
    x = torch.zeros((1, 2))
    state = torch.zeros((1, 1, 2))
    common = {
        "native": causal,
        "native_symbol": "native.causal_conv1d",
        "layer_name": "model.layers.0.linear_attn",
        "state_may_be_mutated": True,
        "query_start_loc": torch.tensor([0, 1]),
        "cache_indices": torch.tensor([0]),
        "initial_state_mode": None,
        "num_accepted_tokens": None,
        "activation_mode": 1,
        "pad_slot_id": -1,
        "run_mode": 0,
        "head_num": 0,
    }
    dispatcher.execute_with_runtime_probe(
        GDNOperator.CAUSAL_CONV1D,
        SIGNATURE,
        selection,
        x,
        torch.zeros((1, 2)),
        None,
        state,
        phase="prefill",
        **common,
    )
    common["run_mode"] = 1
    dispatcher.execute_with_runtime_probe(
        GDNOperator.CAUSAL_CONV1D,
        SIGNATURE,
        selection,
        x,
        torch.zeros((1, 2)),
        None,
        state,
        phase="decode",
        **common,
    )

    assert calls == 4
    torch.testing.assert_close(state, torch.full_like(state, 2))


def test_runtime_probe_falls_back_on_invalid_output_contract():
    dispatcher = A5GDNOperatorDispatcher(parse_gdn_backend_config("auto", ""), is_a5=True)

    def native(gate, **kwargs):
        del kwargs
        return gate

    selection = dispatcher.select(
        GDNOperator.CHUNK_LOCAL_CUMSUM,
        SIGNATURE,
        native=native,
        native_symbol="native.chunk_local_cumsum",
        fla_resolver=lambda: (
            lambda gate, **kwargs: gate[..., :1],
            "fla_npu.ops.triton.chunk_local_cumsum",
        ),
    )
    gate = torch.zeros((1, 3, 2))
    output = dispatcher.execute_with_runtime_probe(
        GDNOperator.CHUNK_LOCAL_CUMSUM,
        SIGNATURE,
        selection,
        gate,
        native=native,
        native_symbol="native.chunk_local_cumsum",
        phase="prefill",
        layer_name="model.layers.0.linear_attn",
        state_may_be_mutated=False,
        chunk_size=64,
        cu_seqlens=None,
        chunk_indices=None,
        block_indices=None,
    )

    assert output is gate


def test_stage1_warmup_runs_prefill_and_both_causal_modes_once(monkeypatch):
    adapter = A5GDNAdapter(
        parse_gdn_backend_config("auto", ""),
        SIGNATURE,
        layer_name="model.layers.0.linear_attn",
        is_a5=True,
    )
    calls = []
    monkeypatch.setattr(adapter, "prefill", lambda **kwargs: calls.append(("prefill", kwargs)))
    monkeypatch.setattr(
        adapter,
        "causal_conv1d",
        lambda **kwargs: calls.append(("causal", kwargs["run_mode"])),
    )

    adapter.warmup(
        conv_weight=torch.zeros((4, 16), dtype=torch.bfloat16),
        conv_bias=None,
        state_dtype=torch.float32,
    )
    adapter.warmup(
        conv_weight=torch.zeros((4, 16), dtype=torch.bfloat16),
        conv_bias=None,
        state_dtype=torch.float32,
    )

    assert [call[0] for call in calls] == ["prefill", "causal", "causal"]
    assert calls[1:] == [("causal", 0), ("causal", 1)]


def test_decode_pipeline_normalizes_once_and_preserves_native_state_mutation():
    calls = []
    state = torch.zeros((4, 2, 4, 4))

    def l2norm(x):
        calls.append("l2norm_fwd")
        return x + 1

    def recurrent(**kwargs):
        calls.append("recurrent_gated_delta_rule")
        assert kwargs["query"].shape == (2, 1, 4)
        assert kwargs["key"].shape == (2, 1, 4)
        kwargs["state"].add_(3)
        return kwargs["value"] + 4

    output = run_gdn_decode_pipeline(
        q=torch.zeros((1, 2, 1, 4)),
        k=torch.zeros((1, 2, 1, 4)),
        v=torch.zeros((1, 2, 2, 4)),
        g=torch.zeros((1, 2, 2)),
        beta=torch.zeros((1, 2, 2)),
        state=state,
        scale=0.5,
        actual_seq_lengths=torch.tensor([0, 1, 1], dtype=torch.int32),
        ssm_state_indices=torch.tensor([1, 3], dtype=torch.int32),
        l2norm=l2norm,
        recurrent=recurrent,
    )

    assert calls == ["l2norm_fwd", "l2norm_fwd", "recurrent_gated_delta_rule"]
    assert output.shape == (1, 2, 2, 4)
    torch.testing.assert_close(output, torch.full_like(output, 4))
    torch.testing.assert_close(state, torch.full_like(state, 3))


def test_decode_pipeline_copies_functional_state_once():
    state = torch.zeros((2, 1, 2, 2))

    def recurrent(**kwargs):
        return kwargs["value"], torch.full_like(kwargs["state"][:1], 5)

    run_gdn_decode_pipeline(
        q=torch.zeros((1, 1, 1, 2)),
        k=torch.zeros((1, 1, 1, 2)),
        v=torch.zeros((1, 1, 1, 2)),
        g=torch.zeros((1, 1, 1)),
        beta=torch.zeros((1, 1, 1)),
        state=state,
        scale=2**-0.5,
        actual_seq_lengths=torch.tensor([0, 1], dtype=torch.int32),
        ssm_state_indices=torch.tensor([0], dtype=torch.int32),
        l2norm=lambda value: value,
        recurrent=recurrent,
    )

    torch.testing.assert_close(state[0], torch.full_like(state[0], 5))
    torch.testing.assert_close(state[1], torch.zeros_like(state[1]))


def _fake_gdn_layer():
    return SimpleNamespace(
        num_k_heads=2,
        num_v_heads=4,
        tp_size=1,
        head_k_dim=128,
        head_v_dim=128,
        prefix="model.layers.0.linear_attn",
    )


def test_a5_routing_constructs_and_caches_one_adapter(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_GDN_BACKEND", "auto")
    monkeypatch.delenv("VLLM_ASCEND_GDN_OP_BACKENDS", raising=False)
    layer = _fake_gdn_layer()
    activation = torch.zeros((1, 128), dtype=torch.bfloat16)
    state = torch.zeros((2, 4, 128, 128), dtype=torch.float32)

    with (
        patch("vllm_ascend.ops.gdn.is_950", return_value=True),
        patch(
            "vllm_ascend.ops.gdn.get_pcp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
    ):
        first = AscendGatedDeltaNetAttention._get_a5_gdn_adapter(layer, activation, state)
        second = AscendGatedDeltaNetAttention._get_a5_gdn_adapter(layer, activation, state)

    assert isinstance(first, A5GDNAdapter)
    assert second is first


def test_a5_routing_shares_dispatcher_across_layers(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_GDN_BACKEND", "auto")
    monkeypatch.delenv("VLLM_ASCEND_GDN_OP_BACKENDS", raising=False)

    with (
        patch.dict(AscendGatedDeltaNetAttention._a5_gdn_dispatchers, {}, clear=True),
        patch("vllm_ascend.ops.gdn.is_950", return_value=True),
        patch(
            "vllm_ascend.ops.gdn.get_pcp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
    ):
        first = AscendGatedDeltaNetAttention._get_a5_gdn_adapter(
            _fake_gdn_layer(),
            torch.zeros((1, 128), dtype=torch.bfloat16),
            torch.float32,
        )
        second = AscendGatedDeltaNetAttention._get_a5_gdn_adapter(
            _fake_gdn_layer(),
            torch.zeros((1, 128), dtype=torch.bfloat16),
            torch.float32,
        )

    assert first is not None
    assert second is not None
    assert second.dispatcher is first.dispatcher


def test_a5_routing_preserves_exact_native_path(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_GDN_BACKEND", "native")
    monkeypatch.delenv("VLLM_ASCEND_GDN_OP_BACKENDS", raising=False)

    with (
        patch("vllm_ascend.ops.gdn.is_950", return_value=True),
        patch(
            "vllm_ascend.ops.gdn.get_pcp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
    ):
        adapter = AscendGatedDeltaNetAttention._get_a5_gdn_adapter(
            _fake_gdn_layer(),
            torch.zeros((1, 128), dtype=torch.bfloat16),
            torch.zeros((2, 4, 128, 128), dtype=torch.float32),
        )

    assert adapter is None


def test_a5_routing_rejects_non_bfloat16_strict_operator_override(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_GDN_BACKEND", "auto")
    monkeypatch.setenv(
        "VLLM_ASCEND_GDN_OP_BACKENDS",
        "causal_conv1d=fla_npu",
    )

    with (
        patch("vllm_ascend.ops.gdn.is_950", return_value=True),
        patch(
            "vllm_ascend.ops.gdn.get_pcp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
        pytest.raises(RuntimeError, match="requires bfloat16"),
    ):
        AscendGatedDeltaNetAttention._get_a5_gdn_adapter(
            _fake_gdn_layer(),
            torch.zeros((1, 128), dtype=torch.float16),
            torch.float32,
        )


def test_a5_mixed_decode_prefill_routes_and_merges_outputs():
    conv_state = torch.zeros((3, 1, 2))
    ssm_state = torch.zeros((2, 1, 2, 2))

    def rearrange_mixed_qkv(value):
        if value is None:
            return None, None, None
        projected = value.reshape(1, value.shape[0], 1, 2)
        return projected, projected, projected

    layer = SimpleNamespace(
        prefix="layers.0.linear_attn",
        kv_cache=(conv_state, ssm_state),
        conv1d=SimpleNamespace(weight=torch.zeros((2, 1, 2)), bias=None),
        activation=None,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        rearrange_mixed_qkv=rearrange_mixed_qkv,
    )

    chunk_meta = SimpleNamespace(
        cu_seqlens_host=(0, 2),
        chunk_indices_chunk64=torch.zeros((1, 2), dtype=torch.int32),
        chunk_indices_chunk64_host=(0, 0),
        cu_seqlens_kern=(0, 2),
        keep_meta=None,
        block_indices_cumsum=torch.zeros((1, 2), dtype=torch.int32),
        chunk_indices_large_block=torch.zeros((1, 2), dtype=torch.int32),
    )
    metadata = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=2,
        num_decodes=1,
        num_decode_tokens=1,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=3,
        non_spec_query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([0, 1], dtype=torch.int32),
        prefill_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        prefill_state_indices=torch.tensor([1], dtype=torch.int64),
        prefill_has_initial_state=torch.tensor([True]),
    )
    metadata.non_spec_prefill_metadata = SimpleNamespace(
        causal_conv1d=SimpleNamespace(
            query_start_loc=metadata.non_spec_query_start_loc,
            cache_indices=torch.tensor([0, 1], dtype=torch.int32),
            initial_state_mode=torch.tensor([True, True]),
        ),
        chunk=chunk_meta,
    )
    metadata.non_spec_decode_metadata = SimpleNamespace(
        actual_seq_lengths=torch.tensor([0, 1], dtype=torch.int32),
    )

    adapter = SimpleNamespace()

    def causal_conv1d(**kwargs):
        assert kwargs["run_mode"] == 0
        torch.testing.assert_close(
            kwargs["query_start_loc"],
            torch.tensor([0, 1, 3], dtype=torch.int32),
        )
        indices = kwargs["cache_indices"].to(torch.int64)
        update = torch.ones_like(kwargs["conv_state"].index_select(0, indices))
        kwargs["conv_state"].index_add_(0, indices, update)
        return kwargs["x"]

    def decode(**kwargs):
        indices = kwargs["ssm_state_indices"].to(torch.int64)
        update = torch.ones_like(kwargs["state"].index_select(0, indices))
        kwargs["state"].index_add_(0, indices, update)
        return torch.full_like(kwargs["v"], 10)

    def prefill(**kwargs):
        output = torch.full_like(kwargs["v"], 20)
        return output, kwargs["initial_state"] + 2

    adapter.causal_conv1d = causal_conv1d
    adapter.decode = decode
    adapter.prefill = prefill

    forward_context = ForwardContext(
        no_compile_layers={layer.prefix: layer},
        attn_metadata={layer.prefix: metadata},
        slot_mapping={},
    )
    core_attn_out = torch.empty((3, 1, 2))
    gating = (
        torch.zeros((1, 3, 1)),
        torch.zeros((1, 3, 1)),
    )

    with (
        override_forward_context(forward_context),
        patch(
            "vllm_ascend.ops.gdn.get_pcp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
        patch.object(
            AscendGatedDeltaNetAttention,
            "_get_a5_gdn_adapter",
            return_value=adapter,
        ),
        patch(
            "vllm_ascend.ops.gdn.DeviceOperator.fused_gdn_gating",
            return_value=gating,
        ),
        patch("vllm_ascend.ops.gdn.maybe_save_kv_layer_to_connector"),
    ):
        AscendGatedDeltaNetAttention._forward_core(
            layer,
            torch.zeros((3, 2)),
            torch.zeros((3, 1)),
            torch.zeros((3, 1)),
            core_attn_out,
        )

    torch.testing.assert_close(core_attn_out[0], torch.full_like(core_attn_out[0], 10))
    torch.testing.assert_close(core_attn_out[1:], torch.full_like(core_attn_out[1:], 20))
    torch.testing.assert_close(conv_state[:2], torch.ones_like(conv_state[:2]))
    torch.testing.assert_close(conv_state[2], torch.zeros_like(conv_state[2]))
    torch.testing.assert_close(ssm_state[0], torch.ones_like(ssm_state[0]))
    torch.testing.assert_close(ssm_state[1], torch.full_like(ssm_state[1], 2))
