# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""Late WKV scheduling, SP row ownership and exceptional retirement."""

from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm.models.deepseek_v41")

from vllm_ascend.models.deepseek_v41 import model as model_mod
from vllm_ascend.models.deepseek_v41.engram.windowed import (
    EngramProjectionSchedule,
    router_projection_window,
    shard_engram_rows,
)
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner


@pytest.fixture
def window_runtime(monkeypatch):
    calls: list[tuple[object, ...]] = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            calls.append(("wait", self.name, event.name))

        def wait_stream(self, stream):
            calls.append(("join", self.name, stream.name))

    class Event:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            calls.append(("record", stream.name, self.name))

        def wait(self, stream):
            calls.append(("external_wait", stream.name, self.name))

        def reset(self, stream):
            calls.append(("reset", stream.name, self.name))

    main, side = Stream("main"), Stream("side")
    current = [main]

    @contextmanager
    def on_stream(stream):
        old = current[0]
        current[0] = stream
        try:
            yield
        finally:
            current[0] = old

    monkeypatch.setattr(torch.npu, "current_stream", lambda: current[0])
    monkeypatch.setattr(torch.npu, "stream", on_stream)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: calls.append(("retain", stream.name)))
    return calls, side, Event


def make_projection(runtime, failure=None):
    calls, side, event = runtime
    rows = torch.arange(24, dtype=torch.bfloat16).view(6, 4) / 8
    weight = torch.arange(24, dtype=torch.bfloat16).view(6, 4) / 16
    inputs = {1: rows, 7: rows + 2}
    outputs = {target: torch.full((5, 6), float("nan"), dtype=torch.bfloat16) for target in inputs}

    def shard(values):
        calls.append(("shard",))
        return values[1::2]

    def project(values, target):
        calls.append(("project", target))
        if target == failure:
            raise RuntimeError("projection failed")
        return torch.nn.functional.linear(values, weight)

    layers = {target: SimpleNamespace(project=partial(project, target=target)) for target in inputs}
    projection = EngramProjectionSchedule(
        layers,
        inputs,
        outputs,
        side,
        {target: event(str(target)) for target in inputs},
        tokens=3,
        shard=shard,
        lookup_ready={target: event(f"rows{target}") for target in inputs},
        external=True,
    )
    expected = {target: torch.nn.functional.linear(values[1::2], weight) for target, values in inputs.items()}
    return projection, expected


def test_windowed_projection_waits_once_and_uses_only_sp_rows(window_runtime):
    projection, expected = make_projection(window_runtime)
    calls = window_runtime[0]
    with projection:
        projection.submit_router(0)
        first = projection.finish(1)
        projection.submit_router(1)
        late = projection.finish(7)
    for target, actual in ((1, first), (7, late)):
        torch.testing.assert_close(actual, expected[target], rtol=0, atol=0)
        assert torch.isnan(projection.outputs[target][3:]).all()
        assert calls.count(("external_wait", "side", f"rows{target}")) == 1
        assert calls.count(("reset", "side", f"rows{target}")) == 1
        assert calls.index(("reset", "side", f"rows{target}")) < calls.index(("project", target))
    assert calls.count(("shard",)) == 2
    assert calls.count(("join", "main", "side")) == 2


def test_skipped_router_retains_original_lookup_for_fallback(window_runtime):
    projection, expected = make_projection(window_runtime)
    with projection:
        projection.submit_router(0)
        torch.testing.assert_close(projection.finish(1), expected[1], rtol=0, atol=0)
        assert not projection.ready_for(7)
        with pytest.raises(RuntimeError, match="not submitted"):
            projection.finish(7)
    assert ("external_wait", "side", "rows7") not in window_runtime[0]


def test_no_window_keeps_original_lookup_event_unconsumed(window_runtime):
    projection, _ = make_projection(window_runtime)
    with projection:
        projection.submit_router(99)
    assert not projection.ready_for(1)
    assert not window_runtime[0]


@pytest.mark.parametrize("failure", [1, 7])
def test_projection_error_joins_even_before_first_piece_completes(window_runtime, failure):
    projection, _ = make_projection(window_runtime, failure)
    with pytest.raises(RuntimeError, match="projection failed"), projection:
        projection.submit_router(0)
        projection.finish(1)
        projection.submit_router(1)
    assert window_runtime[0][-1] == ("join", "main", "side")


def test_duplicate_router_notifications_project_only_once(window_runtime):
    projection, _ = make_projection(window_runtime)
    with projection:
        projection.submit_router(0)
        projection.submit_router(0)
    assert window_runtime[0].count(("project", 1)) == 1


def test_decoder_passes_window_to_moe_after_attention(monkeypatch):
    calls = []
    layer = SimpleNamespace(use_sequence_parallel=True)
    for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"):
        setattr(layer, name, None)
    layer.hc_pre = lambda x, *args: (x.mean(1), None, None, None)
    layer.input_layernorm = lambda x: x

    def self_attn(*args):
        calls.append("attention")
        return args[1]

    def hc_post(x, residual, *args):
        calls.append("hc_post")
        return residual

    def reduce_scatter(x):
        calls.append("reduce_scatter")
        return x

    layer.self_attn = self_attn
    layer.hc_post = hc_post
    layer.rms_norm_cast = lambda x: (x, x.float())

    def moe(x, **kwargs):
        calls.append("moe")
        kwargs["engram_wkv_window"]()
        return x

    layer.mlp = moe
    monkeypatch.setattr(model_mod, "sp_all_gather", lambda x: x)
    monkeypatch.setattr(model_mod, "sp_reduce_scatter", reduce_scatter)
    hidden = torch.ones(3, 4, 8)
    result, _ = model_mod.DeepseekV41DecoderLayer.forward(
        layer, torch.arange(3), hidden, None, engram_wkv_window=lambda: calls.append("window")
    )
    assert calls[:3] == ["attention", "reduce_scatter", "hc_post"]
    assert calls.index("moe") < calls.index("window") < len(calls) - 1
    torch.testing.assert_close(result, hidden)


@pytest.mark.parametrize("precast", [False, True])
@pytest.mark.parametrize("method", ["_compute_router_logits", "_apply_router_linear"])
def test_runner_notifies_only_after_gate_matmul(precast, method):
    calls = []
    gate = torch.nn.Linear(4, 6, bias=False, dtype=torch.float32)
    gate.register_forward_hook(lambda *args: calls.append("gate"))
    if precast:
        gate.weight_fp32 = gate.weight
    runner = SimpleNamespace(gate=gate, _engram_wkv_window=lambda: calls.append("window"))
    runner._notify_router_ready = lambda output: AscendMoERunner._notify_router_ready(runner, output)
    inputs = torch.ones(3, 4)
    with torch.inference_mode():
        if method == "_compute_router_logits":
            output = AscendMoERunner._compute_router_logits(runner, inputs, inputs)
        else:
            output = AscendMoERunner._apply_router_linear(runner, inputs)
        expected = torch.nn.functional.linear(inputs, gate.weight)
    torch.testing.assert_close(output, expected)
    assert calls == (["window"] if precast else ["gate", "window"])


@pytest.mark.parametrize("failed", [False, True])
def test_router_callback_scope_restores_previous_value(failed):
    original, active = object(), object()
    experts = SimpleNamespace(_engram_wkv_window=original)
    try:
        with router_projection_window(experts, active):
            assert experts._engram_wkv_window is active
            if failed:
                raise RuntimeError("router failed")
    except RuntimeError:
        assert failed
    assert experts._engram_wkv_window is original


@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_moe_window_is_after_router_and_restored_on_return(monkeypatch, internal, failed):
    calls = []
    linear = torch.nn.functional.linear

    def router(*args):
        calls.append("router")
        return linear(*args)

    monkeypatch.setattr(model_mod.F, "linear", router)
    gate = SimpleNamespace(weight=torch.ones(6, 8))

    class Experts:
        is_internal_router = internal
        _engram_wkv_window = None

        def __call__(self, hidden_states, router_logits, input_ids):
            if internal:
                result = router(router_logits, gate.weight)
                AscendMoERunner._notify_router_ready(self, result)
            calls.append("experts")
            if failed:
                raise RuntimeError("experts failed")
            return hidden_states

    experts = Experts()
    moe = SimpleNamespace(experts=experts, gate=gate, is_sequence_parallel=False, tp_size=1)
    hidden = torch.ones(3, 8)
    if failed:
        with pytest.raises(RuntimeError, match="experts failed"):
            model_mod.DeepseekV41MoE.forward(moe, hidden, engram_wkv_window=lambda: calls.append("window"))
    else:
        actual = model_mod.DeepseekV41MoE.forward(moe, hidden, engram_wkv_window=lambda: calls.append("window"))
        torch.testing.assert_close(actual, hidden)
    assert calls == ["router", "window", "experts"]
    assert experts._engram_wkv_window is None


@pytest.mark.parametrize("count", [0, 3, 8])
@pytest.mark.parametrize("size", [2, 4])
def test_frozen_tp_shard_matches_padding_contract(count, size):
    rows = torch.arange(count * 4).view(count, 4)
    padding = (-count) % size
    padded = torch.cat((rows, rows.new_zeros((padding, 4)))) if padding else rows
    for rank in range(size):
        actual = shard_engram_rows(rows, rank=rank, size=size)
        expected = padded.chunk(size)[rank] if count else rows
        torch.testing.assert_close(actual, expected)
