# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run production BFG submission/packing on CPU with NPU streams simulated.

This verifies placement and tensor lifetimes, not NPU concurrency or latency.
Run with --confcutdir=tests/ut/ops.
"""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from einops import rearrange

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def runtime(monkeypatch):
    events: list[Any] = []
    current = ["main"]

    class Stream:
        def __init__(self, name):
            self.name = name

        def record_event(self):
            event = f"{self.name}:event:{len(events)}"
            events.append(event)
            return event

        def wait_event(self, event):
            events.append(f"{self.name}:wait:{event}")

    main, aux = Stream("main"), Stream("aux")

    @contextmanager
    def stream(selected):
        old = current[0]
        current[0] = selected.name
        yield
        current[0] = old

    sigmoid = torch.Tensor.sigmoid

    def tracked_sigmoid(tensor):
        events.append(("sigmoid", current[0]))
        return sigmoid(tensor)

    monkeypatch.setattr(torch.Tensor, "sigmoid", tracked_sigmoid)
    contiguous = torch.Tensor.contiguous

    def copy_if_needed(tensor, *args, **kwargs):
        events.append(("contiguous", current[0], tensor.data_ptr(), tensor.is_contiguous(), tuple(tensor.shape)))
        return contiguous(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "contiguous", copy_if_needed)
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda tensor, selected: events.append(("record", tensor.data_ptr(), selected.name)),
    )
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda: main), raising=False)
    scope = dict(torch=torch, rearrange=rearrange, npu_stream_switch=stream, _kda_bfg_stream=lambda: aux)
    tree = ast.parse((ROOT / "vllm_ascend/ops/kimi_kda.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendKimiK3DeltaAttention")
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in ("_postprocess_bfg", "_run_overlapped_qkv_bfg", "_forward")
    ]
    for method in methods:
        method.decorator_list = []
    recurrent_tree = ast.parse((ROOT / "vllm_ascend/ops/kda.py").read_text())
    recurrent = next(n for n in recurrent_tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_recurrent_kda")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, *methods, recurrent], type_ignores=[])), "kda.py", "exec"
        ),
        scope,
    )
    model = SimpleNamespace(head_dim=128, _fused_bfg_output_sizes=(12, 1536, 1536), uses_mixed_projection=True)
    model._postprocess_bfg = lambda packed: scope["_postprocess_bfg"](model, packed)
    scope["recurrent_kda"] = lambda q, k, v, gate, *args, **kwargs: (gate, None)
    return SimpleNamespace(model=model, scope=scope, events=events, current=current)


def consume_gate(runtime, gate, beta):
    q = torch.empty_like(gate)
    return runtime.scope["run_recurrent_kda"](
        q,
        q,
        q,
        gate,
        beta,
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.zeros(12),
        torch.zeros(1536),
        lower_bound=-5.0,
    )


@pytest.mark.parametrize("tokens", [1, 4, 128])
@pytest.mark.parametrize("quant_tuple", [False, True])
def test_aux_bfg_main_quant_and_gate_packing_after_sigmoid(runtime, tokens, quant_tuple):
    model = runtime.model
    hidden = torch.randn(tokens, 8, dtype=torch.bfloat16)
    for iteration in range(2):
        packed = torch.randn(tokens, 3084, dtype=torch.bfloat16) + iteration
        quantized = (torch.empty_like(hidden), torch.ones(tokens, 1)) if quant_tuple else hidden

        def project(value, packed=packed):
            assert value is hidden
            runtime.events.append(("bfg_matmul", runtime.current[0]))
            return packed

        def quantize(value, quantized=quantized):
            assert value is hidden
            runtime.events.append(("quantize", runtime.current[0]))
            return quantized

        def matmul(value, quantized=quantized):
            assert value is quantized
            runtime.events.append(("qkv_matmul", runtime.current[0]))
            return hidden

        model._project_bfg = project
        model._quantize_fused_qkv = quantize
        model._matmul_fused_qkv = matmul
        runtime.events.clear()
        _, beta, gate, output_gate = runtime.scope["_run_overlapped_qkv_bfg"](model, hidden)
        events = list(runtime.events)
        assert gate.is_contiguous()
        torch.testing.assert_close(gate, packed[:, 12:1548].reshape(1, tokens, 12, 128), rtol=0, atol=0)
        torch.testing.assert_close(output_gate, packed[:, 1548:].reshape(tokens, 12, 128), rtol=0, atol=0)
        torch.testing.assert_close(beta, packed[:, :12].float().sigmoid().unsqueeze(0), rtol=0, atol=0)
        bfg = events.index(("bfg_matmul", "aux"))
        quant = events.index(("quantize", "main"))
        qkv = events.index(("qkv_matmul", "main"))
        sigmoid = events.index(("sigmoid", "aux"))
        copies = [(i, event) for i, event in enumerate(events) if isinstance(event, tuple) and event[0] == "contiguous"]
        assert len(copies) == 1
        copy_index, copy = copies[0]
        assert bfg < quant < qkv < sigmoid < copy_index
        assert copy[1] == "aux" and copy[3] is (tokens == 1)
        assert copy[4] == (1, tokens, 12, 128)
        main_events = [event for event in events if isinstance(event, str) and event.startswith("main:event:")]
        aux_events = [event for event in events if isinstance(event, str) and event.startswith("aux:event:")]
        assert len(main_events) == len(aux_events) == 2
        assert events.index(main_events[0]) < events.index("aux:wait:" + main_events[0]) < bfg
        assert bfg < events.index(aux_events[0]) < quant < events.index(main_events[1])
        assert events.index(main_events[1]) < events.index("main:wait:" + aux_events[0]) < qkv
        assert qkv < events.index("aux:wait:" + main_events[1]) < sigmoid
        assert copy_index < events.index(aux_events[1]) < events.index("main:wait:" + aux_events[1])
        assert events[-1] == "main:wait:" + aux_events[1]
        assert ("record", hidden.data_ptr(), "aux") in events
        for tensor in (beta, gate, output_gate):
            assert ("record", tensor.data_ptr(), "main") in events
        before = len(runtime.events)
        result = consume_gate(runtime, gate, beta)
        assert result.data_ptr() == gate.data_ptr()
        assert all(
            event[3] for event in runtime.events[before:] if isinstance(event, tuple) and event[0] == "contiguous"
        )


def test_padded_and_mixed_gates_accept_full_projection_copy(runtime):
    # Deliberate simplicity tradeoff: no runtime metadata enters projections.
    # Packing covers capacity before downstream live slicing / index_select.
    packed = torch.randn(8, 3084, dtype=torch.bfloat16)
    _, gate, _ = runtime.scope["_postprocess_bfg"](runtime.model, packed)
    copy = next(event for event in runtime.events if isinstance(event, tuple) and event[0] == "contiguous")
    assert copy[4] == (1, 8, 12, 128)
    live = gate[:, :4]
    assert live.is_contiguous() and live.data_ptr() == gate.data_ptr()
    for indices in (torch.tensor([0, 2]), torch.tensor([1, 3])):
        selected = live.index_select(1, indices)
        assert selected.is_contiguous() and selected.data_ptr() != gate.data_ptr()
        expected = packed[:4, 12:1548].reshape(1, 4, 12, 128).index_select(1, indices)
        torch.testing.assert_close(selected, expected)


def test_non_bfg_caller_keeps_recurrent_gate_materialization(runtime):
    packed = torch.randn(4, 3084, dtype=torch.bfloat16)
    gate = packed[:, 12:1548].reshape(1, 4, 12, 128)
    assert not gate.is_contiguous()
    result = consume_gate(runtime, gate, torch.zeros(1, 4, 12))
    assert result.is_contiguous() and result.data_ptr() != gate.data_ptr()
    torch.testing.assert_close(result, gate, rtol=0, atol=0)
