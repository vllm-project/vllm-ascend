# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import vllm.envs as envs
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.config import CUDAGraphMode

from vllm_ascend.models.deepseek_v41 import decoder_replay_layers


@pytest.mark.parametrize("num_late_aux", [0, 2])
def test_decoder_replay_custom_op_returns_replay_tensors(monkeypatch, num_late_aux):
    hidden_states = torch.randn(4, 2, 8)
    pre_mix = torch.randn(4, 2)
    positions = torch.arange(4)
    input_ids = torch.arange(4)
    replay_outputs = [
        torch.randn_like(hidden_states),
        torch.randn_like(pre_mix),
        *(torch.randn(4, 8) for _ in range(num_late_aux)),
    ]

    class Replay:
        def __call__(self, *args):
            assert all(
                actual is expected
                for actual, expected in zip(args, (hidden_states, pre_mix, positions, input_ids), strict=True)
            )
            return tuple(replay_outputs)

    forward_context = SimpleNamespace(no_compile_layers={"decoder_replay": Replay()})
    monkeypatch.setattr(decoder_replay_layers, "get_forward_context", lambda: forward_context)

    outputs = decoder_replay_layers.dsv41_decoder_replay_forward(
        hidden_states,
        pre_mix,
        positions,
        input_ids,
        num_late_aux,
        "decoder_replay",
    )

    assert len(outputs) == len(replay_outputs)
    assert all(actual is expected for actual, expected in zip(outputs, replay_outputs, strict=True))


def test_decoder_replay_custom_op_checks_auxiliary_output_count(monkeypatch):
    class Replay:
        def __call__(self, *args):
            return torch.empty(4, 2, 8), torch.empty(4, 2)

    forward_context = SimpleNamespace(no_compile_layers={"decoder_replay": Replay()})
    monkeypatch.setattr(decoder_replay_layers, "get_forward_context", lambda: forward_context)

    with pytest.raises(RuntimeError, match="expected 1"):
        decoder_replay_layers.dsv41_decoder_replay_forward(
            torch.empty(4, 2, 8),
            torch.empty(4, 2),
            torch.arange(4),
            None,
            1,
            "decoder_replay",
        )


@pytest.mark.parametrize("num_late_aux", [0, 2])
def test_decoder_replay_fake_outputs_have_expected_shapes(num_late_aux):
    hidden_states = torch.randn(4, 2, 8)
    pre_mix = torch.randn(4, 2)
    positions = torch.arange(4)

    outputs = decoder_replay_layers.dsv41_decoder_replay_forward_fake(
        hidden_states,
        pre_mix,
        positions,
        None,
        num_late_aux,
        "decoder_replay",
    )

    assert [output.shape for output in outputs] == [
        hidden_states.shape,
        pre_mix.shape,
        *((positions.shape[0], hidden_states.shape[-1]) for _ in range(num_late_aux)),
    ]


def test_registered_decoder_replay_op_supports_fake_tensor_outputs():
    with FakeTensorMode() as fake_mode:
        hidden_states = fake_mode.from_tensor(torch.randn(4, 2, 8))
        pre_mix = fake_mode.from_tensor(torch.randn(4, 2))
        positions = fake_mode.from_tensor(torch.arange(4))
        input_ids = fake_mode.from_tensor(torch.arange(4))

        outputs = torch.ops.vllm.dsv41_decoder_replay_forward(
            hidden_states,
            pre_mix,
            positions,
            input_ids,
            2,
            "decoder_replay",
        )

    assert [output.shape for output in outputs] == [(4, 2, 8), (4, 2), (4, 8), (4, 8)]


@pytest.mark.parametrize("num_late_aux", [0, 2])
def test_piecewise_replay_reuses_outputs_and_reads_current_request(monkeypatch, num_late_aux):
    hidden = torch.zeros(4, 2, 8)
    mix = torch.zeros(4, 2)
    positions = torch.arange(4)
    outputs = [torch.empty_like(hidden), torch.empty_like(mix), *(torch.empty(4, 8) for _ in range(num_late_aux))]
    pointers = [tensor.data_ptr() for tensor in outputs]
    context = SimpleNamespace(no_compile_layers={})
    monkeypatch.setattr(decoder_replay_layers, "get_forward_context", lambda: context)

    # A captured eager segment reuses its callable and outputs, while the
    # current forward context and replay-row selection change each step.
    def replay():
        decoder_replay_layers.dsv41_decoder_replay_forward_with_output(
            hidden, mix, positions, None, outputs, "decoder_replay"
        )

    for value in [7, 19]:
        fresh = [torch.full_like(tensor, value) for tensor in outputs]
        context.no_compile_layers["decoder_replay"] = lambda *args, fresh=fresh: tuple(fresh)
        replay()
        assert [tensor.data_ptr() for tensor in outputs] == pointers
        for actual, expected in zip(outputs, fresh, strict=True):
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
def test_piecewise_model_route_selects_fixed_output_operator(monkeypatch, mode):
    context = SimpleNamespace(cudagraph_runtime_mode=mode)
    monkeypatch.setattr(decoder_replay_layers, "get_forward_context", lambda: context)
    monkeypatch.setattr(decoder_replay_layers, "is_forward_context_available", lambda: True, raising=False)
    seen = []

    def inplace(hidden, mix, positions, input_ids, outputs, layer_name):
        seen.append(layer_name)
        for output in outputs:
            output.fill_(23)

    monkeypatch.setattr(torch.ops.vllm, "dsv41_decoder_replay_forward_with_output", inplace, raising=False)
    outputs = decoder_replay_layers.run_decoder_replay(
        torch.zeros(4, 2, 8), torch.zeros(4, 2), torch.arange(4), None, 2, "decoder_replay", use_piecewise=True
    )
    assert seen == ["decoder_replay"]
    assert [tensor.shape for tensor in outputs] == [(4, 2, 8), (4, 2), (4, 8), (4, 8)]
    assert all(torch.all(tensor == 23) for tensor in outputs)


def test_piecewise_warmup_trace_keeps_fixed_output_op_without_runtime_guards(monkeypatch):
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE, no_compile_layers={})
    monkeypatch.setattr(decoder_replay_layers, "get_forward_context", lambda: context)
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    def forward(hidden, mix, positions):
        return decoder_replay_layers.run_decoder_replay(
            hidden, mix, positions, None, 0, "decoder_replay", use_piecewise=True
        )

    library = torch.library.Library("vllm", "IMPL", "CPU")
    library.impl(
        "dsv41_decoder_replay_forward_with_output", decoder_replay_layers.dsv41_decoder_replay_forward_with_output
    )
    try:
        compiled = torch.compile(forward, backend=backend, fullgraph=True)
        hidden, mix, positions = torch.zeros(4, 2, 8), torch.zeros(4, 2), torch.arange(4)
        context.no_compile_layers["decoder_replay"] = lambda *args: (hidden + 7, mix + 7)
        outputs = compiled(hidden, mix, positions)
        assert all(torch.all(tensor == 7) for tensor in outputs)
        assert len(graphs) == 1
        assert any(
            node.target
            in (
                torch.ops.vllm.dsv41_decoder_replay_forward_with_output,
                torch.ops.vllm.dsv41_decoder_replay_forward_with_output.default,
            )
            for node in graphs[0].graph.nodes
        )
        context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        context.no_compile_layers["decoder_replay"] = lambda *args: (hidden + 19, mix + 19)
        # vLLM uses this graph directly without checking the NONE warmup's
        # Python guards. Every opaque split op must read the current context.
        outputs = graphs[0](hidden, mix, positions)
        assert all(torch.all(tensor == 19) for tensor in outputs)
    finally:
        library._destroy()
        torch._dynamo.reset()


def test_registered_inplace_decoder_replay_supports_fake_tensor_outputs():
    with FakeTensorMode() as fake_mode:
        hidden = fake_mode.from_tensor(torch.randn(4, 2, 8))
        mix = fake_mode.from_tensor(torch.randn(4, 2))
        positions = fake_mode.from_tensor(torch.arange(4))
        outputs = [torch.empty_like(hidden), torch.empty_like(mix), hidden.new_empty((4, 8))]
        result = torch.ops.vllm.dsv41_decoder_replay_forward_with_output(
            hidden, mix, positions, None, outputs, "decoder_replay"
        )
    assert result is None
    assert [tensor.shape for tensor in outputs] == [(4, 2, 8), (4, 2), (4, 8)]


@pytest.mark.skipif(not envs.VLLM_USE_BREAKABLE_CUDAGRAPH, reason="Requires import-time breakable graph opt-in")
def test_breakable_replay_flattens_buffers_and_keeps_nested_hooks_current(monkeypatch):
    from vllm.compilation import breakable_cudagraph as breakable

    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE, no_compile_layers={})
    monkeypatch.setattr(decoder_replay_layers, "get_forward_context", lambda: context)
    monkeypatch.setattr(breakable, "get_forward_context", lambda: context)
    monkeypatch.setattr(breakable, "is_forward_context_available", lambda: True)
    weak_args = []
    monkeypatch.setattr(breakable, "_weak_ref_capture_arg", lambda value: (weak_args.append(value), value)[1])
    events = []

    class Capture:
        _capturing = True
        segments = []

        def add_eager(self, callback):
            self.segments.append(callback)
            self._capturing = False
            try:
                return callback()
            finally:
                self._capturing = True

    capture = Capture()
    monkeypatch.setattr(breakable.BreakableCUDAGraphCapture, "current", lambda: capture)

    @breakable.eager_break_during_capture
    def nested_attention():
        events.append(context.request)

    hidden, mix, positions = torch.zeros(4, 2, 8), torch.zeros(4, 2), torch.arange(4)
    outputs = [torch.empty_like(hidden), torch.empty_like(mix)]

    def replay(*args):
        nested_attention()
        return hidden + context.request, mix + context.request

    context.no_compile_layers["decoder_replay"] = replay
    context.request = 7
    decoder_replay_layers.dsv41_decoder_replay_forward_with_output(
        hidden, mix, positions, None, outputs, "decoder_replay"
    )
    assert len(capture.segments) == 1
    assert not any(isinstance(value, (list, tuple)) for value in weak_args)
    assert all(any(value is output for value in weak_args) for output in outputs)
    context.request = 19
    capture._capturing = False
    capture.segments[0]()
    assert events == [7, 19]
    assert all(torch.all(output == 19) for output in outputs)
