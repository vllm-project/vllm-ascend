# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

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
