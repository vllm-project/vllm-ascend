# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Auxiliary hidden state capture for GLM-5.3-Flash.

EAGLE3 and DFlash/DFlash2 drafters consume the target's intermediate layer
outputs (``dflash_config.target_layer_ids``), concatenated into the drafter's
``fc``. GLM-5.3-Flash carries an mHC hyper-connection residual stream and defers
each layer's ``hc_post`` to the next layer's fused pre, so the value at a layer
boundary has to be materialized before it can be handed to a drafter.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm_ascend.models.glm5next.model import Glm5NextDecoderLayer, Glm5NextModel

PP_LAST_RANK = SimpleNamespace(is_first_rank=True, is_last_rank=True)


def _make_layer(layer_idx: int, n: int = 2) -> Glm5NextDecoderLayer:
    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_idx = layer_idx
    layer.n = n
    # Stands in for the mHC post kernel: any pure combination of the four
    # inputs is enough to show the deferred state is what gets materialized.
    layer.hc_post = lambda x, residual, post, comb: x + residual + post + comb
    return layer


class TestContractLayerOutput:
    def test_materializes_deferred_mhc_state(self):
        layer = _make_layer(3)
        # [num_tokens, num_residual_streams, hidden_size]
        hidden_states = torch.arange(12.0).reshape(2, 2, 3)
        residual = torch.ones_like(hidden_states)
        post = torch.full_like(hidden_states, 2.0)
        comb = torch.full_like(hidden_states, 3.0)

        contracted = layer.contract_layer_output(hidden_states, residual, post, comb)

        # hc_post, then the mean over the residual streams (hc_contract).
        torch.testing.assert_close(contracted, (hidden_states + 6.0).mean(dim=1))
        assert contracted.shape == (2, 3)

    @pytest.mark.parametrize(
        "residual, post, comb",
        [
            # Non-mHC layer and the final mHC layer: already summed and contracted.
            (None, None, None),
            # Defensive: a partially missing deferred state must not be combined.
            (torch.ones(2, 3), None, None),
            (torch.ones(2, 3), torch.ones(2, 3), None),
        ],
    )
    def test_passes_through_already_contracted_output(self, residual, post, comb):
        layer = _make_layer(3)
        hidden_states = torch.arange(6.0).reshape(2, 3)

        assert layer.contract_layer_output(hidden_states, residual, post, comb) is hidden_states


class _StubLayer:
    """Replaces a decoder layer body with an index marker so that each capture
    is traceable, while keeping the deferred mHC state threaded through."""

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx

    def __call__(self, positions, hidden_states, residual, post, comb):
        return hidden_states + self.layer_idx, residual, post, comb

    def contract_layer_output(self, hidden_states, residual, post, comb):
        # Distinguishable from the raw layer output so a missing contraction fails.
        return hidden_states * 2


def _make_model(num_layers: int, aux_layers: tuple[int, ...], is_sequence_parallel: bool = False) -> Glm5NextModel:
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model._active_layers = [_StubLayer(i) for i in range(num_layers)]
    model.is_sequence_parallel = is_sequence_parallel
    model.norm = lambda hidden_states: hidden_states * 10
    model._set_aux_hidden_state_layers(aux_layers)
    return model


def _run_forward(model: Glm5NextModel, num_tokens: int = 3, hidden_size: int = 5):
    inputs_embeds = torch.zeros(num_tokens, hidden_size)
    with patch("vllm_ascend.models.glm5next.model.get_pp_group", return_value=PP_LAST_RANK):
        return model.forward(None, torch.arange(num_tokens), None, inputs_embeds=inputs_embeds)


class TestForwardAuxCapture:
    def test_returns_bare_tensor_when_no_drafter_requested_aux(self):
        """The common case: no EAGLE3/DFlash drafter, so no tuple and no extra work."""
        model = _make_model(num_layers=4, aux_layers=())

        hidden_states = _run_forward(model)

        assert isinstance(hidden_states, torch.Tensor)
        torch.testing.assert_close(hidden_states, torch.full((3, 5), 60.0))

    def test_captures_requested_layers_in_order(self):
        # Index i means "the stream after i layers", so 0 is the embedding output
        # and 4 is the output of the last of the four layers.
        model = _make_model(num_layers=4, aux_layers=(0, 2, 4))

        hidden_states, aux_hidden_states = _run_forward(model)

        # Layer i adds i, so the running sums are 0, 0, 1, 3, 6.
        expected = [
            torch.zeros(3, 5),  # embedding output, captured as-is
            torch.full((3, 5), 2.0),  # after 2 layers: 1, contracted
            torch.full((3, 5), 12.0),  # after 4 layers: 6, contracted
        ]
        assert len(aux_hidden_states) == len(expected)
        for captured, want in zip(aux_hidden_states, expected):
            torch.testing.assert_close(captured, want)
        # Auxiliary states stay un-normalized; only the final output is normed.
        torch.testing.assert_close(hidden_states, torch.full((3, 5), 60.0))

    def test_captures_are_all_gathered_under_sequence_parallelism(self):
        """Captures run on this rank's SP shard, so they need the same all-gather
        and trim the final hidden states get; otherwise the drafter's ``fc``
        receives a slice of the batch."""
        model = _make_model(num_layers=2, aux_layers=(1, 2), is_sequence_parallel=True)
        num_tokens = 3

        def fake_all_gather(tensor):
            # Mimics a 2-rank gather: padded beyond num_tokens, hence the trim.
            return torch.cat([tensor, torch.full_like(tensor, -1.0)], dim=0)

        with (
            patch("vllm_ascend.models.glm5next.model.get_pp_group", return_value=PP_LAST_RANK),
            patch("vllm_ascend.models.glm5next.model.sp_shard", side_effect=lambda t: t),
            patch("vllm_ascend.models.glm5next.model.sp_all_gather", side_effect=fake_all_gather),
        ):
            _, aux_hidden_states = model.forward(
                None,
                torch.arange(num_tokens),
                None,
                inputs_embeds=torch.zeros(num_tokens, 5),
            )

        for captured in aux_hidden_states:
            assert captured.shape == (num_tokens, 5)
            assert not (captured == -1.0).any()
