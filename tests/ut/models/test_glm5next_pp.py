# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline-parallel contract of the GLM-5.3-Flash text model.

An mHC decoder layer defers its ``hc_post`` state (``post`` / ``comb``) to the
next layer. Across a PP boundary that state must travel with the intermediate
tensors: dropping it would make multi-stage execution differ from the
single-stage math. These tests pin the payload key set (which must match
``make_empty_intermediate_tensors``) and the hand-off between two stages.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.glm5next import model as glm5next_model
from vllm_ascend.models.glm5next.model import Glm5NextModel

HIDDEN_SIZE = 8
N_STREAMS = 4


def _stub_model(*, mhc: bool) -> Glm5NextModel:
    # The PP plumbing only reads ``config`` plus the active layer slice, so a
    # stub avoids building the NPU-bound attention / MoE stack.
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        mhc=mhc,
        mhc_num_residual_streams=N_STREAMS,
    )
    model.is_sequence_parallel = False
    return model


def _pp_group(*, is_first_rank: bool, is_last_rank: bool):
    return lambda: SimpleNamespace(is_first_rank=is_first_rank, is_last_rank=is_last_rank, world_size=2)


class _DeferredStateLayer(nn.Module):
    """Emulates an mHC decoder layer.

    Records the deferred state it received and hands a fresh one to the next
    layer, mirroring ``Glm5NextDecoderLayer``'s 4-tuple contract.
    """

    def __init__(self) -> None:
        super().__init__()
        self.incoming: tuple[torch.Tensor | None, torch.Tensor | None] | None = None

    def forward(self, positions, hidden_states, residual, post, comb):
        del positions
        self.incoming = (post, comb)
        return (
            hidden_states + 1,
            residual,
            torch.full((hidden_states.shape[0], N_STREAMS, 1), 2.0),
            torch.full((hidden_states.shape[0], N_STREAMS, N_STREAMS), 3.0),
        )


@pytest.mark.parametrize("mhc", [True, False])
def test_make_empty_intermediate_tensors_matches_pp_payload(mhc):
    model = _stub_model(mhc=mhc)

    tensors = model.make_empty_intermediate_tensors(batch_size=5, dtype=torch.bfloat16, device=torch.device("cpu"))

    assert tensors["hidden_states"].shape == (5, HIDDEN_SIZE)
    assert tensors["hidden_states"].dtype == torch.bfloat16
    if mhc:
        # The per-token stream stays 2D; the hyper-connection streams live in
        # ``residual`` and the deferred state is FP32 (hc_pre kernel output).
        assert tensors["residual"].shape == (5, N_STREAMS, HIDDEN_SIZE)
        assert tensors["post"].shape == (5, N_STREAMS, 1)
        assert tensors["comb"].shape == (5, N_STREAMS, N_STREAMS)
        assert tensors["post"].dtype == torch.float32
        assert tensors["comb"].dtype == torch.float32
    else:
        assert set(tensors.tensors) == {"hidden_states", "residual"}
        assert tensors["residual"].shape == (5, HIDDEN_SIZE)


@pytest.mark.parametrize("mhc", [True, False])
def test_non_last_rank_ships_state_the_next_rank_loads(monkeypatch, mhc):
    model = _stub_model(mhc=mhc)
    model._active_layers = [_DeferredStateLayer()]
    monkeypatch.setattr(glm5next_model, "get_pp_group", _pp_group(is_first_rank=True, is_last_rank=False))
    inputs_embeds = torch.zeros(3, HIDDEN_SIZE, dtype=torch.bfloat16)

    out = model(input_ids=None, positions=torch.arange(3), intermediate_tensors=None, inputs_embeds=inputs_embeds)

    # Next rank indexes these keys directly, so the shipped set must match the
    # placeholder factory key-for-key.
    expected_keys = {"hidden_states", "residual"} | ({"post", "comb"} if mhc else set())
    assert set(out.tensors) == expected_keys
    torch.testing.assert_close(out["hidden_states"], inputs_embeds + 1)
    if mhc:
        assert out["post"].dtype == torch.float32
        assert out["comb"].dtype == torch.float32


def test_non_first_rank_continues_deferred_mhc_state(monkeypatch):
    model = _stub_model(mhc=True)
    layer = _DeferredStateLayer()
    model._active_layers = [layer]
    model.norm = nn.Identity()
    monkeypatch.setattr(glm5next_model, "get_pp_group", _pp_group(is_first_rank=False, is_last_rank=True))
    hidden_states = torch.zeros(3, HIDDEN_SIZE, dtype=torch.bfloat16)
    residual = torch.zeros(3, N_STREAMS, HIDDEN_SIZE, dtype=torch.bfloat16)
    post = torch.ones(3, N_STREAMS, 1)
    comb = torch.ones(3, N_STREAMS, N_STREAMS)
    payload = IntermediateTensors({"hidden_states": hidden_states, "residual": residual, "post": post, "comb": comb})

    out = model(input_ids=None, positions=torch.arange(3), intermediate_tensors=payload)

    # The receiving first layer must run hc_post_pre on the previous rank's
    # state instead of falling back to a standalone pre.
    assert layer.incoming is not None
    assert layer.incoming[0] is post
    assert layer.incoming[1] is comb
    torch.testing.assert_close(out, hidden_states + 1)


def test_non_first_rank_without_mhc_reads_plain_payload(monkeypatch):
    model = _stub_model(mhc=False)
    layer = _DeferredStateLayer()
    model._active_layers = [layer]
    model.norm = nn.Identity()
    monkeypatch.setattr(glm5next_model, "get_pp_group", _pp_group(is_first_rank=False, is_last_rank=True))
    hidden_states = torch.zeros(3, HIDDEN_SIZE, dtype=torch.bfloat16)
    payload = IntermediateTensors(
        {"hidden_states": hidden_states, "residual": torch.zeros(3, HIDDEN_SIZE, dtype=torch.bfloat16)}
    )

    out = model(input_ids=None, positions=torch.arange(3), intermediate_tensors=payload)

    assert layer.incoming == (None, None)
    torch.testing.assert_close(out, hidden_states + 1)


class _StateDependentLayer(nn.Module):
    """CPU stand-in whose output depends on every PP payload tensor."""

    def __init__(self, index, mhc):
        super().__init__()
        self.index = index
        self.mhc = mhc

    def forward(self, positions, hidden_states, residual, post, comb):
        if residual is None:
            residual = hidden_states.clone()
            if self.mhc:
                residual = residual[:, None, :].expand(-1, N_STREAMS, -1).clone()
        if self.mhc:
            if post is not None:
                residual = residual + post + comb.sum(dim=-1, keepdim=True)
            hidden_states = hidden_states + residual.sum(dim=1)
            post = hidden_states.new_full((len(positions), N_STREAMS, 1), self.index + 1).float()
            comb = hidden_states.new_full((len(positions), N_STREAMS, N_STREAMS), self.index + 2).float()
        else:
            hidden_states = hidden_states + residual
        return hidden_states, residual + self.index + 1, post, comb


@pytest.mark.parametrize("mhc", [True, False])
@pytest.mark.parametrize("num_tokens", [1, 5])
@pytest.mark.parametrize("partition", [(3, 3), (1, 2, 3), (1, 2, 1, 2)])
def test_pipeline_matches_single_stage_for_state_dependent_layers(monkeypatch, mhc, num_tokens, partition):
    positions = torch.arange(num_tokens)
    embeddings = torch.arange(num_tokens * HIDDEN_SIZE, dtype=torch.float32).reshape(num_tokens, HIDDEN_SIZE)
    layers = [_StateDependentLayer(i, mhc) for i in range(sum(partition))]
    model = _stub_model(mhc=mhc)
    model._active_layers = layers
    model.norm = nn.Identity()
    monkeypatch.setattr(glm5next_model, "get_pp_group", _pp_group(is_first_rank=True, is_last_rank=True))
    expected = model(None, positions, None, inputs_embeds=embeddings)

    payload = None
    start = 0
    for rank, count in enumerate(partition):
        stage = _stub_model(mhc=mhc)
        stage._active_layers = layers[start : start + count]
        stage.norm = nn.Identity()
        last = rank == len(partition) - 1
        monkeypatch.setattr(glm5next_model, "get_pp_group", _pp_group(is_first_rank=rank == 0, is_last_rank=last))
        output = stage(None, positions, payload, inputs_embeds=embeddings if rank == 0 else None)
        if not last:
            buffers = stage.make_empty_intermediate_tensors(num_tokens, embeddings.dtype, embeddings.device)
            assert set(output.tensors) == set(buffers.tensors)
            for key, tensor in output.tensors.items():
                assert tensor.shape == buffers[key].shape
                assert tensor.dtype == buffers[key].dtype
                buffers[key].copy_(tensor)
            payload = buffers
        start += count

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
