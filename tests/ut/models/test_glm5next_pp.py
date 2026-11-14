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
from vllm.config.compilation import CompilationMode
from vllm.distributed import parallel_state
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.pp_utils import PPHandler
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

from vllm_ascend.models.glm5next import model as glm5next_model
from vllm_ascend.models.glm5next.model import Glm5NextModel

HIDDEN_SIZE = 8
N_STREAMS = 4


def _stub_model(*, mhc: bool) -> Glm5NextModel:
    # The PP plumbing only reads ``config`` plus the active layer slice, so a
    # stub avoids building the NPU-bound attention / MoE stack.
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.do_not_compile = True
    model.config = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        mhc=mhc,
        hc_mult=N_STREAMS,
    )
    model.is_sequence_parallel = False
    model.start_layer = 0
    model.end_layer = 1
    model.aux_hidden_state_layers = ()
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


@pytest.mark.parametrize(
    "pp_rank,method,draft_arch,has_embedding",
    [
        (0, None, None, True),
        (1, "dflash", "DFlash2DraftModel", False),
        (2, "dflash", "DFlash2DraftModel", True),
        (2, "dflash", "DFlashDraftModel", False),
        (2, "mtp", "DFlash2DraftModel", False),
        (2, None, None, False),
    ],
)
def test_dflash2_pp_embedding_is_loaded_by_target(monkeypatch, pp_rank, method, draft_arch, has_embedding):
    group = SimpleNamespace(is_first_rank=pp_rank == 0, is_last_rank=pp_rank == 2, world_size=3)
    monkeypatch.setattr(glm5next_model, "get_pp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: group)
    monkeypatch.setattr(eagle_utils, "get_pp_group", lambda: group)
    monkeypatch.setattr(glm5next_model, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(glm5next_model, "make_layers", lambda *args, **kwargs: (0, 0, nn.ModuleList()))
    monkeypatch.setattr(glm5next_model, "RMSNorm", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(
        glm5next_model, "VocabParallelEmbedding", lambda vocab, hidden, **kw: nn.Embedding(vocab, hidden)
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                vocab_size=4,
                hidden_size=HIDDEN_SIZE,
                index_topk=None,
                num_hidden_layers=3,
                num_attention_heads=1,
                rms_norm_eps=1e-5,
                n_routed_experts=None,
                num_nextn_predict_layers=0,
            )
        ),
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        speculative_config=(
            SimpleNamespace(method=method, draft_model_config=SimpleNamespace(architectures=[draft_arch]))
            if method is not None
            else None
        ),
    )
    target = Glm5NextModel(vllm_config=config)
    if not has_embedding:
        assert isinstance(target.embed_tokens, PPMissingLayer)
        return

    weight = torch.arange(4 * HIDDEN_SIZE, dtype=torch.float32).reshape(4, HIDDEN_SIZE)
    with torch.no_grad():
        assert target.load_weights([("embed_tokens.weight", weight)]) == {"embed_tokens.weight"}
    torch.testing.assert_close(target.embed_tokens.weight, weight)
    if group.is_last_rank:
        draft_inner = nn.Module()
        draft_inner.embed_tokens = nn.Embedding(4, HIDDEN_SIZE)
        eagle_utils.maybe_share_target_embed(SimpleNamespace(has_own_embed_tokens=False), draft_inner, target)
        assert draft_inner.embed_tokens is target.embed_tokens


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


@pytest.mark.parametrize("stage_sizes", [(3,), (1, 2), (1, 1, 1)])
def test_aux_hidden_states_follow_global_layer_order(monkeypatch, stage_sizes):
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    # A snapshot requested at a PP boundary includes the preceding layer's deferred mHC state.
    monkeypatch.setattr(
        Glm5NextModel,
        "_aux_hidden_state",
        lambda self, hidden, residual, post, comb: hidden if post is None else hidden + post[:, 0] + comb[:, 0, :1],
    )
    positions = torch.arange(3)
    inputs = torch.zeros(3, HIDDEN_SIZE)
    payload = None
    start = 0
    for rank, size in enumerate(stage_sizes):
        group = SimpleNamespace(
            is_first_rank=rank == 0,
            is_last_rank=rank == len(stage_sizes) - 1,
            world_size=len(stage_sizes),
        )
        monkeypatch.setattr(parallel_state, "get_pp_group", lambda group=group: group)
        monkeypatch.setattr(glm5next_model, "get_pp_group", lambda group=group: group)
        model = _stub_model(mhc=True)
        model.start_layer = start
        model.end_layer = start + size
        model._active_layers = [_DeferredStateLayer() for _ in range(size)]
        model.norm = nn.Identity()
        model._set_aux_hidden_state_layers((0, 1, 2, 3))
        target = SimpleNamespace(model=model)
        output = model(None, positions, payload, inputs_embeds=inputs)
        if not group.is_last_rank:
            handler = PPHandler.__new__(PPHandler)
            handler.configure_aux_hidden_state_relay(target)
            payload = handler.relay_aux_hidden_states(payload, output)
        start += size

    hidden, aux = output
    torch.testing.assert_close(hidden, inputs + 3)
    assert len(aux) == 4
    for snapshot, value in zip(aux, (0, 6, 7, 8)):
        torch.testing.assert_close(snapshot, inputs + value)
