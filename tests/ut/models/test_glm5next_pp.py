# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline-parallel contract of the GLM-5.3-Flash text model.

An mHC decoder layer defers its ``hc_post`` state (``post`` / ``comb``) to the
next layer. Across a PP boundary that state must travel with the intermediate
tensors: dropping it would make multi-stage execution differ from the
single-stage math. These tests pin the payload key set (which must match
``make_empty_intermediate_tensors``) and the hand-off between two stages.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from vllm.config import LoadConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.distributed import parallel_state
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.pp_utils import PPHandler
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import TargetDependentARSpeculator

from vllm_ascend.models.glm5next import model as glm5next_model
from vllm_ascend.models.glm5next import mtp as mtp_module
from vllm_ascend.models.glm5next.model import Glm5NextForCausalLM, Glm5NextForConditionalGeneration, Glm5NextModel
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator

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
    "pp_rank,draft_arch,has_embedding",
    [
        pytest.param(0, None, True, id="first-stage-loads"),
        pytest.param(1, "DFlash2DraftModel", False, id="middle-stage-skips-with-draft"),
        pytest.param(2, None, False, id="last-stage-skips-without-draft"),
        pytest.param(2, "DFlashDraftModel", True, id="last-stage-shares-with-dflash"),
        pytest.param(2, "DFlash2DraftModel", True, id="last-stage-shares-with-dflash2"),
    ],
)
def test_spec_decode_pp_embedding_is_loaded_by_target(monkeypatch, pp_rank, draft_arch, has_embedding):
    group = SimpleNamespace(is_first_rank=pp_rank == 0, is_last_rank=pp_rank == 2, world_size=3)
    monkeypatch.setattr(glm5next_model, "get_pp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: group)
    # Older vLLM pins import this helper directly; main uses in_single_pp_stage.
    if hasattr(eagle_utils, "get_pp_group"):
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
            # vLLM checks draft_model_config.may_use_target_embed (and uses_target_hidden_states).
            SimpleNamespace(
                method="dflash",
                uses_target_hidden_states=True,
                draft_model_config=SimpleNamespace(architectures=[draft_arch], may_use_target_embed=True),
            )
            if draft_arch is not None
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
        # The draft checkpoint has no embedding of its own. The upstream
        # sharing check must reuse the embedding loaded by the target.
        draft_inner = nn.Module()
        draft_inner.embed_tokens = nn.Embedding(4, HIDDEN_SIZE)
        eagle_utils.maybe_share_target_embed(SimpleNamespace(has_own_embed_tokens=False), draft_inner, target)
        assert draft_inner.embed_tokens is target.embed_tokens


def _make_pp2_mtp_target_and_configs(tmp_path):
    hf_text_config = SimpleNamespace(
        vocab_size=4,
        hidden_size=HIDDEN_SIZE,
        index_topk=None,
        num_hidden_layers=45,
        num_attention_heads=1,
        rms_norm_eps=1e-5,
        n_routed_experts=None,
        num_nextn_predict_layers=1,
    )
    load_config = LoadConfig(
        load_format="safetensors",
        model_loader_extra_config={"enable_weights_track": True},
    )
    target_model_config = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        quantization=None,
        multimodal_config=None,
        is_moe=False,
        hf_text_config=hf_text_config,
    )
    draft_hf_config = SimpleNamespace(
        num_hidden_layers=45,
        num_nextn_predict_layers=1,
        n_routed_experts=0,
        index_share_for_mtp_iteration=False,
    )
    draft_model_config = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        quantization=None,
        multimodal_config=None,
        is_moe=False,
        hf_config=draft_hf_config,
        hf_text_config=hf_text_config,
        may_use_target_embed=hasattr(mtp_module.Glm5NextMTP, "has_own_embed_tokens"),
    )
    target_config = SimpleNamespace(
        model_config=target_model_config,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            use_sequence_parallel_moe=False,
            enable_expert_parallel=False,
            enable_ep_weight_filter=False,
        ),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        load_config=load_config,
        quant_config=None,
        speculative_config=SimpleNamespace(
            method="mtp",
            uses_target_hidden_states=True,
            draft_model_config=draft_model_config,
            draft_load_config=None,
            draft_lm_head_quantization=None,
            apply_draft_overrides=lambda cfg: cfg,
        ),
    )
    draft_config = SimpleNamespace(
        model_config=draft_model_config,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_expert_parallel=False,
            enable_ep_weight_filter=False,
        ),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        load_config=load_config,
        quant_config=None,
        speculative_config=target_config.speculative_config,
    )
    target_inner = Glm5NextModel(vllm_config=target_config, prefix="language_model.model")
    causal_lm = Glm5NextForCausalLM.__new__(Glm5NextForCausalLM)
    nn.Module.__init__(causal_lm)
    causal_lm.config = hf_text_config
    causal_lm.model = target_inner
    target = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
    nn.Module.__init__(target)
    target.language_model = causal_lm
    target._language_model_names = ["language_model"]
    return target, causal_lm, target_inner, target_config, draft_config


def _setup_pp2_mtp_loader_env(monkeypatch):
    pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=True, world_size=2)
    single_pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True, world_size=1)
    monkeypatch.setattr(parallel_state, "_PP", pp_group)
    if not vllm_version_is("0.34.0"):
        monkeypatch.setattr(parallel_state, "_PP_SINGLE_STAGE", single_pp_group)
    monkeypatch.setattr(glm5next_model, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(glm5next_model, "make_layers", lambda *args, **kwargs: (0, 0, nn.ModuleList()))
    monkeypatch.setattr(glm5next_model, "RMSNorm", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(
        glm5next_model, "VocabParallelEmbedding", lambda vocab, hidden, **kw: nn.Embedding(vocab, hidden)
    )
    monkeypatch.setattr(mtp_module, "fused_moe_make_expert_params_mapping", lambda *args, **kwargs: [])


def _make_pp2_mtp_draft():
    draft = mtp_module.Glm5NextMTP.__new__(mtp_module.Glm5NextMTP)
    nn.Module.__init__(draft)
    draft.config = SimpleNamespace(num_hidden_layers=45, num_nextn_predict_layers=1, n_routed_experts=0)
    draft.is_rot_weight_used = False
    draft.model = nn.Module()
    draft.model.mtp_start_layer_idx = 45
    draft.model.num_mtp_layers = 1
    draft.model.embed_tokens = nn.Embedding(4, HIDDEN_SIZE)
    draft.model.embed_tokens.weight.data.fill_(float("nan"))
    layer = nn.Module()
    layer.enorm = nn.LayerNorm(HIDDEN_SIZE, bias=False)
    layer.enorm.weight.data.fill_(float("nan"))
    draft.model.layers = nn.ModuleDict({"45": layer})
    return draft


def test_pp2_private_mtp_embedding_releases_target_copy(monkeypatch, tmp_path):
    _setup_pp2_mtp_loader_env(monkeypatch)

    target_embedding = torch.arange(4 * HIDDEN_SIZE, dtype=torch.float32).reshape(4, HIDDEN_SIZE)
    private_embedding = torch.full((4, HIDDEN_SIZE), 5.0)
    mtp_norm = torch.full((HIDDEN_SIZE,), 3.0)
    save_file(
        {
            "model.language_model.embed_tokens.weight": target_embedding,
            "model.language_model.layers.45.enorm.weight": mtp_norm,
            "model.language_model.layers.45.embed_tokens.weight": private_embedding,
        },
        tmp_path / "model.safetensors",
    )

    target, _, target_inner, target_config, draft_config = _make_pp2_mtp_target_and_configs(tmp_path)
    assert isinstance(target_inner.embed_tokens, nn.Embedding)
    target_inner.embed_tokens.weight.data.fill_(float("nan"))

    target_loader = DefaultModelLoader(target_config.load_config)
    with set_current_vllm_config(target_config), torch.no_grad():
        target_loader.load_weights(target, target_config.model_config)
    torch.testing.assert_close(target_inner.embed_tokens.weight, target_embedding)

    draft = _make_pp2_mtp_draft()
    private_draft_embed = draft.model.embed_tokens

    def load_draft_from_disk(*, vllm_config, model_config):
        draft_loader = DefaultModelLoader(vllm_config.load_config)
        with set_current_vllm_config(vllm_config), torch.no_grad():
            draft_loader.load_weights(draft, model_config)
        return draft

    def load_model(speculator, target):
        speculator.model = speculator.load_draft_model(target, set())

    monkeypatch.setattr(eagle_utils, "get_model", load_draft_from_disk)
    monkeypatch.setattr(TargetDependentARSpeculator, "load_model", load_model)
    speculator = AscendMTPSpeculator.__new__(AscendMTPSpeculator)
    speculator.method = "mtp"
    speculator.vllm_config = target_config
    speculator.draft_vllm_config = draft_config
    draft_ctx = cast(Any, parallel_state).single_pp_stage() if not vllm_version_is("0.34.0") else nullcontext()
    with draft_ctx:
        if not vllm_version_is("0.34.0"):
            assert parallel_state.get_pp_group().world_size == 1
            assert parallel_state.get_pp_group().is_first_rank is True
        speculator.load_model(target)

    assert speculator.model is draft
    torch.testing.assert_close(draft.model.layers["45"].enorm.weight, mtp_norm)
    assert draft.has_own_embed_tokens is True
    assert draft.model.embed_tokens is private_draft_embed
    torch.testing.assert_close(draft.model.embed_tokens.weight, private_embedding)
    assert isinstance(target_inner.embed_tokens, PPMissingLayer)


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
        handler = PPHandler.__new__(PPHandler)
        handler.configure_aux_hidden_state_relay(target)
        if not group.is_last_rank:
            payload = handler.relay_aux_hidden_states(payload, output)
        elif hasattr(handler, "add_upstream_aux_hidden_states"):
            # Newer vLLM assembles the full list in the runner, while v0.33.0
            # collects earlier stages' snapshots inside the model forward.
            output = handler.add_upstream_aux_hidden_states(payload, output)
        start += size

    hidden, aux = output
    torch.testing.assert_close(hidden, inputs + 3)
    assert len(aux) == 4
    for snapshot, value in zip(aux, (0, 6, 7, 8)):
        torch.testing.assert_close(snapshot, inputs + value)
